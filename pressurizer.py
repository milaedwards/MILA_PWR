# pressurizer.py  –  AP1000 lumped pressurizer (NO CoolProp)
#
# Every PropsSI call has been replaced with:
#   • Wagner correlation   → T_sat(P), P_sat(T)       (< 1 K error 0.1-22 MPa)
#   • NIST saturation table + linear interp → rho_l, rho_v, h_l, h_v, cp_l
#   • Subcooled liquid: rho ≈ rho_l(P), h ≈ h_l(P) + cp_l*(T-T_sat)
#
# The physics, control bands, surge/flash/spray logic are IDENTICAL to the
# original CoolProp version.  Only the property look-up layer changed.

import math
import bisect
from config import Config as cfg

# ================================================================
# Water-property helper layer  (module-level, stateless)
# ================================================================

# --- Wagner saturation correlation constants (IAPWS) ---
_Tc  = 647.096   # K   critical temperature
_Pc  = 22.064e6  # Pa  critical pressure
_A1, _A2, _A3, _A4 = -7.85951783, 1.84408259, -11.7866497, 22.6807411
_A5, _A6           =  -15.9618719,  1.80122502

def _P_sat(T_K: float) -> float:
    """Saturation pressure [Pa] from temperature [K] (Wagner)."""
    if T_K <= 273.15:
        return 611.657          # triple point
    if T_K >= _Tc:
        return _Pc
    tau = 1.0 - T_K / _Tc
    Tr  = T_K / _Tc
    ln_Pr = (
        _A1*tau + _A2*tau**1.5 + _A3*tau**3
        + _A4*tau**3.5 + _A5*tau**4 + _A6*tau**7.5
    ) / Tr
    return _Pc * math.exp(ln_Pr)

def _T_sat(P_Pa: float) -> float:
    """Saturation temperature [K] from pressure [Pa] (bisection on Wagner)."""
    if P_Pa <= 611.657:
        return 273.15
    if P_Pa >= _Pc:
        return _Tc
    lo, hi = 273.15, _Tc
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _P_sat(mid) < P_Pa:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# --- NIST saturation table (pressure-indexed) ---
# Columns: (P_MPa, T_K, rho_l, rho_v, h_l, h_v, cp_l, s_l, s_v)
_SAT = [
    ( 0.10,  372.76,  958.4,    0.5978,  417.5e3,  2675.0e3,  4217, 1302.8,  7359.4),
    ( 0.50,  424.98,  917.0,    2.950,   639.7e3,  2748.7e3,  4181, 1860.5,  6821.3),
    ( 1.00,  453.03,  887.0,    5.974,   762.8e3,  2778.1e3,  4217, 2138.7,  6586.5),
    ( 2.00,  485.53,  852.0,   12.04,    908.8e3,  2799.5e3,  4350, 2447.0,  6340.1),
    ( 3.00,  507.43,  823.0,   18.65,   1008.4e3,  2804.2e3,  4540, 2645.0,  6185.2),
    ( 5.00,  536.67,  775.0,   32.55,   1154.2e3,  2794.3e3,  5000, 2921.9,  5973.4),
    ( 5.764,546.13,  757.0,   29.28,   1154.2e3,  2787.0e3,  5200, 2920.2,  5973.4),
    ( 7.00,  558.15,  740.0,   39.00,   1267.4e3,  2773.0e3,  5400, 3122.6,  5822.1),
    (10.00,  584.15,  688.4,   58.35,   1407.6e3,  2724.7e3,  5600, 3359.6,  5614.1),
    (12.00,  596.76,  654.0,   73.00,   1491.3e3,  2683.5e3,  6000, 3477.6,  5480.5),
    (15.00,  611.72,  603.0,  100.0,    1610.5e3,  2610.5e3,  6800, 3615.8,  5310.8),
    (15.513,617.15,  598.5,  109.6,    1610.5e3,  2611.0e3,  6500, 3684.8,  5309.8),
    (18.00,  629.99,  545.0,  140.0,    1740.0e3,  2508.0e3,  8000, 3790.0,  5110.0),
    (20.00,  638.90,  487.0,  182.0,    1826.3e3,  2409.7e3, 10000, 3870.0,  4930.0),
    (21.50,  644.49,  425.0,  240.0,    1930.0e3,  2294.0e3, 20000, 3970.0,  4740.0),
]
_SAT_P_Pa = [row[0] * 1e6 for row in _SAT]

def _sat_interp(P_Pa: float, col: int) -> float:
    """Linear interp on NIST sat table. col: 2=rho_l 3=rho_v 4=h_l 5=h_v 6=cp_l 7=s_l 8=s_v"""
    if P_Pa <= _SAT_P_Pa[0]:
        return _SAT[0][col]
    if P_Pa >= _SAT_P_Pa[-1]:
        return _SAT[-1][col]
    i = bisect.bisect_right(_SAT_P_Pa, P_Pa) - 1
    i = min(i, len(_SAT) - 2)
    P0, P1 = _SAT_P_Pa[i], _SAT_P_Pa[i + 1]
    t = (P_Pa - P0) / (P1 - P0) if P1 != P0 else 0.0
    return _SAT[i][col] * (1.0 - t) + _SAT[i + 1][col] * t

def _rho_l(P_Pa: float) -> float:  return _sat_interp(P_Pa, 2)
def _rho_v(P_Pa: float) -> float:  return _sat_interp(P_Pa, 3)
def _h_l(P_Pa: float) -> float:    return _sat_interp(P_Pa, 4)
def _h_v(P_Pa: float) -> float:    return _sat_interp(P_Pa, 5)
def _cp_l(P_Pa: float) -> float:   return _sat_interp(P_Pa, 6)

def _rho_subcooled(T_K: float, P_Pa: float) -> float:
    """Subcooled density ≈ rho_l at P (weak P-dependence below dome)."""
    return _rho_l(P_Pa)

def _h_subcooled(T_K: float, P_Pa: float) -> float:
    """Subcooled enthalpy ≈ h_l(P) + cp_l(P)·(T - T_sat(P))."""
    return _h_l(P_Pa) + _cp_l(P_Pa) * (T_K - _T_sat(P_Pa))


# ================================================================
# PressurizerModel
# ================================================================
class PressurizerModel:
    """
    Lumped AP1000 pressurizer model.

    Heater band:  15.30 – 15.51 MPa
    Spray band:   15.58 – 15.92 MPa   (deadband 15.51–15.58)
    """

    def __init__(self):
        # Geometry
        self.pzr_volume         = 59.47
        self.pzr_inner_diameter = 2.28
        self.pzr_height         = 16.27
        self.pzr_area           = math.pi * (self.pzr_inner_diameter / 2.0) ** 2

        # Hardware
        self.HEATER_POWER    = 1.6e6   # W
        self.SPRAY_FLOW_RATE = 44.1    # kg/s
        self.RELIEF_SETPOINT = 17.1e6  # Pa
        self.surge_deadband_K = 0.005  # [K] per-step noise filter (was 0.5, too large for dt=0.1s)

        # Control gains
        self.K_SURGE = 50.0  #************************CHANGED FROM 1.5***************************************************
        self.K_P0    = 5.0e3  #****************************CHANGED FROM 50000**************************************************

        # Nominal reference
        self.Tavg_nom = 0.5 * (cfg.T_hot_nom_K + cfg.T_cold_nom_K)
        self.P_nom    = cfg.P_pri_nom_Pa

        # Spray source temperature
        self.SPRAY_TEMP = cfg.T_cold_nom_K

        # Initial conditions — consistent with relax_surge_inventory target (50% level)
        self.pressure = self.P_nom
        self.T_sat    = _T_sat(self.pressure)

        level_init = 0.5 * self.pzr_height        # same target as relax_surge_inventory
        self.V_liq = level_init * self.pzr_area
        self.V_vap = self.pzr_volume - self.V_liq

        self.rho_l = _rho_l(self.pressure)
        self.rho_v = _rho_v(self.pressure)
        self.m_l   = self.rho_l * self.V_liq
        self.m_v   = self.rho_v * self.V_vap
        self.T_liq = self.T_sat - 5.0

        # Logging
        self.last_pressure = self.pressure
        self.last_level    = 0.0
        self.last_heater   = 0.0
        self.last_spray    = 0.0

        # Placeholder volumes
        self.V_hot_m3  = 7.5
        self.V_cold_m3 = 11.5

        # Pressure bands
        self.P_HEAT_ON   = 15.30e6
        self.P_HEAT_OFF  = 15.51e6
        self.P_SPRAY_ON  = 15.58e6
        self.P_SPRAY_OFF = 15.51e6
        self.P0_SHIFT_LIMIT = 0.5e6

        # Heater smoothing
        self.P_SETPOINT = 15.50e6
        self.HEATER_LAG_TAU = 10.0 # seconds (thermal inertia)
        self.heater_frac = 0.0 # 0–1 actual applied heater power

        self.dP0 = 0.0
        self.tau_dP0 = 300.0   # [s] rate-limit for pressure setpoint shift
        self.T_hot_prev = None

        self.surge_equilibrium_offset_K = 30.0  # *********************CHANGED FROM 25************************

        # Exposed state for GUI
        self.surge_direction = "NEUTRAL"  # "IN-SURGE", "OUT-SURGE", or "NEUTRAL"

    # ----------------------------------------------------------
    # Control
    # ----------------------------------------------------------
    def evaluate_control(self, T_hot, T_cold):
        T_avg = 0.5 * (T_hot + T_cold)
        self.t_avg = T_avg

        dP0_target = self.K_P0 * (T_avg - self.Tavg_nom)
        dP0_target = max(-self.P0_SHIFT_LIMIT, min(self.P0_SHIFT_LIMIT, dP0_target))

        # Rate-limit the setpoint shift (prevents snap-on oscillation)
        self.dP0 += (dP0_target - self.dP0) * self.dt / self.tau_dP0

        P_heat_on   = self.P_HEAT_ON   + self.dP0
        P_heat_off  = self.P_HEAT_OFF  + self.dP0
        # Spray bands are NOT shifted — spray should never activate
        # during a cooldown transient when dP0 is negative
        P_spray_on  = self.P_SPRAY_ON
        P_spray_off = self.P_SPRAY_OFF

        heater = self.last_heater
        spray  = self.last_spray

        if spray > 0.5:
            if self.pressure < P_spray_off:
                spray = 0.0
        else:
            if self.pressure > P_spray_on:
                spray = 1.0

        if heater > 0.5:
            if self.pressure > P_heat_off:
                heater = 0.0
        else:
            if self.pressure < P_heat_on:
                heater = 1.0

        if spray > 0.5:
            heater = 0.0

        self.heater      = heater
        self.spray       = spray
        self.last_heater = heater
        self.last_spray  = spray
        return heater, spray

    # ----------------------------------------------------------
    # Liquid energy
    # ----------------------------------------------------------
    def add_liquid_energy(self, Q):
        if self.m_l <= 0.0:
            return
        T_sat  = _T_sat(self.pressure)
        cp     = _cp_l(self.pressure)
        self.T_liq = min(self.T_liq, T_sat)
        self.T_liq += Q / (self.m_l * cp)

    # ----------------------------------------------------------
    # Surge
    # ----------------------------------------------------------
    def apply_surge(self, T_hot, dt):
        # Initialize reference on first call → no startup transient
        if self.T_hot_prev is None:
            self.T_hot_prev = T_hot
            self.surge_direction = "NEUTRAL"
            return

        dT_hot = T_hot - self.T_hot_prev
        self.T_hot_prev = T_hot

        # Only respond to *changes* in hot-leg temperature
        if abs(dT_hot) < self.surge_deadband_K:
            self.surge_direction = "NEUTRAL"
            return

        dm = self.K_SURGE * dT_hot # * dt
        if abs(dm) < 1e-9:
            self.surge_direction = "NEUTRAL"
            return

        # Track direction for GUI
        if dm > 0.0:
            self.surge_direction = "IN-SURGE"
        else:
            self.surge_direction = "OUT-SURGE"

        T_sat = _T_sat(self.pressure)

        if dm > 0.0:
            # Surge INTO pressurizer
            h_in = _h_l(self.pressure) if T_hot >= T_sat - 1e-3 else _h_subcooled(T_hot, self.pressure)
            h_liq = _h_l(self.pressure) if self.T_liq >= T_sat - 1e-3 else _h_subcooled(self.T_liq, self.pressure)
            self.m_l += dm
            self.add_liquid_energy(dm * (h_in - h_liq))
        else:
            # Surge OUT of pressurizer
            dm_out = min(-dm, 0.9 * self.m_l)
            h_l = _h_l(self.pressure) if self.T_liq >= T_sat - 1e-3 else _h_subcooled(self.T_liq, self.pressure)
            self.m_l -= dm_out
            self.add_liquid_energy(-dm_out * h_l)

    # ----------------------------------------------------------
    # Surge relax
    # ----------------------------------------------------------

    def relax_surge_inventory(self, dt):
        # Slowly restore nominal liquid inventory
        level_nom = 0.5 * self.pzr_height
        V_liq_nom = level_nom * self.pzr_area
        # V_liq_nom = self.pzr_volume - self.V_vap
        rho_l = _rho_l(self.pressure)
        m_l_nom = rho_l * V_liq_nom

        tau = 500.0 # 300.0  # seconds (RCS mixing time) #*********************8CHANGED FROM 1500*******************************
        self.m_l += (m_l_nom - self.m_l) * dt / tau

    # ----------------------------------------------------------
    # Flash
    # ----------------------------------------------------------
    def flash_if_needed(self):
        T_sat = _T_sat(self.pressure)
        if self.T_liq <= T_sat + 1e-3:
            self.T_liq = T_sat
            return

        h_l_sat = _h_l(self.pressure)
        h_v_sat = _h_v(self.pressure)
        h_liq   = h_l_sat if self.T_liq >= T_sat - 5e-3 else _h_subcooled(self.T_liq, self.pressure)

        excess = (h_liq - h_l_sat) * self.m_l
        if excess <= 0.0:
            self.T_liq = T_sat
            return

        denom = h_v_sat - h_l_sat
        if denom <= 0.0:
            self.T_liq = T_sat
            return

        m_flash = min(excess / denom, 0.05 * self.m_l)
        # m_flash = min(excess / denom, 0.3 * self.m_l)
        self.m_l -= m_flash
        self.m_v += m_flash
        self.T_liq = T_sat

    # ----------------------------------------------------------
    # Pressure solve
    # ----------------------------------------------------------
    def _sat_vapor_pressure_from_density(self, rho_target: float) -> float:
        P_lo, P_hi = 1.0e5, 21.5e6
        if rho_target <= _rho_v(P_lo):
            return P_lo
        if rho_target >= _rho_v(P_hi):
            return P_hi
        for _ in range(50):
            P_mid = 0.5 * (P_lo + P_hi)
            if _rho_v(P_mid) < rho_target:
                P_lo = P_mid
            else:
                P_hi = P_mid
        return 0.5 * (P_lo + P_hi)

    def _update_volumes(self):
        """Recompute liquid/vapor volumes from current masses."""
        rho_l = _rho_l(self.pressure)
        # Protect against zero/negative liquid mass
        self.m_l = max(self.m_l, 1.0)
        self.V_liq = self.m_l / rho_l
        self.V_liq = min(self.V_liq, 0.95 * self.pzr_volume)  # keep some steam space
        self.V_vap = self.pzr_volume - self.V_liq

    def solve_pressure(self):
        tau_P = 30.0 # 20.0  # seconds  #************************************CHANGED FROM 120.0**********************************
        P_target = self._sat_vapor_pressure_from_density(self.m_v / self.V_vap)
        self.pressure += (P_target - self.pressure) * self.dt / tau_P

    # ----------------------------------------------------------
    # Main step
    # ----------------------------------------------------------
    def step(self, T_hot_K, T_cold_K, dt):
        T_hot_K = min(T_hot_K, 616.45)
        self.dt = dt

        heater, spray = self.evaluate_control(T_hot_K, T_cold_K)

        # --- Continuous heater modulation (no bang-bang) ---
        P_err = (self.P_SETPOINT + self.dP0) - self.pressure

        # Proportional heater command over 0–0.20 MPa
        heater_cmd = max(0.0, min(1.0, P_err / 0.40e6)) #*****************************CHANGED FROM 0.20e6***************************

        # First-order lag (heater thermal inertia)
        self.heater_frac += (heater_cmd - self.heater_frac) * dt / self.HEATER_LAG_TAU
        self.heater_frac = max(0.0, min(1.0, self.heater_frac))

        # Apply heater energy
        if self.heater_frac > 1e-4:
            Q = self.HEATER_POWER * self.heater_frac * dt
            dh = max(_h_v(self.pressure) - _h_l(self.pressure), 1.0)
            dm_v = min(Q / dh, self.m_l)
            self.m_l -= dm_v
            self.m_v += dm_v

        # Spray
        if spray > 0.0:
            dm = self.SPRAY_FLOW_RATE * dt
            if dm > 0.0:
                T_sat = _T_sat(self.pressure)
                h_l_sat = _h_l(self.pressure)
                h_v_sat = _h_v(self.pressure)
                h_fg = max(h_v_sat - h_l_sat, 1.0)

                h_in = h_l_sat if self.SPRAY_TEMP >= T_sat - 5e-3 else _h_subcooled(self.SPRAY_TEMP, self.pressure)
                h_liq = h_l_sat if self.T_liq >= T_sat - 5e-3 else _h_subcooled(self.T_liq, self.pressure)

                Q_warm = max(0.0, dm * (h_l_sat - h_in))
                m_cond = min(self.m_v, Q_warm / h_fg)
                h_eff = min(h_in + (m_cond * h_fg) / dm, h_l_sat)

                self.m_l += dm + m_cond
                self.m_v -= m_cond
                self.add_liquid_energy(dm * (h_eff - h_liq))

        self.apply_surge(T_hot_K, dt)
        self.relax_surge_inventory(dt * 0.1)
        # if abs(self.pressure - self.P_SETPOINT) < 0.02e6:
            # dm = 0.001 * self.m_v * dt
            # self.m_v -= dm
            # self.m_l += dm
        self.flash_if_needed()
        self._update_volumes()
        self.solve_pressure()

        if self.pressure > self.RELIEF_SETPOINT:
            self.pressure = self.RELIEF_SETPOINT

        # Level
        T_sat = _T_sat(self.pressure)
        rho_l = _rho_l(self.pressure) if self.T_liq >= T_sat - 5e-3 else _rho_subcooled(self.T_liq, self.pressure)
        level = min(max((self.m_l / rho_l) / self.pzr_area, 0.0), self.pzr_height)
        self.level = level

        self.last_pressure = self.pressure
        self.last_level = level

        return self.pressure, self.level, self.heater, self.spray
