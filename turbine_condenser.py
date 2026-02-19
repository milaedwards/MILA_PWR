
# turbine_condenser.py  –  Condensing turbine + flow controller (NO CoolProp)
#
# The two CoolProp calls were:
#   s_in          = PropsSI("S", "P", inlet_p, "H", h_in, "Water")
#   h_out_isen    = PropsSI("H", "P", outlet_p, "S", s_in, "Water")
#
# Replacement strategy:
#   1) s_in is looked up from the NIST saturation table.  The turbine inlet
#      is always near-saturation steam from the SG, so s ≈ s_v(P_inlet).
#   2) At the condenser (~9.82 kPa) the exhaust is deep in the two-phase dome.
#      h_out_isen = h_l + x*(h_v - h_l)   where  x = (s_in - s_l)/(s_v - s_l)
#
# Everything else (efficiency model, flow controller) is unchanged.

import bisect
from config import Config

# ================================================================
# Shared saturation table  (same source as pressurizer.py)
# Columns: P_MPa, T_K, rho_l, rho_v, h_l, h_v, cp_l, s_l, s_v
# ================================================================
_SAT = [
    ( 0.00982, 316.00,  996.5,  0.06034,  184.0e3,  2584.0e3,  4180,  622.6,  8150.2),   # condenser
    ( 0.10,    372.76,  958.4,  0.5978,   417.5e3,  2675.0e3,  4217, 1302.8,  7359.4),
    ( 0.50,    424.98,  917.0,  2.950,    639.7e3,  2748.7e3,  4181, 1860.5,  6821.3),
    ( 1.00,    453.03,  887.0,  5.974,    762.8e3,  2778.1e3,  4217, 2138.7,  6586.5),
    ( 2.00,    485.53,  852.0, 12.04,     908.8e3,  2799.5e3,  4350, 2447.0,  6340.1),
    ( 3.00,    507.43,  823.0, 18.65,    1008.4e3,  2804.2e3,  4540, 2645.0,  6185.2),
    ( 5.00,    536.67,  775.0, 32.55,    1154.2e3,  2794.3e3,  5000, 2921.9,  5973.4),
    ( 5.764,   546.13,  757.0, 29.28,    1154.2e3,  2787.0e3,  5200, 2920.2,  5973.4),
    ( 7.00,    558.15,  740.0, 39.00,    1267.4e3,  2773.0e3,  5400, 3122.6,  5822.1),
    (10.00,    584.15,  688.4, 58.35,    1407.6e3,  2724.7e3,  5600, 3359.6,  5614.1),
    (12.00,    596.76,  654.0, 73.00,    1491.3e3,  2683.5e3,  6000, 3477.6,  5480.5),
    (15.00,    611.72,  603.0,100.0,     1610.5e3,  2610.5e3,  6800, 3615.8,  5310.8),
    (15.513,   617.15,  598.5,109.6,     1610.5e3,  2611.0e3,  6500, 3684.8,  5309.8),
    (18.00,    629.99,  545.0,140.0,     1740.0e3,  2508.0e3,  8000, 3790.0,  5110.0),
    (20.00,    638.90,  487.0,182.0,     1826.3e3,  2409.7e3, 10000, 3870.0,  4930.0),
    (21.50,    644.49,  425.0,240.0,     1930.0e3,  2294.0e3, 20000, 3970.0,  4740.0),
]
_SAT_P_Pa = [row[0] * 1e6 for row in _SAT]

def _sat_interp(P_Pa: float, col: int) -> float:
    if P_Pa <= _SAT_P_Pa[0]:
        return _SAT[0][col]
    if P_Pa >= _SAT_P_Pa[-1]:
        return _SAT[-1][col]
    i = bisect.bisect_right(_SAT_P_Pa, P_Pa) - 1
    i = min(i, len(_SAT) - 2)
    P0, P1 = _SAT_P_Pa[i], _SAT_P_Pa[i + 1]
    t = (P_Pa - P0) / (P1 - P0) if P1 != P0 else 0.0
    return _SAT[i][col] * (1.0 - t) + _SAT[i + 1][col] * t

# col indices:  4=h_l  5=h_v  7=s_l  8=s_v
def _h_l(P): return _sat_interp(P, 4)
def _h_v(P): return _sat_interp(P, 5)
def _s_l(P): return _sat_interp(P, 7)
def _s_v(P): return _sat_interp(P, 8)


class TurbineModel:
    """
    Condensing steam turbine + simple flow controller.

    Uses Config for nominal values; efficiencies and condenser pressure
    fall back to hard-coded AP1000-consistent defaults if not present.
    """

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()

        self.turbine_efficiency   = getattr(self.cfg, "turbine_efficiency",   0.78711)
        self.generator_efficiency = getattr(self.cfg, "generator_efficiency", 0.90)
        self.outlet_p             = getattr(self.cfg, "P_condenser_Pa",       9820.53)   # Pa

    # ------------------------------------------------------
    def turbine_power(
        self,
        inlet_p: float,      # [Pa]
        inlet_h: float,      # [J/kg]
        outlet_p: float,     # [Pa]
        m_dot_steam: float,  # [kg/s]
    ) -> float:
        """
        Electrical power [MW] via isentropic expansion.

        Inlet entropy = s_v(inlet_p) because the SG delivers near-saturation
        steam.  Outlet enthalpy is computed from two-phase quality at the
        condenser pressure.
        """
        h_in = inlet_h

        # Inlet entropy (sat-vapor at SG pressure)
        s_in = _s_v(inlet_p)

        # Isentropic outlet (two-phase at condenser)
        s_l_out = _s_l(outlet_p)
        s_v_out = _s_v(outlet_p)
        h_l_out = _h_l(outlet_p)
        h_v_out = _h_v(outlet_p)

        s_fg = s_v_out - s_l_out
        x_isen = max(0.0, min(1.0, (s_in - s_l_out) / s_fg)) if s_fg > 0.0 else 1.0

        h_out_isen = h_l_out + x_isen * (h_v_out - h_l_out)

        # Actual outlet with isentropic efficiency
        h_out_actual = h_in - self.turbine_efficiency * (h_in - h_out_isen)

        specific_work = h_in - h_out_actual   # [J/kg]

        return self.generator_efficiency * specific_work * m_dot_steam / 1.0e6  # [MW]

    # ------------------------------------------------------
    def update_mass_flow_rate(
        self,
        m_dot_steam: float,
        power_supplied: float,
        power_dem: float,
        dt: float,
    ) -> float:
        """
        Flow controller with NRC-style rate limits:
            10 %/s step   +   5 %/min ramp
        """
        eps   = 1.0e-6
        denom = max(abs(power_dem), eps)

        power_percent_change = (power_supplied - power_dem) / denom

        m_dot_step = m_dot_steam * 0.10 * dt          # 10 %/s
        m_dot_ramp = m_dot_steam * 0.05 * dt / 60.0   # 5 %/min

        if power_percent_change > 0.10:
            m_dot_update = -(m_dot_step + m_dot_ramp)
        elif power_percent_change < -0.10:
            m_dot_update =  (m_dot_step + m_dot_ramp)
        else:
            m_dot_update = -power_percent_change * m_dot_steam * dt

        return max(m_dot_steam + m_dot_update, 0.0)

    # ======================================================
    #                   MAIN STEP
    # ======================================================
    def step(
        self,
        inlet_h: float,
        inlet_p: float,
        m_dot_steam: float,
        power_dem: float,
        dt: float,
    ) -> tuple[float, float]:
        """
        Returns:
            power_output_MW [MW]
            m_dot_steam_new [kg/s]
        """
        power_output = self.turbine_power(
            inlet_p=inlet_p,
            inlet_h=inlet_h,
            outlet_p=self.outlet_p,
            m_dot_steam=m_dot_steam,
        )

        m_dot_cmd_next = self.update_mass_flow_rate(
            m_dot_steam=m_dot_steam,
            power_supplied=power_output,
            power_dem=power_dem,
            dt=dt,
        )

        return power_output, m_dot_cmd_next
