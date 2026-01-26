# steam_generator.py
# Clean AP1000 Delta-125 Steam Generator model
# Includes:
#   • SGCore         – detailed SG thermal/pressure physics
#   • SteamGenerator – wrapper used by ICSystem
#
# Units are indicated in comments.

import math
from dataclasses import dataclass, field
from typing import Dict

from config import Config

try:
    import CoolProp.CoolProp as CP
except ImportError:
    CP = None

def delta_h_steam_fw_J_kg(P_Pa: float, cfg: Config) -> float:
    """
    Full enthalpy rise used to convert W <-> kg/s:
      Δh(P) = h_steam_out(P, sat vapor) - h_fw_in(P, T_fw)
    """
    P = max(float(P_Pa), 1e5)

    if CP is not None:
        try:
            h_g = CP.PropsSI("H", "P", P, "Q", 1, "Water")          # sat vapor
            h_fw = CP.PropsSI("H", "P", P, "T", cfg.T_fw_K, "Water") # feedwater in
            return max(h_g - h_fw, 1.0)
        except Exception:
            pass

    # Fallback: nominal constant from config (still consistent)
    return max(cfg.delta_h_steam_fw_J_kg, 1.0)


# ======================================================================
#                           SGCore (PHYSICS)
# ======================================================================

@dataclass
class SGCore:
    """
    Internal Delta-125 Steam Generator model.

    Models:
      • Primary τ-chain: RX upper → hot leg → SG inlet → tube nodes → SG outlet → cold leg
      • Tube / metal double-lag chain (primary–metal–steam)
      • Steam-dome pressure with mass–energy ODE (Fix-1C style)
      • Saturated steam temperature from CoolProp (if available)

    All dynamics are state-in/state-out through a flat dict.
    """

    cfg: Config = field(default_factory=Config)

    # Primary hydraulic taus [s]
    tau_hot: float = field(init=False)
    tau_sgi: float = field(init=False)
    tau_sgu: float = field(init=False)
    tau_cold: float = field(init=False)

    # Tube / metal / secondary taus [s]
    tau_p: float = field(init=False)   # primary tube segment τ
    tau_pm: float = field(init=False)  # primary → metal
    tau_mp: float = field(init=False)  # metal coupling
    tau_ms: float = field(init=False)  # metal → secondary

    # Metal → steam conductance [W/K]
    G_ms_W_K: float = field(init=False)

    def __post_init__(self):
        c = self.cfg

        # Hydraulic taus (per-loop, from Config)
        self.tau_hot = c.tau_hot_s
        self.tau_sgi = c.tau_sgi_s
        self.tau_sgu = c.tau_sgu_s
        self.tau_cold = c.tau_cold_s

        # Thermal taus
        # Current model uses a single primary τ (reuse τ_pm_s) plus pm/mp/ms from Config
        self.tau_p = c.tau_pm_s
        self.tau_pm = c.tau_pm_s
        self.tau_mp = c.tau_mp_s
        self.tau_mp = c.tau_mp_metal_s
        #self.tau_ms = c.tau_ms_s
        self.tau_ms = c.tau_ms_metal_s

        # Metal → steam conductance from calibrated Config
        self.G_ms_W_K = c.G_ms_W_K

    # ------------------------------------------------------------------
    @staticmethod
    def _lag(T_node: float, T_src: float, tau_s: float, dt: float) -> float:
        """
        First-order lag:
            dT/dt = (T_src − T_node) / tau

        T_node [K], T_src [K], tau_s [s], dt [s]
        """
        return T_node + (T_src - T_node) * (dt / tau_s)

    # ------------------------------------------------------------------
    #                MASS–ENERGY PRESSURE MODEL (Fix-1C)
    # ------------------------------------------------------------------
    def _pressure_Fix1C(
        self,
        Ts_old: float,      # [K] old steam dome temperature
        Ts_new: float,      # [K] predicted new steam dome temperature
        P_old: float,       # [Pa] old steam pressure
        m_dot_steam: float, # [kg/s] steam outflow
        q_ms_W: float,      # [W] metal→steam heat
        dt: float           # [s]
    ) -> float:
        """
        Kerlin/Upadhyaya-style mass–energy pressure ODE under saturated conditions.

        Approximations:
          m_s = (P * V) / (R * T)
          m_evap = q_ms / h_fg
          dm/dt = m_evap − m_dot_steam
          dP/dt = (R*T/V) * [ dm/dt + (P/T) * dT/dt ]

        R      – water vapor gas constant [J/kg-K]
        V      – steam-dome volume [m³]
        h_fg   – effective latent heat from Config.delta_h_steam_fw_J_kg
        """

        c = self.cfg

        R = 461.5                     # [J/kg-K]
        V = c.V_steam_m3              # [m³]
        #h_fg = max(c.delta_h_steam_fw_J_kg, 1.0e5)  # [J/kg]
        delta_h = delta_h_steam_fw_J_kg(P_old, c)  # [J/kg]

        # dT/dt of steam dome [K/s]
        dTdt = (Ts_new - Ts_old) / dt

        # Evaporation rate [kg/s]
        #m_evap = q_ms_W / h_fg
        m_evap = q_ms_W / delta_h

        # Net steam mass change [kg/s]
        dms_dt = m_evap - m_dot_steam

        # Pressure derivative [Pa/s]
        dPdt = (R * Ts_old / V) * (dms_dt + (P_old / Ts_old) * dTdt)

        # Explicit Euler update
        return P_old + dPdt * dt

    # ------------------------------------------------------------------
    #                    MAIN SG PHYSICS STEP
    # ------------------------------------------------------------------
    def step(self, s: Dict[str, float]) -> Dict[str, float]:
        """
        Advance SGCore one timestep.

        Input state dict must contain:
          • "dt"        [s]
          • "T_rxu"     [K] reactor outlet (upper plenum)
          • "T_hot"     [K] hot leg bulk
          • "T_sgi"     [K] SG inlet plenum
          • "T_p1"      [K] primary tube node 1
          • "T_p2"      [K] primary tube node 2
          • "T_m1"      [K] metal node 1
          • "T_m2"      [K] metal node 2
          • "T_sgu"     [K] SG outlet plenum
          • "T_cold"    [K] cold leg bulk
          • "T_s"       [K] steam dome temperature
          • "p_s"       [Pa] steam pressure
          • "M_dot_stm" [kg/s] steam mass flow to turbine

        Returns a new dict with "*_new" keys and updated "p_s_new" and "T_s_new".
        """

        dt = s["dt"]  # [s]

        # Extract state
        T_rxu = s["T_rxu"]   # [K] reactor outlet plenum
        T_hot = s["T_hot"]   # [K]
        T_sgi = s["T_sgi"]   # [K]
        T_p1 = s["T_p1"]     # [K]
        T_p2 = s["T_p2"]     # [K]
        T_m1 = s["T_m1"]     # [K]
        T_m2 = s["T_m2"]     # [K]
        T_sgu = s["T_sgu"]   # [K]
        T_cold = s["T_cold"] # [K]
        T_s = s["T_s"]       # [K] steam dome
        P_s = s["p_s"]       # [Pa]
        m_dot = s["M_dot_stm"]  # [kg/s]

        # --- numerical safety ---
        # This model calls saturation properties and uses P in denominators.
        # If P drifts to non-physical (<= 0) during init/settle, CoolProp
        # throws and the pressure state can run away. Keep it positive.
        P_s = max(P_s, 1.0e5)

        # ------------------------------------------------------
        # 1) Primary hydraulic: RXU → hot → SG inlet
        # ------------------------------------------------------
        T_hot_new = self._lag(T_hot, T_rxu, self.tau_hot, dt)
        T_sgi_new = self._lag(T_sgi, T_hot_new, self.tau_sgi, dt)

        # ------------------------------------------------------
        # 2) Tube / metal thermal double-lag
        # ------------------------------------------------------
        # Primary tube nodes
        dTp1_dt = (T_sgi_new - T_p1) / self.tau_p - (T_p1 - T_m1) / self.tau_pm
        dTp2_dt = (T_p1 - T_p2) / self.tau_p - (T_p2 - T_m2) / self.tau_pm

        # Metal nodes
        dTm1_dt = (T_p1 - T_m1) / self.tau_mp - (T_m1 - T_s) / self.tau_ms
        dTm2_dt = (T_p2 - T_m2) / self.tau_mp - (T_m2 - T_s) / self.tau_ms

        T_p1_new = T_p1 + dTp1_dt * dt
        T_p2_new = T_p2 + dTp2_dt * dt
        T_m1_new = T_m1 + dTm1_dt * dt
        T_m2_new = T_m2 + dTm2_dt * dt

        # ------------------------------------------------------
        # 3) Outlet plenum → cold leg
        # ------------------------------------------------------
        T_sgu_new = self._lag(T_sgu, T_p2_new, self.tau_sgu, dt)
        T_cold_new = self._lag(T_cold, T_sgu_new, self.tau_cold, dt)

        # ------------------------------------------------------
        # 4) Predict steam temperature from saturation at old P_s
        # ------------------------------------------------------
        if CP is not None:
            try:
                T_s_pred = CP.PropsSI("T", "P", P_s, "Q", 0, "Water")
            except Exception:
                T_s_pred = T_s
        else:
            T_s_pred = T_s

        # ------------------------------------------------------
        # 5) Metal→steam heat flow q_ms [W]
        # ------------------------------------------------------
        T_m_avg_new = 0.5 * (T_m1_new + T_m2_new)
        q_ms_W = self.G_ms_W_K * (T_m_avg_new - T_s)

        if getattr(self, "_printed_gms", False) is False:
            print(f"[SGCore] G_ms_W_K = {self.G_ms_W_K:.3e} W/K")
            self._printed_gms = True

        # --- Enforce primary-side energy consistency across the SG ---
        m_dot_pri = self.cfg.m_dot_primary_nom_kg_s
        cp_pri = self.cfg.cp_primary_J_kgK

        # Clamp to avoid weird sign issues
        q_ms_pos_W = max(q_ms_W, 0.0)

        dT_sg_K = q_ms_pos_W / max(m_dot_pri * cp_pri, 1.0)  # K
        T_sgu_target = T_sgi_new - dT_sg_K

        # Relax outlet toward energy-consistent target with existing outlet time constant
        T_sgu_new = self._lag(T_sgu, T_sgu_target, self.tau_sgu, dt)

        # Cold leg follows outlet
        T_cold_new = self._lag(T_cold, T_sgu_new, self.tau_cold, dt)
        # ------------------------------------------------------------

        # ------------------------------------------------------
        # 6) Pressure update via Fix-1C ODE
        # ------------------------------------------------------
        P_s_new = self._pressure_Fix1C(
            Ts_old=T_s,
            Ts_new=T_s_pred,
            P_old=P_s,
            m_dot_steam=m_dot,
            q_ms_W=q_ms_W,
            dt=dt,
        )

        # Enforce physical bounds (prevents negative pressures during init/settle)
        P_s_new = max(P_s_new, 1.0e5)

        # ------------------------------------------------------
        # 7) Updated saturated steam temperature at new pressure
        # ------------------------------------------------------
        if CP is not None:
            try:
                T_s_new = CP.PropsSI("T", "P", P_s_new, "Q", 0, "Water")
            except Exception:
                T_s_new = T_s
        else:
            T_s_new = T_s

        # ------------------------------------------------------
        # 8) Return new state
        # ------------------------------------------------------
        return {
            "T_rxu_new": T_rxu,        # [K] RXU boundary unchanged here
            "T_hot_new": T_hot_new,    # [K]
            "T_sgi_new": T_sgi_new,    # [K]
            "T_p1_new": T_p1_new,      # [K]
            "T_p2_new": T_p2_new,      # [K]
            "T_m1_new": T_m1_new,      # [K]
            "T_m2_new": T_m2_new,      # [K]
            "T_sgu_new": T_sgu_new,    # [K]
            "T_cold_new": T_cold_new,  # [K]
            "p_s_new": P_s_new,        # [Pa]
            "T_s_new": T_s_new,        # [K]
            "q_ms_W_new": q_ms_W,      # [W] latest metal→steam heat flow
        }


# ======================================================================
#                      SteamGenerator WRAPPER
# ======================================================================

@dataclass
class SteamGenerator:
    """
    Wrapper around SGCore for use in ICSystem.

    Interface:

        step(
            T_hot_K: float,              # primary hot-leg temperature [K]
            m_dot_steam_cmd_kg_s: float, # turbine-requested steam flow [kg/s]
            dt: float                    # timestep [s]
        ) -> (
            T_cold_K: float,             # primary cold-leg temperature [K]
            m_dot_steam_actual: float,   # actual steam produced [kg/s]
            P_sec_Pa: float,             # secondary pressure [Pa]
            T_sec_K: float,              # steam dome temperature [K]
            sg_limited: bool,            # True if SG clipping steam flow
            h_steam_J_kg: float          # steam enthalpy [J/kg]
        )

    Also exposes legacy attributes for compatibility with existing code:
        • T_metal_K
        • T_sec_K
        • P_sec_Pa
        • m_dot_steam_kg_s
    """

    cfg: Config
    core: SGCore = field(init=False)
    state: Dict[str, float] = field(init=False)

    # Legacy / compatibility attributes
    T_metal_K: float = field(init=False)        # [K] average metal temperature
    T_sec_K: float = field(init=False)          # [K] steam dome temperature
    P_sec_Pa: float = field(init=False)         # [Pa] secondary pressure
    m_dot_steam_kg_s: float = field(init=False) # [kg/s] actual steam flow
    q_ms_last_W: float = field(init=False)      # [W] last metal→steam heat flow

    def __post_init__(self):
        c = self.cfg
        self.core = SGCore(c)

        print(f"[SteamGenerator] cfg.G_ms_calib_W_K = {getattr(c, 'G_ms_calib_W_K', None)}")

        # Initialize SGCore state near DCD-nominal conditions
        self.state = {
            # primary side
            "T_rxu":  c.T_hot_nom_K,      # [K] reactor outlet plenum
            "T_hot":  c.T_hot_nom_K,      # [K] hot leg
            "T_sgi":  c.T_hot_nom_K,      # [K] SG inlet plenum
            "T_p1":   c.T_hot_nom_K,      # [K] primary tube node 1
            "T_p2":   c.T_cold_nom_K,     # [K] primary tube node 2
            "T_m1":   c.T_metal_nom_K,    # [K] metal node 1
            "T_m2":   c.T_metal_nom_K,    # [K] metal node 2
            "T_sgu":  c.T_cold_nom_K,     # [K] SG outlet plenum
            "T_cold": c.T_cold_nom_K,     # [K] cold leg back to core

            # secondary side
            "T_s":  c.T_sec_nom_K,        # [K] steam dome
            "p_s":  c.P_sec_nom_Pa,       # [Pa] steam pressure

            # flow + timestep
            "M_dot_stm": c.m_dot_steam_nom_kg_s,  # [kg/s]
            "dt":        c.dt,                    # [s]
        }

        self._init_internal_nodes_at_steady()

        # Initialize legacy attributes consistently
        self.T_metal_K = c.T_metal_nom_K
        self.T_sec_K = c.T_sec_nom_K
        self.P_sec_Pa = c.P_sec_nom_Pa
        self.m_dot_steam_kg_s = c.m_dot_steam_nom_kg_s
        #self.q_ms_last_W = c.m_dot_steam_nom_kg_s * c.delta_h_steam_fw_J_kg
        self.q_ms_last_W = c.m_dot_steam_nom_kg_s * delta_h_steam_fw_J_kg(c.P_sec_nom_Pa, c)

        # Drive the SG state toward steady conditions so we start balanced
        self._settle_to_equilibrium()

        # Force an exact nominal startup point (idealized PWR):
        #   Psec = P_nom, m_dot = m_dot_nom, and q_ms trimmed so dP/dt ≈ 0 at t=0.
        self._force_nominal_start()

    # ---------------------------------------------------------------
    def _apply_core_update(self, s_new: Dict[str, float]) -> None:
        """Copy SGCore outputs back into our mutable state dict."""

        s = self.state
        s["T_rxu"] = s_new["T_rxu_new"]
        s["T_hot"] = s_new["T_hot_new"]
        s["T_sgi"] = s_new["T_sgi_new"]
        s["T_p1"] = s_new["T_p1_new"]
        s["T_p2"] = s_new["T_p2_new"]
        s["T_m1"] = s_new["T_m1_new"]
        s["T_m2"] = s_new["T_m2_new"]
        s["T_sgu"] = s_new["T_sgu_new"]
        s["T_cold"] = s_new["T_cold_new"]
        s["p_s"] = s_new["p_s_new"]
        s["T_s"] = s_new["T_s_new"]
        self.q_ms_last_W = s_new["q_ms_W_new"]

    # ---------------------------------------------------------------
    def _update_legacy_attrs(self) -> None:
        """Keep backwards-compatible attributes updated."""

        self.T_metal_K = 0.5 * (self.state["T_m1"] + self.state["T_m2"])
        self.T_sec_K = self.state["T_s"]
        self.P_sec_Pa = self.state["p_s"]
        self.m_dot_steam_kg_s = self.state["M_dot_stm"]

    # ---------------------------------------------------------------
    def _stage_steady(self, T_in: float, T_s: float):
        """
        Solve the stage steady-state for the coupled ODE pair:

          0 = (T_in - Tp)/tau_p  - (Tp - Tm)/tau_pm
          0 = (Tp  - Tm)/tau_mp - (Tm - T_s)/tau_ms

        Returns (Tp, Tm).
        """
        A = self.core.tau_p
        B = self.core.tau_pm
        C = self.core.tau_mp
        D = self.core.tau_ms

        # Derived closed-form solution
        denom = (A + B) - (A * D / (D + C))
        if abs(denom) < 1e-9:
            # fallback: just return something sane
            Tp = T_in
            Tm = 0.5 * (T_in + T_s)
            return Tp, Tm

        Tp = (B * T_in + (A * C / (D + C)) * T_s) / denom
        Tm = (D * Tp + C * T_s) / (D + C)
        return Tp, Tm

    def _init_internal_nodes_at_steady(self):
        """Initialize tube/metal nodes so the SG ODEs start near equilibrium."""
        c = self.cfg
        s = self.state

        # Steam saturation temp at nominal pressure (prefer CoolProp)
        if CP is not None:
            try:
                T_s = CP.PropsSI("T", "P", c.P_sec_nom_Pa, "Q", 0, "Water")
            except Exception:
                T_s = c.T_sec_nom_K
        else:
            T_s = c.T_sec_nom_K

        # Use nominal hot-leg as the primary inlet boundary
        T_in = c.T_hot_nom_K

        # Stage 1 steady
        Tp1, Tm1 = self._stage_steady(T_in, T_s)

        # Stage 2 steady (inlet is stage 1 tube temp)
        Tp2, Tm2 = self._stage_steady(Tp1, T_s)

        # Write state
        s["T_rxu"] = T_in
        s["T_hot"] = T_in
        s["T_sgi"] = T_in

        s["T_s"] = T_s
        s["p_s"] = c.P_sec_nom_Pa

        s["T_p1"] = Tp1
        s["T_m1"] = Tm1
        s["T_p2"] = Tp2
        s["T_m2"] = Tm2

        # Outlet/cold leg start consistent with tube-out (good enough)
        s["T_sgu"] = Tp2
        s["T_cold"] = Tp2

        self._update_legacy_attrs()

    def _sat_T_at_P(self, P_Pa: float) -> float:
        """Saturation temperature at pressure P (uses CoolProp if available)."""
        P = max(float(P_Pa), 1.0e5)
        if CP is not None:
            try:
                return float(CP.PropsSI("T", "P", P, "Q", 0, "Water"))
            except Exception:
                pass
        return float(self.cfg.T_sec_nom_K)

    def _force_nominal_start(self) -> None:
        """
        Idealized startup snap:
          - Force Psec = P_nom and Ts = Tsat(P_nom)
          - Force m_dot = m_dot_nom
          - Trim G_ms so q_ms = m_dot_nom * Δh(P_nom) (so Fix-1C has ~zero dP/dt at t=0)
        """
        c = self.cfg
        s = self.state

        P_nom = float(c.P_sec_nom_Pa)
        T_s_nom = float(self._sat_T_at_P(P_nom))

        # hard-set nominal secondary conditions + nominal steam flow
        s["p_s"] = P_nom
        s["T_s"] = T_s_nom
        s["M_dot_stm"] = float(c.m_dot_steam_nom_kg_s)

        # target heat required to support nominal steam flow at nominal pressure
        dh = float(delta_h_steam_fw_J_kg(P_nom, c))  # J/kg
        q_target_W = float(c.m_dot_steam_nom_kg_s) * dh

        # trim the metal->steam conductance so q_ms hits the target at the current (settled) metal temperature
        T_m_avg = 0.5 * (float(s["T_m1"]) + float(s["T_m2"]))
        dT = max(T_m_avg - T_s_nom, 1.0e-3)
        G_new = q_target_W / dT

        # keep it sane
        G_new = max(1.0e6, min(1.0e10, G_new))

        self.core.G_ms_W_K = G_new
        # keep cfg consistent (optional but helps debugging)
        try:
            c.G_ms_W_K = G_new
        except Exception:
            pass

        # make debug caps consistent at t=0
        self.q_ms_last_W = q_target_W

        self._update_legacy_attrs()

    def _settle_to_equilibrium(self) -> None:
        """Advance SGCore internally so we begin close to steady state (without drifting Psec)."""

        settle_time = getattr(self.cfg, "steamgen_settle_time_s", 0.0)
        if settle_time <= 0.0:
            return

        dt = max(self.cfg.dt, 1.0e-6)
        steps = max(1, int(math.ceil(settle_time / dt)))

        P_nom = float(self.cfg.P_sec_nom_Pa)
        T_s_nom = float(self._sat_T_at_P(P_nom))

        self.state["dt"] = dt

        for _ in range(steps):
            # Fixed boundaries for ideal nominal initialization
            self.state["T_rxu"] = self.cfg.T_hot_nom_K
            self.state["p_s"] = P_nom
            self.state["T_s"] = T_s_nom
            self.state["M_dot_stm"] = float(self.cfg.m_dot_steam_nom_kg_s)

            s_new = self.core.step(self.state)
            self._apply_core_update(s_new)

            # Override any Fix-1C pressure/temperature drift during init settle
            self.state["p_s"] = P_nom
            self.state["T_s"] = T_s_nom

        self._update_legacy_attrs()

    # ---------------------------------------------------------------
    def step(self, T_hot_K: float, Q_core_W: float, m_dot_steam_cmd_kg_s: float, dt: float):
        """
        Single-step interface used by ICSystem.
        """

        c = self.cfg
        s = self.state

        # 1) Update boundary conditions and valve command
        s["T_rxu"] = T_hot_K
        s["dt"] = dt

        # Available steam is limited both by dome pressure and by the
        # instantaneous metal→steam heat flow (i.e., actual boiling rate).
        pressure_ratio = max(s["p_s"], 0.0) / max(c.P_sec_nom_Pa, 1.0)
        if pressure_ratio <= 0.0:
            m_dot_pressure_cap = 0.0
        else:
            m_dot_pressure_cap = c.m_dot_steam_nom_kg_s * (
                pressure_ratio ** c.steam_flow_pressure_exp
            )

        #h_fg = max(c.delta_h_steam_fw_J_kg, 1.0)

        # Full Δh: sat vapor out minus feedwater in at T_fw
        try:
            from CoolProp.CoolProp import PropsSI
            P = max(s["p_s"], 1e5)
            h_steam = PropsSI("H", "P", P, "Q", 1, "Water")  # J/kg, sat vapor
            h_fw = PropsSI("H", "P", P, "T", c.T_fw_K, "Water")  # J/kg, feedwater
            delta_h = max(h_steam - h_fw, 1.0)
        except Exception:
            # fallback to config constant
            delta_h = max(c.delta_h_steam_fw_J_kg, 1.0)

        #m_dot_heat_cap = max(self.q_ms_last_W, 0.0) / delta_h
        m_dot_heat_cap = max(self.q_ms_last_W, 0.0) / delta_h  # keep for debug only

        # Primary energy cap based on REACTOR thermal power (idealized: all goes to SG)
        #m_dot_primary_cap = max(Q_core_W, 0.0) / h_fg

        #m_dot_capacity = min(m_dot_pressure_cap, m_dot_heat_cap)
        #m_dot_capacity = m_dot_heat_cap
        #m_dot_actual = min(m_dot_steam_cmd_kg_s, m_dot_capacity)
        m_dot_actual = min(m_dot_steam_cmd_kg_s, m_dot_pressure_cap)


        #h_fg = max(c.delta_h_steam_fw_J_kg, 1.0)
        #m_dot_heat_cap = max(self.q_ms_last_W, 0.0) / h_fg

        # NEW: Primary-side energy balance cap (can't boil more than primary heat removal)
        #Q_primary_W = c.m_dot_primary_nom_kg_s * c.cp_primary_J_kgK * max(T_hot_K - T_cold_K, 0.0)
        #m_dot_primary_cap = max(Q_primary_W, 0.0) / h_fg

        # Combine caps
        #m_dot_capacity = min(m_dot_pressure_cap, m_dot_heat_cap, m_dot_primary_cap)

        #m_dot_capacity = min(m_dot_pressure_cap, m_dot_heat_cap)
        #m_dot_actual = min(m_dot_steam_cmd_kg_s, m_dot_capacity)

        self._dbg_caps = {
            "pressure_cap": m_dot_pressure_cap,
            "heat_cap": m_dot_heat_cap,
            "cmd": m_dot_steam_cmd_kg_s,
            "p_sec_MPa": s["p_s"] / 1e6,
            "delta_h_MJkg": delta_h / 1e6,
            "q_ms_MW": self.q_ms_last_W / 1e6,
            "Q_steam_MW": (m_dot_actual * delta_h) / 1e6,
        }

        #h_fg = max(c.delta_h_steam_fw_J_kg, 1.0)

        # Existing cap (boiling-rate-based)
        #m_dot_heat_cap = max(self.q_ms_last_W, 0.0) / h_fg

        # NEW: primary-side energy cap (prevents impossible steam production)
        #Q_primary_W = c.m_dot_primary_nom_kg_s * c.cp_primary_J_kgK * max(T_hot_K - s["T_cold"], 0.0)
        #m_dot_primary_cap = max(Q_primary_W, 0.0) / h_fg

        # Combine caps
        #m_dot_capacity = min(m_dot_pressure_cap, m_dot_heat_cap, m_dot_primary_cap)
        #m_dot_actual = min(m_dot_steam_cmd_kg_s, m_dot_capacity)

        if m_dot_actual < 0.0:
            m_dot_actual = 0.0

        s["M_dot_stm"] = m_dot_actual

        # 2) Advance SGCore physics
        s_new = self.core.step(s)

        # 3) Update stored state
        self._apply_core_update(s_new)

        # 4) Steam enthalpy for turbine at new pressure (ideal: saturated vapor at dome pressure)
        if CP is not None:
            try:
                P = max(s["p_s"], 1e5)
                h_steam_turb = CP.PropsSI("H", "P", P, "Q", 1, "Water")  # J/kg
            except Exception:
                h_steam_turb = c.h_steam_J_kg
        else:
            h_steam_turb = c.h_steam_J_kg

        # Flag when the SG could not deliver the commanded flow
        sg_limited = bool(m_dot_actual + 1.0e-6 < m_dot_steam_cmd_kg_s)

        # 5) Update legacy attributes so old code keeps working
        self._update_legacy_attrs()

        # 6) Return standard interface tuple
        return (
            s["T_cold"],    # primary cold-leg temperature [K]
            m_dot_actual,   # actual steam [kg/s]
            s["p_s"],       # secondary pressure [Pa]
            s["T_s"],       # steam dome temperature [K]
            sg_limited,     # SG limiting flag
            h_steam_turb,        # steam enthalpy [J/kg]
        )
