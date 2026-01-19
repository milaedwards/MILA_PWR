# steam_generator.py
# Clean AP1000 Delta-125 Steam Generator model
# Includes:
#   • SGCore         – detailed SG thermal/pressure physics
#   • SteamGenerator – wrapper used by ICSystem
#
# Units are indicated in comments.

from dataclasses import dataclass, field
from typing import Dict

from config import Config

try:
    import CoolProp.CoolProp as CP
except ImportError:
    CP = None


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
        self.tau_ms = c.tau_ms_s

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
        h_fg = max(c.delta_h_steam_fw_J_kg, 1.0e5)  # [J/kg]

        # dT/dt of steam dome [K/s]
        dTdt = (Ts_new - Ts_old) / dt

        # Evaporation rate [kg/s]
        m_evap = q_ms_W / h_fg

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

    def __post_init__(self):
        c = self.cfg
        self.core = SGCore(c)

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

        # Initialize legacy attributes consistently
        self.T_metal_K = c.T_metal_nom_K
        self.T_sec_K = c.T_sec_nom_K
        self.P_sec_Pa = c.P_sec_nom_Pa
        self.m_dot_steam_kg_s = c.m_dot_steam_nom_kg_s

    # ---------------------------------------------------------------
    def step(self, T_hot_K: float, m_dot_steam_cmd_kg_s: float, dt: float):
        """
        Single-step interface used by ICSystem.
        """

        c = self.cfg
        s = self.state

        # 1) Update boundary conditions and flow
        s["T_rxu"] = T_hot_K
        s["M_dot_stm"] = m_dot_steam_cmd_kg_s
        s["dt"] = dt

        # 2) Advance SGCore physics
        s_new = self.core.step(s)

        # 3) Update stored state
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

        # 4) Steam enthalpy for turbine at new pressure
        if CP is not None:
            try:
                # Saturated steam enthalpy at current dome pressure
                h_steam = CP.PropsSI("H", "P", s["p_s"], "Q", 1, "Water")
            except Exception:
                h_steam = c.h_steam_J_kg
        else:
            h_steam = c.h_steam_J_kg

        # For now, no explicit SG capacity limiter:
        sg_limited = False
        m_dot_actual = m_dot_steam_cmd_kg_s

        # 5) Update legacy attributes so old code keeps working
        self.T_metal_K = 0.5 * (s["T_m1"] + s["T_m2"])
        self.T_sec_K = s["T_s"]
        self.P_sec_Pa = s["p_s"]
        self.m_dot_steam_kg_s = m_dot_actual

        # 6) Return standard interface tuple
        return (
            s["T_cold"],    # primary cold-leg temperature [K]
            m_dot_actual,   # actual steam [kg/s]
            s["p_s"],       # secondary pressure [Pa]
            s["T_s"],       # steam dome temperature [K]
            sg_limited,     # SG limiting flag
            h_steam,        # steam enthalpy [J/kg]
        )
