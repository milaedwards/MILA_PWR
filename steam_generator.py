from dataclasses import dataclass, field
import math
from config import Config

@dataclass
class SteamGenerator:
    cfg: Config

    # Secondary header states
    P_sec_Pa: float = field(init=False)  # [Pa] dynamic
    T_sec_K: float = field(init=False)   # [K] fixed unless sg_dT_dP_K_per_Pa used

    # Legacy / compatibility attributes
    T_metal_K: float = field(init=False)        # [K] placeholder
    m_dot_steam_kg_s: float = field(init=False) # [kg/s]

    # Debug (not required by ICSystem)
    m_dot_steam_A_kg_s: float = field(init=False)  # [kg/s]
    m_dot_steam_B_kg_s: float = field(init=False)  # [kg/s]
    T_cold_A_K: float = field(init=False)          # [K]
    T_cold_B_K: float = field(init=False)          # [K]
    
    # Internal valve mapping diagnostics
    steam_valve_u: float = field(init=False)        # [-] internal valve signal 0..1
    m_dot_valve_kg_s: float = field(init=False)     # [kg/s] valve-limited total demand
    valve_saturated: bool = field(init=False)       # [-] True if valve signal u is clamped at 1.0
    m_dot_mismatch_kg_s: float = field(init=False)  # [kg/s] (m_cmd_total - m_total) used for P_sec

    # Valve / demand lag (command side)
    m_cmd_dyn_kg_s: float = field(init=False)        # [kg/s] lagged total steam command used by SG
    tau_valve_s: float = field(init=False)           # [s] first-order lag time constant

    # Steam production / separator lag (delivered side)
    m_dot_dyn_kg_s: float = field(init=False)        # [kg/s] lagged delivered steam flow
    tau_sep_s: float = field(init=False)             # [s] steam separator / dome lag time constant
  
    def __post_init__(self):
        c = self.cfg

        # Initialize secondary header
        self.P_sec_Pa = c.P_sec_nom_Pa  # [Pa]
        if (not math.isfinite(self.P_sec_Pa)) or (self.P_sec_Pa <= 0.0):
            self.P_sec_Pa = 5.764e6  # [Pa] fallback nominal SG pressure

        self.T_sec_K = c.T_sec_nom_K    # [K]
        if (not math.isfinite(self.T_sec_K)) or (self.T_sec_K <= 0.0):
            self.T_sec_K = 560.0  # [K] fallback nominal steam temp

        # Initialize legacy fields
        self.T_metal_K = self._safe_num(getattr(c, "T_metal_nom_K", None), 560.0)
        if (not math.isfinite(self.T_metal_K)) or (self.T_metal_K <= 0.0):
            self.T_metal_K = 560.0

        self.m_dot_steam_kg_s = self._safe_num(getattr(c, "m_dot_steam_nom_kg_s", None), 0.0)
        self.m_dot_steam_kg_s = self._safe_pos(self.m_dot_steam_kg_s)
        if self.m_dot_steam_kg_s <= 0.0:
            self.m_dot_steam_kg_s = 1.0

        # Initialize per-loop debug fields
        self.m_dot_steam_A_kg_s = 0.5 * self.m_dot_steam_kg_s
        self.m_dot_steam_B_kg_s = 0.5 * self.m_dot_steam_kg_s
        self.T_cold_A_K = self._safe_num(getattr(c, "T_cold_nom_K", None), 550.0)
        self.T_cold_B_K = self._safe_num(getattr(c, "T_cold_nom_K", None), 550.0)
        
        # Initialize valve mapping diagnostics
        self.steam_valve_u = 1.0
        self.m_dot_valve_kg_s = self.m_dot_steam_kg_s
        self.valve_saturated = False
        self.m_dot_mismatch_kg_s = 0.0

        # Valve lag init (default 2 s if not provided in Config)
        self.tau_valve_s = self._safe_num(getattr(c, "sg_tau_valve_s", None), 2.0)
        if (not math.isfinite(self.tau_valve_s)) or (self.tau_valve_s <= 0.0):
            self.tau_valve_s = 2.0

        # Start lagged command at nominal so first transient is clean
        self.m_cmd_dyn_kg_s = self._safe_pos(self.m_dot_steam_kg_s)

        # Steam production / separator lag init (default 8 s if not provided in Config)
        self.tau_sep_s = self._safe_num(getattr(c, "sg_tau_sep_s", None), 8.0)
        if (not math.isfinite(self.tau_sep_s)) or (self.tau_sep_s <= 0.0):
            self.tau_sep_s = 8.0

        # Start delivered flow at nominal
        self.m_dot_dyn_kg_s = self._safe_pos(self.m_dot_steam_kg_s)

    @staticmethod
    def _safe_pos(x: float) -> float:
        if (not math.isfinite(x)) or (x < 0.0):
            return 0.0
        return x

    @staticmethod
    def _safe_num(x, fallback: float) -> float:
        if (x is None) or (not math.isfinite(x)):
            return fallback
        return x

    def _loop_step(
        self,
        T_hot_K: float,
        m_cmd_kg_s: float,
        m_dot_pri_kg_s: float,
        cp_pri_J_kgK: float,
        T_fallback_K: float,
        delta_h_J_kg: float,
        m_cap_kg_s: float,
        T_min_K: float,
    ):

        # Available primary heat above minimum cold-leg temperature (physical floor)
        dT_avail = T_hot_K - T_min_K  # [K]
        if not math.isfinite(dT_avail):
            dT_avail = 0.0
        if dT_avail < 0.0:
            dT_avail = 0.0

        Q_avail_W = (
            self._safe_pos(m_dot_pri_kg_s)
            * max(self._safe_pos(cp_pri_J_kgK), 0.0)
            * dT_avail
        )  # [W]

        # Maximum steam flow supported by available heat
        m_max_by_heat = Q_avail_W / max(self._safe_pos(delta_h_J_kg), 1.0e5)  # [kg/s]

        # Actual steam flow (capacity-limited)
        m_cmd = self._safe_pos(m_cmd_kg_s)
        m_cap = self._safe_pos(m_cap_kg_s)
        m_dot_stm = min(m_cmd, m_max_by_heat, m_cap)  # [kg/s]
        m_dot_stm = self._safe_pos(m_dot_stm)

        loop_limited = (m_dot_stm + 1e-9 < m_cmd)

        # Primary outlet temperature from steam production
        Q_dot_W = m_dot_stm * max(self._safe_pos(delta_h_J_kg), 1.0e5)  # [W]
        denom = max(self._safe_pos(m_dot_pri_kg_s), 1e-6) * max(self._safe_pos(cp_pri_J_kgK), 1000.0)  # [W/K]
        dT = Q_dot_W / denom  # [K]

        T_cold_K = T_hot_K - dT  # [K]

        # Guard rails
        if not math.isfinite(T_cold_K):
            T_cold_K = T_fallback_K

        # Ensure bounds are not inverted (must happen before clamping)
        if T_min_K > T_hot_K:
            T_min_K = T_hot_K

        if T_cold_K > T_hot_K:
            T_cold_K = T_hot_K
        if T_cold_K < T_min_K:
            T_cold_K = T_min_K

        return T_cold_K, m_dot_stm, loop_limited

    def step(self, T_hot_K: float, m_dot_steam_cmd_kg_s: float, dt: float):
        """
        Inputs:
          T_hot_K [K]: plant hot-leg temperature into SGs
          m_dot_steam_cmd_kg_s [kg/s]: turbine-requested TOTAL steam flow
          dt [s]: timestep

        Returns:
          (T_cold_K, m_dot_actual, P_sec_Pa, T_sec_K, sg_limited, h_steam_J_kg)
        """
        c = self.cfg

        # Pull required config numbers
        delta_h = self._safe_num(getattr(c, "delta_h_steam_fw_J_kg", None), 1.8e6)  # [J/kg]
        if (not math.isfinite(delta_h)) or (delta_h <= 0.0):
            delta_h = 1.8e6
            
        h_steam = self._safe_num(getattr(c, "h_steam_J_kg", None), 2.8e6)  # [J/kg]
        if (not math.isfinite(h_steam)) or (h_steam <= 0.0):
            h_steam = 2.8e6

        m_dot_pri_total = self._safe_num(getattr(c, "m_dot_primary_nom_kg_s", None), float("nan"))  # [kg/s]
        if (not math.isfinite(m_dot_pri_total)) or (m_dot_pri_total <= 0.0):
            m_dot_pri_total = 1.0e4  # [kg/s]

        cp_pri = self._safe_num(getattr(c, "cp_coolant_J_kgK", None), 5500.0)  # [J/kg-K]
        if (not math.isfinite(cp_pri)) or (cp_pri <= 0.0):
            cp_pri = 5500.0

        T_cold_target = self._safe_num(getattr(c, "T_cold_nom_K", None), 550.0)  # [K]
        if (not math.isfinite(T_cold_target)) or (T_cold_target <= 0.0):
            T_cold_target = 550.0
        if not math.isfinite(T_hot_K):
            T_hot_K = T_cold_target

        T_min = self._safe_num(getattr(c, "T_cold_min_K", None), 450.0)  # [K]
        if (not math.isfinite(T_min)) or (T_min <= 0.0):
            T_min = 450.0

        # Prevent inverted constraints if config is wrong
        if T_min > T_cold_target:
            T_min = T_cold_target

        Th_safe = T_hot_K
        if Th_safe < T_min:
            Th_safe = T_min
            T_hot_K = Th_safe

        # Steam demand -> internal valve signal u
        # Treat incoming command as turbine-requested TOTAL steam flow [kg/s]
        m_dem_total = m_dot_steam_cmd_kg_s
        if (not math.isfinite(m_dem_total)) or (m_dem_total < 0.0):
            m_dem_total = 0.0

        # Nominal steam flow used for normalization / valve scaling
        m_nom_total = self._safe_num(getattr(c, "m_dot_steam_nom_kg_s", None), self.m_dot_steam_kg_s)
        if (not math.isfinite(m_nom_total)) or (m_nom_total <= 0.0):
            m_nom_total = max(self._safe_pos(self.m_dot_steam_kg_s), 1.0)

        # Valve signal u in [0,1]
        u_raw = 0.0
        if (m_nom_total > 0.0) and math.isfinite(m_dem_total):
            u_raw = m_dem_total / m_nom_total
        u = max(0.0, min(1.0, u_raw))
        self.valve_saturated = (u >= 1.0 - 1e-12)

        # Valve-limited steam demand
        m_cmd_total = u * m_nom_total
        if not math.isfinite(m_cmd_total):
            m_cmd_total = 0.0

        # Internal diagnostics (pre-lag)
        self.steam_valve_u = u
        self.m_dot_valve_kg_s = m_cmd_total

        # Valve / demand lag (creates mismatch so P_sec can move)
        # First-order lag: m_cmd_dyn follows m_cmd_total with time constant tau_valve_s
        if (math.isfinite(dt)) and (dt > 0.0):
            tau = max(self._safe_pos(self.tau_valve_s), 1e-6)
            alpha = dt / tau
            if alpha > 1.0:
                alpha = 1.0
            self.m_cmd_dyn_kg_s += (m_cmd_total - self.m_cmd_dyn_kg_s) * alpha
        else:
            self.m_cmd_dyn_kg_s = m_cmd_total

        # Use lagged command for the rest of SG math
        m_cmd_total = self._safe_pos(self.m_cmd_dyn_kg_s)

        # Split into Loop A / B
        fA = self._safe_num(getattr(c, "sg_loopA_frac", None), 0.5)  # [-]
        if (not math.isfinite(fA)) or (fA <= 0.0) or (fA >= 1.0):
            fA = 0.5
        fB = 1.0 - fA

        m_pri_A = self._safe_pos(m_dot_pri_total) * fA  # [kg/s]
        m_pri_B = self._safe_pos(m_dot_pri_total) * fB  # [kg/s]

        m_cmd_A = self._safe_pos(m_cmd_total) * fA  # [kg/s]
        m_cmd_B = self._safe_pos(m_cmd_total) * fB  # [kg/s]

        m_cap_A = self._safe_pos(m_nom_total) * fA  # [kg/s]
        m_cap_B = self._safe_pos(m_nom_total) * fB  # [kg/s]

        # Solve each loop algebraically
        T_cold_A, m_A, _ = self._loop_step(
            T_hot_K, m_cmd_A, m_pri_A, cp_pri, T_cold_target, delta_h, m_cap_A, T_min
        )
        T_cold_B, m_B, _ = self._loop_step(
            T_hot_K, m_cmd_B, m_pri_B, cp_pri, T_cold_target, delta_h, m_cap_B, T_min
        )

        m_total_alg = self._safe_pos(m_A) + self._safe_pos(m_B)  # [kg/s] algebraic steam generation

        # Steam production / separator lag (delivered flow)
        # This prevents m_total from instantly equaling m_cmd_total,
        # so mismatch exists and P_sec responds.
        if (math.isfinite(dt)) and (dt > 0.0):
            tau = max(self._safe_pos(self.tau_sep_s), 1e-6)
            alpha = dt / tau
            if alpha > 1.0:
                alpha = 1.0
            self.m_dot_dyn_kg_s += (m_total_alg - self.m_dot_dyn_kg_s) * alpha
        else:
            self.m_dot_dyn_kg_s = m_total_alg

        m_total = self._safe_pos(self.m_dot_dyn_kg_s)  # [kg/s] delivered steam flow used everywhere below

        denom = max(self._safe_pos(m_pri_A) + self._safe_pos(m_pri_B), 1e-6)
        T_cold_A = self._safe_num(T_cold_A, T_cold_target)
        T_cold_B = self._safe_num(T_cold_B, T_cold_target)
        m_pri_A_pos = self._safe_pos(m_pri_A)
        m_pri_B_pos = self._safe_pos(m_pri_B)
        T_cold = (m_pri_A_pos * T_cold_A + m_pri_B_pos * T_cold_B) / denom  # [K]
        
        # Final physical clamp on mixed cold-leg temperature
        if T_cold > Th_safe:
            T_cold = Th_safe
        if T_cold < T_min:
            T_cold = T_min

        # SG is limiting if it cannot meet valve-limited demand
        tol = 1e-6 + 1e-4 * max(m_cmd_total, 0.0)
        
        # Limitation should reflect physical ability
        sg_limited = (m_total_alg < (m_cmd_total - tol))

        P_nom = c.P_sec_nom_Pa  # [Pa]
        if (not math.isfinite(P_nom)) or (P_nom <= 0.0):
            P_nom = 5.764e6  # [Pa] fallback typical SG pressure

        Kp_P_per_kg_s = self._safe_num(getattr(c, "sg_KpP_Pa_per_kg_s", None), 3000.0)  # [Pa/(kg/s)]
        Kp_P_per_kg_s = max(Kp_P_per_kg_s, 0.0)
        tau_P_s = self._safe_num(getattr(c, "sg_tauP_s", None), 5.0)                    # [s]
        dP_max_Pa = self._safe_num(getattr(c, "sg_dPmax_Pa", None), 0.5e6)              # [Pa]
        P_min_Pa = self._safe_num(getattr(c, "sg_Pmin_Pa", None), 1.0e5)                # [Pa]
        P_max_Pa = self._safe_num(getattr(c, "sg_Pmax_Pa", None), 25.0e6)               # [Pa]
        if (not math.isfinite(P_min_Pa)) or (P_min_Pa <= 0.0):
            P_min_Pa = 1.0e5
        
        if (not math.isfinite(P_max_Pa)) or (P_max_Pa <= P_min_Pa):
            P_max_Pa = max(P_min_Pa * 1.1, 25.0e6)

        if (not math.isfinite(tau_P_s)) or (tau_P_s <= 0.0):
            tau_P_s = 1.0
        
        if (not math.isfinite(dP_max_Pa)) or (dP_max_Pa < 0.0):
            dP_max_Pa = 0.0

        mismatch = (m_cmd_total - m_total)  # [kg/s]
        if not math.isfinite(mismatch):
            mismatch = 0.0
        self.m_dot_mismatch_kg_s = mismatch

        P_sec_cmd = P_nom - Kp_P_per_kg_s * mismatch  # [Pa]
        if not math.isfinite(P_sec_cmd):
            P_sec_cmd = P_nom
      
        # Clamp around nominal
        P_lo = P_nom - dP_max_Pa
        P_hi = P_nom + dP_max_Pa
        if P_lo > P_hi:
            P_lo, P_hi = P_hi, P_lo
        if P_sec_cmd < P_lo:
            P_sec_cmd = P_lo
        if P_sec_cmd > P_hi:
            P_sec_cmd = P_hi

        if P_sec_cmd < P_min_Pa:
            P_sec_cmd = P_min_Pa
        if P_sec_cmd > P_max_Pa:
            P_sec_cmd = P_max_Pa

        # First-order lag toward command
        if (math.isfinite(dt)) and (dt > 0.0):
            alpha = dt / tau_P_s
            if alpha > 1.0:
                alpha = 1.0
            self.P_sec_Pa += (P_sec_cmd - self.P_sec_Pa) * alpha

        # Absolute clamps
        if not math.isfinite(self.P_sec_Pa):
            self.P_sec_Pa = P_nom
        elif self.P_sec_Pa < P_min_Pa:
            self.P_sec_Pa = P_min_Pa
        if self.P_sec_Pa > P_max_Pa:
            self.P_sec_Pa = P_max_Pa

        dT_dP = self._safe_num(getattr(c, "sg_dT_dP_K_per_Pa", None), 0.0)  # [K/Pa]
        if not math.isfinite(dT_dP):
            dT_dP = 0.0
        # Prevent extreme slopes from unit mistakes (K/Pa)
        if abs(dT_dP) > 1.0e-3:
            dT_dP = 0.0
        self.T_sec_K = c.T_sec_nom_K + dT_dP * (self.P_sec_Pa - P_nom)  # [K]
        if not math.isfinite(self.T_sec_K):
            self.T_sec_K = c.T_sec_nom_K

        if self.T_sec_K < 200.0:
            self.T_sec_K = 200.0
        if self.T_sec_K > 1200.0:
            self.T_sec_K = 1200.0


        # Update legacy fields
        self.m_dot_steam_kg_s = m_total
        self.m_dot_steam_A_kg_s = m_A
        self.m_dot_steam_B_kg_s = m_B
        self.T_cold_A_K = T_cold_A
        self.T_cold_B_K = T_cold_B

        # Placeholder metal temperature
        Th = self._safe_num(T_hot_K, T_cold_target)
        Tc = self._safe_num(T_cold, T_cold_target)
        self.T_metal_K = 0.5 * (Th + Tc)  # [K]

        # Final output guard rails (prevents NaNs propagating into ICSystem)
        if not math.isfinite(T_cold):
            T_cold = T_cold_target
        if not math.isfinite(m_total):
            m_total = 0.0

        return (
            T_cold,        # [K]
            m_total,       # [kg/s]
            self.P_sec_Pa, # [Pa] dynamic (Level 3)
            self.T_sec_K,  # [K] fixed unless sg_dT_dP_K_per_Pa used
            sg_limited,    # [-]
            h_steam,       # [J/kg]
        )