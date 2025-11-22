# sg.py

from dataclasses import dataclass, field

try:
    import CoolProp.CoolProp as clp
except ImportError:
    clp = None

from config_2 import Config


@dataclass
class SG:
    """
    Delta-125 Steam Generator + primary loop segment.
    Stateless: all dynamics are in the input/output state dicts.
    """

    cfg: Config = field(default_factory=Config)  # [-] full plant / SG configuration

    # Linear Ts warning threshold
    Ts_linear_warn_K: float = 2.0  # [K] max deviation allowed for linear Ts check

    # Precomputed from cfg in __post_init__
    tau_hot: float = field(init=False)   # [s] hot-leg hydraulic tau
    tau_sgi: float = field(init=False)   # [s] SG inlet plenum tau
    tau_sgu: float = field(init=False)   # [s] SG outlet plenum tau
    tau_cold: float = field(init=False)  # [s] cold-leg hydraulic tau
    tau_rxi: float = field(init=False)   # [s] reactor inlet hydraulic tau

    tau_p1: float = field(init=False)    # [s] SG primary lump 1 tau
    tau_p2: float = field(init=False)    # [s] SG primary lump 2 tau
    tau_pm1: float = field(init=False)   # [s] primary→metal node 1 tau
    tau_pm2: float = field(init=False)   # [s] primary→metal node 2 tau
    tau_mp1: float = field(init=False)   # [s] metal 1 coupling tau
    tau_mp2: float = field(init=False)   # [s] metal 2 coupling tau
    tau_ms1: float = field(init=False)   # [s] metal→secondary node 1 tau
    tau_ms2: float = field(init=False)   # [s] metal→secondary node 2 tau

    G_ms1_W_K: float = field(init=False)  # [W/K] metal node 1 → secondary conductance
    G_ms2_W_K: float = field(init=False)  # [W/K] metal node 2 → secondary conductance

    H_ws_minus_cpTfw_J_kg: float = field(init=False)  # [J/kg] h_ws − cp_fw * T_fw
    p_s_nom_Pa: float = field(init=False)             # [Pa] nominal steam pressure

    def __post_init__(self):
        c = self.cfg

        # Hydraulic taus from Config [s]
        self.tau_hot  = c.tau_hot_s   # [s]
        self.tau_sgi  = c.tau_sgi_s   # [s]
        self.tau_sgu  = c.tau_sgu_s   # [s]
        self.tau_cold = c.tau_cold_s  # [s]
        self.tau_rxi  = c.tau_rxi_s   # [s]

        # SG primary / metal taus [s]
        self.tau_p1  = c.tau_pm_s    # [s]
        self.tau_p2  = c.tau_pm_s    # [s]
        self.tau_pm1 = c.tau_pm_s    # [s]
        self.tau_pm2 = c.tau_pm_s    # [s]
        self.tau_mp1 = c.tau_mp_s    # [s]
        self.tau_mp2 = c.tau_mp_s    # [s]
        self.tau_ms1 = c.tau_ms_s    # [s]
        self.tau_ms2 = c.tau_ms_s    # [s]

        # Split metal→secondary conductance into two equal halves [W/K]
        self.G_ms1_W_K = 0.5 * c.G_ms_W_K  # [W/K]
        self.G_ms2_W_K = 0.5 * c.G_ms_W_K  # [W/K]

        # Enthalpy term from config [J/kg]
        self.H_ws_minus_cpTfw_J_kg = c.H_ws_minus_cpTfw_J_kg  # [J/kg]

        # Nominal SG pressure [Pa]
        self.p_s_nom_Pa = c.P_SEC_CONST_Pa  # [Pa]

    # ===================== Helper: first-order lag =====================
    @staticmethod
    def _lag(T_node: float, T_src: float, tau_s: float, dt_s: float) -> float:
        """
        First-order lag: dT/dt = (T_src − T_node) / tau.
        """
        return T_node + (T_src - T_node) * (dt_s / tau_s)

    # ===================== Helper: import state =====================
    def _get_state(self, s: dict) -> dict:
        """
        Normalize incoming state dict, fill defaults from Config.
        Required: 'dt'
        """
        c = self.cfg

        T_rxu  = s.get("T_rxu",  c.T_HOT_INIT_K)   # [K] reactor outlet (hot leg entry)
        T_hot  = s.get("T_hot",  c.T_HOT_INIT_K)   # [K] hot leg
        T_sgi  = s.get("T_sgi",  c.T_HOT_INIT_K)   # [K] SG inlet plenum
        T_p1   = s.get("T_p1",   c.T_HOT_INIT_K)   # [K] SG primary lump 1
        T_p2   = s.get("T_p2",   c.T_COLD_INIT_K)  # [K] SG primary lump 2
        T_m1   = s.get("T_m1",   c.T_HOT_INIT_K)   # [K] SG metal node 1
        T_m2   = s.get("T_m2",   c.T_COLD_INIT_K)  # [K] SG metal node 2
        T_sgu  = s.get("T_sgu",  c.T_COLD_INIT_K)  # [K] SG outlet plenum
        T_cold = s.get("T_cold", c.T_COLD_INIT_K)  # [K] cold leg
        T_rxi  = s.get("T_rxi",  c.T_COLD_INIT_K)  # [K] reactor inlet

        p_s = s.get("p_s", self.p_s_nom_Pa)   # [Pa] steam pressure

        # Steam temperature [K]
        if "T_s" in s:
            T_s = s["T_s"]
        else:
            if clp is not None:
                T_s = clp.PropsSI("T", "P", p_s, "Q", 0.0, "Water")  # [K]
            else:
                T_s = c.T_SAT_SEC_K  # [K] plant-level saturation default

        # Steam mass flow [kg/s], assumed equal to feed flow
        M_dot_stm = s.get("M_dot_stm", c.mdot_fw_kg_s)  # [kg/s]

        if "dt" not in s:
            raise ValueError("state_in['dt'] is required.")
        dt = s["dt"]  # [s]

        return {
            "T_rxu": T_rxu,
            "T_hot": T_hot,
            "T_sgi": T_sgi,
            "T_p1": T_p1,
            "T_p2": T_p2,
            "T_m1": T_m1,
            "T_m2": T_m2,
            "T_sgu": T_sgu,
            "T_cold": T_cold,
            "T_rxi": T_rxi,
            "T_s": T_s,
            "p_s": p_s,
            "M_dot_stm": M_dot_stm,
            "dt": dt,
        }

    # ===================== RXU → hot → SG inlet =====================
    def _step_primary_head(self, T_rxu, T_hot, T_sgi, dt):
        """
        Hot leg and SG inlet plenum.
        """
        T_hot_new = self._lag(T_hot, T_rxu, self.tau_hot, dt)   # [K]
        T_sgi_new = self._lag(T_sgi, T_hot_new, self.tau_sgi, dt)  # [K]
        return T_hot_new, T_sgi_new

    # ===================== SG primary + metal chain =====================
    def _step_sg_chain(self, T_sgi, T_p1, T_p2, T_m1, T_m2, T_s, dt):
        """
        Two primary lumps and two metal nodes in SG thermal chain.
        """
        dTp1_dt = (T_sgi - T_p1) / self.tau_p1 - (T_p1 - T_m1) / self.tau_pm1  # [K/s]
        dTp2_dt = (T_p1  - T_p2) / self.tau_p2 - (T_p2 - T_m2) / self.tau_pm2  # [K/s]
        dTm1_dt = (T_p1  - T_m1) / self.tau_mp1 - (T_m1 - T_s) / self.tau_ms1  # [K/s]
        dTm2_dt = (T_p2  - T_m2) / self.tau_mp2 - (T_m2 - T_s) / self.tau_ms2  # [K/s]

        T_p1_new = T_p1 + dTp1_dt * dt  # [K]
        T_p2_new = T_p2 + dTp2_dt * dt  # [K]
        T_m1_new = T_m1 + dTm1_dt * dt  # [K]
        T_m2_new = T_m2 + dTm2_dt * dt  # [K]

        return T_p1_new, T_p2_new, T_m1_new, T_m2_new

    # ===================== SG outlet → cold → RXI =====================
    def _step_primary_tail(self, T_p2_new, T_sgu, T_cold, T_rxi, dt):
        """
        SG outlet plenum → cold leg → reactor inlet.
        """
        T_sgu_new  = self._lag(T_sgu,  T_p2_new,  self.tau_sgu,  dt)  # [K]
        T_cold_new = self._lag(T_cold, T_sgu_new, self.tau_cold, dt)  # [K]
        T_rxi_new  = self._lag(T_rxi,  T_cold_new, self.tau_rxi, dt)  # [K]
        return T_sgu_new, T_cold_new, T_rxi_new

    # ===================== Ks mixture (compressibility) =====================
    def _compute_Ks_mixture(self, p_s_local: float) -> float:
        """
        Compute effective mixture compressibility coefficient Ks [J/Pa]
        for the SG steam region at pressure p_s_local.
        """
        c = self.cfg
        if clp is None:
            return 1.0e9  # fallback, very stiff

        P = p_s_local          # [Pa]
        V = c.V_steam_m3       # [m^3]
        m_tot = c.m_sec_water_kg  # [kg]

        def h_mix(P_local: float):
            rho_w = clp.PropsSI("D", "P", P_local, "Q", 0.0, "Water")  # [kg/m^3]
            rho_s = clp.PropsSI("D", "P", P_local, "Q", 1.0, "Water")  # [kg/m^3]
            h_w   = clp.PropsSI("H", "P", P_local, "Q", 0.0, "Water")  # [J/kg]
            h_s   = clp.PropsSI("H", "P", P_local, "Q", 1.0, "Water")  # [J/kg]

            m_s = rho_s * V                     # [kg]
            m_w = max(m_tot - m_s, 0.0)         # [kg]
            m_ws = max(m_s + m_w, 1.0e-3)       # [kg]

            x = m_s / m_ws                      # [-]
            h_mix_val = (1.0 - x) * h_w + x * h_s  # [J/kg]
            return h_mix_val, m_ws

        dP = 1.0e4  # [Pa]
        h_low,  m_ws = h_mix(P - dP)  # [J/kg], [kg]
        h_high, _    = h_mix(P + dP)  # [J/kg]

        dhdP = (h_high - h_low) / (2.0 * dP)  # [J/(kg·Pa)]
        Ks = m_ws * dhdP                      # [J/Pa]
        return max(Ks, 1.0e6)

    # ===================== dT_sat/dp (for check) =====================
    def _dTdp_sat(self, p_s: float) -> float:
        """
        Compute dT_sat/dp at given steam pressure [K/Pa].
        """
        if clp is None:
            return 0.0
        dP = 1.0e4  # [Pa]
        T_low  = clp.PropsSI("T", "P", p_s - dP, "Q", 0.0, "Water")  # [K]
        T_high = clp.PropsSI("T", "P", p_s + dP, "Q", 0.0, "Water")  # [K]
        return (T_high - T_low) / (2.0 * dP)  # [K/Pa]

    # ===================== Steam pressure ODE =====================
    def _step_steam_pressure(self, T_m1_new, T_m2_new, T_s_old, p_s_old, M_dot_stm, dt):
        """
        Integrate steam pressure using SG heat balance.
        """
        # Heat from metal to steam [W]
        q_ms1 = self.G_ms1_W_K * (T_m1_new - T_s_old)  # [W]
        q_ms2 = self.G_ms2_W_K * (T_m2_new - T_s_old)  # [W]
        heat_term = q_ms1 + q_ms2                      # [W]

        # Enthalpy carried away with steam [W]
        flow_term = M_dot_stm * self.H_ws_minus_cpTfw_J_kg  # [W]

        # Old Ks, predictor
        Ks_old = self._compute_Ks_mixture(p_s_old)          # [J/Pa]
        dpdt_old = (heat_term - flow_term) / Ks_old         # [Pa/s]
        p_s_mid = p_s_old + dpdt_old * dt                   # [Pa]

        # New Ks, corrector
        Ks_new = self._compute_Ks_mixture(p_s_mid)          # [J/Pa]
        dpdt = (heat_term - flow_term) / Ks_new             # [Pa/s]
        p_s_new = p_s_old + dpdt * dt                       # [Pa]

        return p_s_new

    # ===================== Steam temperature =====================
    def _update_steam_temperature(self, p_s_new, T_s_old):
        """
        Update saturated steam temperature at new pressure [K].
        """
        if clp is not None:
            return clp.PropsSI("T", "P", p_s_new, "Q", 0.0, "Water")  # [K]
        return T_s_old

    # ===================== Main step =====================
    def step(self, state_in: dict) -> dict:
        """
        Advance SG + loop segment one time step.
        Inputs:  state_in dict (must include 'dt')
        Outputs: state_out dict with *_new values.
        """
        s = self._get_state(state_in)

        T_rxu  = s["T_rxu"]   # [K]
        T_hot  = s["T_hot"]   # [K]
        T_sgi  = s["T_sgi"]   # [K]
        T_p1   = s["T_p1"]    # [K]
        T_p2   = s["T_p2"]    # [K]
        T_m1   = s["T_m1"]    # [K]
        T_m2   = s["T_m2"]    # [K]
        T_sgu  = s["T_sgu"]   # [K]
        T_cold = s["T_cold"]  # [K]
        T_rxi  = s["T_rxi"]   # [K]
        T_s    = s["T_s"]     # [K]
        p_s    = s["p_s"]     # [Pa]
        M_dot  = s["M_dot_stm"]  # [kg/s]
        dt     = s["dt"]      # [s]

        # 1) RXU → hot leg → SG inlet
        T_hot_new, T_sgi_new = self._step_primary_head(T_rxu, T_hot, T_sgi, dt)

        # 2) SG primary + metal chain
        T_p1_new, T_p2_new, T_m1_new, T_m2_new = self._step_sg_chain(
            T_sgi_new, T_p1, T_p2, T_m1, T_m2, T_s, dt
        )

        # 3) SG outlet → cold leg → reactor inlet
        T_sgu_new, T_cold_new, T_rxi_new = self._step_primary_tail(
            T_p2_new, T_sgu, T_cold, T_rxi, dt
        )

        # 4) Steam pressure dynamics
        p_s_new = self._step_steam_pressure(
            T_m1_new, T_m2_new, T_s, p_s, M_dot, dt
        )

        # 5) Steam temperature from saturation
        T_s_new = self._update_steam_temperature(p_s_new, T_s)

        # 6) Linear Ts check (optional diagnostic)
        #if clp is not None:
        #    dTdp = self._dTdp_sat(p_s_new)      # [K/Pa]
        #    T_s_lin = dTdp * p_s_new            # [K]
        #    err = abs(T_s_lin - T_s_new)        # [K]
        #    if err > self.Ts_linear_warn_K:
        #        print(
        #            f"[SG WARNING] Ts linear approx deviates by "
        #            f"{err:.2f} K at p_s = {p_s_new:.1f} Pa "
        #            f"(T_lin = {T_s_lin:.2f} K, T_cp = {T_s_new:.2f} K)"
        #        )

        return {
            "T_rxu_new": T_rxu,        # [K]
            "T_hot_new": T_hot_new,    # [K]
            "T_sgi_new": T_sgi_new,    # [K]
            "T_p1_new": T_p1_new,      # [K]
            "T_p2_new": T_p2_new,      # [K]
            "T_m1_new": T_m1_new,      # [K]
            "T_m2_new": T_m2_new,      # [K]
            "T_sgu_new": T_sgu_new,    # [K]
            "T_cold_new": T_cold_new,  # [K]
            "T_rxi_new": T_rxi_new,    # [K]
            "p_s_new": p_s_new,        # [Pa]
            "T_s_new": T_s_new,        # [K]
            "M_dot_stm_new": M_dot,  # [kg/s]
        }