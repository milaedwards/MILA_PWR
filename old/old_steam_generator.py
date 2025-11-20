try:
    import CoolProp.CoolProp as clp
except ImportError:
    clp = None

from old.old_config import TauConfig


class SG:
    """
    Steam generator + primary loop temperature and pressure model.

    step(T_hot, dt, m_dot_primary) → (T_cold [K], m_dot_steam_kg_s [kg/s])
    """

    def __init__(self, cfg: TauConfig | None = None):
        self.cfg = cfg if cfg is not None else TauConfig()
        c = self.cfg

        # ----- timestep [s] -----
        self.dt_default = self._init_dt_default()

        # ----- RCS taus [s] -----
        self.tau_hot  = c.tau_hot_s      # hot leg [s]
        self.tau_sgi  = c.tau_sgi_s      # SG inlet plenum [s]
        self.tau_sgu  = c.tau_sgu_s      # SG outlet plenum [s]
        self.tau_cold = c.tau_cold_s     # cold leg [s]
        self.tau_rxi  = c.tau_rxi_s      # lower plenum / core inlet [s]

        # ----- SG global taus [s] (split here) -----
        tau_pm = c.tau_pm_s              # primary↔metal [s]
        tau_mp = c.tau_mp_s              # metal↔primary [s]
        tau_ms = c.tau_ms_s              # metal↔secondary [s]

        self.tau_p1  = tau_pm            # primary lump 1 [s]
        self.tau_p2  = tau_pm            # primary lump 2 [s]
        self.tau_pm1 = tau_pm            # p1→m1 [s]
        self.tau_pm2 = tau_pm            # p2→m2 [s]
        self.tau_mp1 = tau_mp            # m1→p1 [s]
        self.tau_mp2 = tau_mp            # m2→p2 [s]
        self.tau_ms1 = tau_ms            # m1→secondary [s]
        self.tau_ms2 = tau_ms            # m2→secondary [s]

        # ----- SG conductances [W/K] -----
        G_ms = c.G_ms_W_K                # metal→secondary total [W/K]
        self.G_ms1_W_K = 0.5 * G_ms      # node 1 [W/K]
        self.G_ms2_W_K = 0.5 * G_ms      # node 2 [W/K]

        # ----- flow enthalpy term [J/kg] -----
        self.H_ws_minus_cpTfw_J_kg = c.H_ws_minus_cpTfw_J_kg

        # ----- steam pressure [Pa] -----
        self.p_s_Pa = c.P_SG_sec_Pa

        # ----- Ks [J/Pa] -----
        self.K_s_Pa_s = self._compute_Ks_mixture()

        # ----- steam flow [kg/s] -----
        self.mdot_steam_nom_kg_s = c.mdot_fw_kg_s
        self.m_dot_steam_kg_s = self.mdot_steam_nom_kg_s

        # ----- temperatures [K] -----
        self._init_temperature_states()

        # Primary-side mass flow [kg/s] (input each step; default 0)
        self.m_dot_primary_kg_s = 0.0

    # ================= init helpers =================

    def _init_dt_default(self) -> float:
        c = self.cfg
        # Prefer a nominal dt from config if available; otherwise default to 1.0 s
        return float(getattr(c, "dt_default_s", 1.0))

    def _init_temperature_states(self) -> None:
        c = self.cfg

        # Initialize primary loop temperatures from nominal hot/cold values
        self.T_hot  = c.T_hot_K           # reactor outlet / hot leg [K]
        self.T_sgi  = self.T_hot          # SG inlet [K]
        self.T_sgu  = self.T_hot          # SG outlet [K]
        self.T_cold = c.T_cold_K          # cold leg [K]
        self.T_rxi  = self.T_cold         # core inlet [K]

        # Steam-generator internal nodes
        self.T_p1 = self.T_hot            # primary lump 1 [K]
        self.T_p2 = self.T_hot            # primary lump 2 [K]
        self.T_m1 = self.T_hot            # metal node 1 [K]
        self.T_m2 = self.T_hot            # metal node 2 [K]

    # ================= lag helpers =================

    @staticmethod
    def _lag(T_node: float, T_src: float, tau: float, dt: float) -> float:
        return T_node + (T_src - T_node) * (dt / tau)

    @staticmethod
    def _two_lag(
        T_node: float,
        T_in: float,
        tau_in: float,
        T_out: float,
        tau_out: float,
        dt: float,
    ) -> float:
        dTdt = (T_in - T_node) / tau_in - (T_node - T_out) / tau_out
        return T_node + dTdt * dt

    # ============ Ks mixture ============

    def _compute_Ks_mixture(self) -> float:
        c = self.cfg
        if clp is None:
            return 1.0e9

        P = c.P_SG_sec_Pa        # [Pa]
        V_steam = c.V_steam_m3   # [m³]
        m_tot = c.m_sec_water_kg # [kg]

        def h_mix_at_P(P_local: float):
            rho_w = clp.PropsSI("D", "P", P_local, "Q", 0.0, "Water")  # [kg/m³]
            rho_s = clp.PropsSI("D", "P", P_local, "Q", 1.0, "Water")  # [kg/m³]
            h_w = clp.PropsSI("H", "P", P_local, "Q", 0.0, "Water")    # [J/kg]
            h_s = clp.PropsSI("H", "P", P_local, "Q", 1.0, "Water")    # [J/kg]

            m_s = rho_s * V_steam          # [kg]
            m_w = max(m_tot - m_s, 0.0)    # [kg]
            m_ws = max(m_w + m_s, 1.0)     # [kg]

            x = m_s / m_ws                 # [-]
            h_mix = (1.0 - x) * h_w + x * h_s  # [J/kg]
            return h_mix, m_ws

        dP = 1.0e4
        h_minus, m_ws = h_mix_at_P(P - dP)
        h_plus, _ = h_mix_at_P(P + dP)

        dh_dp = (h_plus - h_minus) / (2.0 * dP)  # [J/kg/Pa]
        K_s = m_ws * dh_dp                       # [J/Pa]
        return max(K_s, 1.0e6)

    # ======== dt / flow helpers ========

    def _select_dt(self, dt: float | None) -> float:
        return float(dt) if dt is not None else self.dt_default

    # ============ reactor chain ============

    def step_reactor_chain(self, T_hot_in: float, dt: float) -> float:
        """
        Propagate primary-loop temperatures from reactor outlet to core inlet.

        Inputs:
            T_hot_in [K] - reactor outlet / hot-leg source temperature
            dt      [s] - timestep

        Returns:
            T_cold [K] - cold-leg temperature to core inlet
        """
        # Hot leg from reactor outlet
        self.T_hot  = self._lag(self.T_hot,  T_hot_in,   self.tau_hot,  dt)
        self.T_sgi  = self._lag(self.T_sgi,  self.T_hot, self.tau_sgi, dt)
        self.T_sgu  = self._lag(self.T_sgu,  self.T_p2,  self.tau_sgu, dt)
        self.T_cold = self._lag(self.T_cold, self.T_sgu, self.tau_cold, dt)
        self.T_rxi  = self._lag(self.T_rxi,  self.T_cold, self.tau_rxi, dt)

        return self.T_cold
    def _compute_Ts_from_pressure(self) -> float:
        """
        Compute secondary saturation temperature from current steam pressure.

        If CoolProp is available, use it; otherwise fall back to a nominal
        saturation temperature from config (T_sec_sat_K) or, as a last resort,
        the nominal hot-leg temperature.
        """
        c = self.cfg

        if clp is not None:
            try:
                return clp.PropsSI("T", "P", self.p_s_Pa, "Q", 1.0, "Water")
            except Exception:
                # If CoolProp fails for any reason, fall back to config
                pass

        return float(getattr(c, "T_sec_sat_K", c.T_hot_K))

    # ============ SG thermal chain ============

    def step_sg_chain(self, T_s: float, dt: float) -> None:
        self.T_p1 = self._two_lag(self.T_p1, self.T_sgi, self.tau_p1, self.T_m1, self.tau_pm1, dt)
        self.T_p2 = self._two_lag(self.T_p2, self.T_p1,  self.tau_p2, self.T_m2, self.tau_pm2, dt)
        self.T_m1 = self._two_lag(self.T_m1, self.T_p1,  self.tau_mp1, T_s,        self.tau_ms1, dt)
        self.T_m2 = self._two_lag(self.T_m2, self.T_p2,  self.tau_mp2, T_s,        self.tau_ms2, dt)

    # ============ steam pressure ============

    def _step_steam_pressure(self, T_s: float, dt: float) -> float:
        q_ms1 = self.G_ms1_W_K * (self.T_m1 - T_s)                     # [W]
        q_ms2 = self.G_ms2_W_K * (self.T_m2 - T_s)                     # [W]
        heat_term = q_ms1 + q_ms2                                      # [W]

        flow_term = self.m_dot_steam_kg_s * self.H_ws_minus_cpTfw_J_kg # [W]

        dpdt = (heat_term - flow_term) / self.K_s_Pa_s                 # [Pa/s]
        self.p_s_Pa += dpdt * dt
        return self.p_s_Pa

    # ============ unified step ============

    def step(
        self,
        T_hot: float,
        dt: float | None = None,
        m_dot_primary_kg_s: float | None = None,
    ) -> tuple[float, float]:
        """
        Steam-generator + primary-loop interface.

        Inputs:
            T_hot              [K]  - reactor outlet / hot-leg temperature
            dt                 [s]  - timestep; if None, uses dt_default
            m_dot_primary_kg_s [kg/s] - primary-side mass flow (RCS loop)

        Returns:
            T_cold             [K]  - cold-leg temperature to core inlet
            m_dot_steam_kg_s   [kg/s] - steam mass flow to the turbine / secondary
        """
        dt_eff = self._select_dt(dt)

        # Store primary mass flow (currently not used directly in the
        # internal thermal model but kept for future extensions and export)
        if m_dot_primary_kg_s is not None:
            self.m_dot_primary_kg_s = float(m_dot_primary_kg_s)

        # Derive secondary saturation temperature from steam pressure
        T_s = self._compute_Ts_from_pressure()

        # Update SG internals and primary-loop temperature chain
        self.step_sg_chain(T_s, dt_eff)
        T_cold = self.step_reactor_chain(T_hot, dt_eff)

        # Update steam pressure based on current temperatures and flow
        self._step_steam_pressure(T_s, dt_eff)

        # For now, keep steam flow at nominal unless overridden elsewhere
        self.m_dot_steam_kg_s = self.mdot_steam_nom_kg_s

        return T_cold, self.m_dot_steam_kg_s
