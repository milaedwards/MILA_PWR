try:
    import CoolProp.CoolProp as CoolProp
except ImportError:  # Allow running without CoolProp in constrained envs
    CoolProp = None

from config import Config


class TurbineModel:
    """
    Condensing steam turbine + simple flow controller.

    Uses Config for:
        - turbine_efficiency
        - generator_efficiency
        - condenser/backpressure outlet_p
    """

    def __init__(self, cfg: Config | None = None):
        # Shared plant config
        self.cfg = cfg or Config()
        self._Psec_int = 0.0  # integral of pressure error (MPa*s)

        # Working fluid
        self.fluid = "Water"

        # Efficiencies (fall back to hard-coded if not in Config)
        self.turbine_efficiency = getattr(
            self.cfg, "turbine_efficiency", 0.7188070892
        )  # [-]
        self.generator_efficiency = getattr(
            self.cfg, "generator_efficiency", 0.90
        )  # [-]

        # Turbine exhaust / condenser pressure [Pa]
        self.outlet_p = getattr(
            self.cfg, "P_condenser_Pa", 9820.53
        )

        # Governor references
        self.nominal_power_MW = getattr(self.cfg, "P_e_nom_MWe", 1122.0)
        self.nominal_m_dot = getattr(self.cfg, "m_dot_steam_nom_kg_s", 1886.188)
        self.max_demand_pu = getattr(self.cfg, "load_demand_max_pu", 1.2)

        self.power_scale = getattr(self.cfg, "turbine_power_scale", None)
        if self.power_scale is None:
            self.power_scale = self._calibrate_power_scale()

    # ------------------------------------------------------
    def _raw_specific_work(self, inlet_p: float, inlet_h: float, outlet_p: float) -> float:
        if CoolProp is None:
            h_fw = getattr(self.cfg, "h_fw_J_kg", 0.0)
            delta_h = max(inlet_h - h_fw, 0.0)
            return self.turbine_efficiency * delta_h

        s_in = CoolProp.PropsSI("S", "P", inlet_p, "H", inlet_h, self.fluid)
        h_out_isentropic = CoolProp.PropsSI("H", "P", outlet_p, "S", s_in, self.fluid)
        h_out_actual = inlet_h - self.turbine_efficiency * (inlet_h - h_out_isentropic)
        return inlet_h - h_out_actual

    def _raw_turbine_power(
        self,
        inlet_p: float,
        inlet_h: float,
        outlet_p: float,
        m_dot_steam: float,
    ) -> float:
        specific_work = self._raw_specific_work(inlet_p, inlet_h, outlet_p)
        return self.generator_efficiency * specific_work * m_dot_steam / 1.0e6

    def _calibrate_power_scale(self) -> float:
        try:
            nominal_power = self._raw_turbine_power(
                inlet_p=getattr(self.cfg, "P_sec_nom_Pa", self.outlet_p),
                inlet_h=getattr(self.cfg, "h_steam_J_kg", 0.0),
                outlet_p=self.outlet_p,
                m_dot_steam=self.nominal_m_dot,
            )
        except Exception:
            return 1.0

        if nominal_power <= 0.0:
            return 1.0

        return self.nominal_power_MW / nominal_power

    def turbine_power(
        self,
        inlet_p: float,   # [Pa]
        inlet_h: float,   # [J/kg]
        outlet_p: float,  # [Pa]
        m_dot_steam: float,  # [kg/s]
    ) -> float:
        """
        Calculate electrical power output [MW] using isentropic thermodynamics.

        Inputs:
            inlet_p    – inlet pressure [Pa]
            inlet_h    – inlet specific enthalpy [J/kg]
            outlet_p   – exhaust pressure [Pa]
            m_dot_steam – mass flow rate [kg/s]
        """

        raw_power = self._raw_turbine_power(
            inlet_p=inlet_p,
            inlet_h=inlet_h,
            outlet_p=outlet_p,
            m_dot_steam=m_dot_steam,
        )

        return raw_power * self.power_scale

    # ------------------------------------------------------
    def update_mass_flow_rate(
        self,
        m_dot_steam: float,  # [kg/s] current commanded flow
        m_dot_target: float,  # [kg/s] demand-derived target
        dt: float,            # [s]
    ) -> float:
        """
        Apply step & ramp limits from NRC description:
            - up to 10%/s step
            - up to 5%/min ramp

        Drives the commanded steam flow toward the load-demand target
        without directly tying turbine power to the demand signal.
        """

        # Demand error expressed as flow difference
        flow_error = m_dot_target - m_dot_steam

        # Rate limits referenced to nominal flow so that we can move from
        # zero flow even when the turbine is initially offline.
        m_dot_step = self.nominal_m_dot * 0.10 * dt         # [kg/s]
        m_dot_ramp = self.nominal_m_dot * 0.05 * dt / 60.0  # [kg/s]
        m_dot_limit = m_dot_step + m_dot_ramp

        # Clamp the update to the physical slew rates
        m_dot_update = max(-m_dot_limit, min(m_dot_limit, flow_error))

        m_dot_new = m_dot_steam + m_dot_update

        # Prevent non-physical flow
        if m_dot_new < 0.0:
            m_dot_new = 0.0

        return m_dot_new

    # ------------------------------------------------------
    def _demand_to_flow(self, power_dem: float) -> float:
        """Map the requested electrical load to a steam-flow target."""

        denom = max(self.nominal_power_MW, 1.0)
        demand_pu = power_dem / denom
        demand_pu = max(0.0, min(demand_pu, self.max_demand_pu))
        return demand_pu * self.nominal_m_dot

    # ======================================================
    #                   MAIN STEP FUNCTION
    # ======================================================
    def step(
        self,
        inlet_h: float,    # [J/kg] steam enthalpy from SG
        inlet_p: float,    # [Pa] steam pressure from SG
        #m_dot_steam: float,  # [kg/s] current turbine flow
        m_dot_cmd: float,
        m_dot_actual: float,
        power_dem: float,    # [MW] demanded power (grid)
        dt: float,           # [s] timestep
    ) -> tuple[float, float]:
        """
        Advance turbine one timestep.

        Returns:
            power_output_MW [MW]
            m_dot_steam_new [kg/s]
        """

        # 1) Compute electrical output at current flow
        power_output = self.turbine_power(
            inlet_p=inlet_p,
            inlet_h=inlet_h,
            outlet_p=self.outlet_p,
            #m_dot_steam=m_dot_steam,
            m_dot_steam=m_dot_actual
        )
        #print(f"The power output by the turbine is {power_output:.7f} MWe")

        # 2) Convert demand to a valve/governor steam-flow target
        m_dot_target = self._demand_to_flow(power_dem)

        P_set = getattr(self.cfg, "P_sec_set_Pa", self.cfg.P_sec_nom_Pa)
        eP_MPa = (P_set - inlet_p) / 1e6

        self._Psec_int += eP_MPa * dt

        lim = getattr(self.cfg, "P_int_limit", getattr(self.cfg, "Psec_int_limit", 5.0))
        Kp = getattr(self.cfg, "Kp_P_mdot_per_MPa", getattr(self.cfg, "Kp_Psec_mdot", 200.0))
        Ki = getattr(self.cfg, "Ki_P_mdot_per_MPa_s", getattr(self.cfg, "Ki_Psec_mdot", 5.0))

        #lim = getattr(self.cfg, "Psec_int_limit", 5.0)
        self._Psec_int = max(-lim, min(lim, self._Psec_int))

        #Kp = getattr(self.cfg, "Kp_Psec_mdot", 200.0)
        #Ki = getattr(self.cfg, "Ki_Psec_mdot", 5.0)

        m_dot_target -= Kp * eP_MPa + Ki * self._Psec_int

        # 3) Rate-limit the commanded flow toward the target
        m_dot_cmd_next = self.update_mass_flow_rate(
            #m_dot_steam=m_dot_steam,
            m_dot_steam=m_dot_cmd,
            m_dot_target=m_dot_target,
            dt=dt,
        )
        #print(
        #    "The new steam mass flow rate is"
        #    f" {m_dot_cmd_next:.7f} kg/s (target {m_dot_target:.7f} kg/s)\n"
        #)

        return power_output, m_dot_cmd_next
