import CoolProp.CoolProp as CoolProp
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

    # ------------------------------------------------------
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

        h_in = inlet_h  # [J/kg]

        # Inlet entropy at (P, H)
        s_in = CoolProp.PropsSI("S", "P", inlet_p, "H", h_in, self.fluid)

        # Isentropic outlet enthalpy
        h_out_isentropic = CoolProp.PropsSI(
            "H", "P", outlet_p, "S", s_in, self.fluid
        )

        # Actual outlet enthalpy with turbine isentropic efficiency
        h_out_actual = h_in - self.turbine_efficiency * (h_in - h_out_isentropic)

        # Specific work [J/kg]
        specific_work = h_in - h_out_actual

        # Electrical power [MW]
        power_output_MW = (
            self.generator_efficiency * specific_work * m_dot_steam / 1.0e6
        )

        return power_output_MW

    # ------------------------------------------------------
    def update_mass_flow_rate(
        self,
        m_dot_steam: float,   # [kg/s] current flow
        power_supplied: float,  # [MW] turbine electrical power
        power_dem: float,       # [MW] requested power
        dt: float,              # [s]
    ) -> float:
        """
        Apply step & ramp limits from NRC description:
            - up to 10%/s step
            - up to 5%/min ramp
        """

        eps = 1.0e-6
        denom = max(abs(power_dem), eps)  # [MW]

        # Power error and relative error
        power_error = power_supplied - power_dem          # [MW]
        power_percent_change = power_error / denom        # [-]

        # Flow limits
        m_dot_step = m_dot_steam * 0.10 * dt         # [kg/s] 10%/s
        m_dot_ramp = m_dot_steam * 0.05 * dt / 60.0  # [kg/s] 5%/min

        # Controller logic
        if power_percent_change > 0.10:
            # Too much power → reduce flow aggressively
            m_dot_update = -(m_dot_step + m_dot_ramp)
        elif power_percent_change < -0.10:
            # Too little power → increase flow aggressively
            m_dot_update = (m_dot_step + m_dot_ramp)
        else:
            # Within ±10% → proportional tweak
            m_dot_update = -power_percent_change * m_dot_steam * dt

        # Apply update
        m_dot_new = m_dot_steam + m_dot_update

        # Prevent non-physical flow
        if m_dot_new < 0.0:
            m_dot_new = 0.0

        return m_dot_new

    # ======================================================
    #                   MAIN STEP FUNCTION
    # ======================================================
    def step(
        self,
        inlet_h: float,    # [J/kg] steam enthalpy from SG
        inlet_p: float,    # [Pa] steam pressure from SG
        m_dot_steam: float,  # [kg/s] current turbine flow
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
            m_dot_steam=m_dot_steam,
        )
        print(f"The power output by the turbine is {power_output:.7f} MWe")

        # 2) Compute next commanded steam flow
        m_dot_cmd_next = self.update_mass_flow_rate(
            m_dot_steam=m_dot_steam,
            power_supplied=power_output,
            power_dem=power_dem,
            dt=dt,
        )
        print(f"The new steam mass flow rate is {m_dot_cmd_next:.7f} kg/s\n")

        return power_output, m_dot_cmd_next
