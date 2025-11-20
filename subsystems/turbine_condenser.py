import CoolProp.CoolProp as CoolProp
from config import Config as cfg
#from ic_system import ICSystem as ICSystem
#from plant_state import PlantState as plant_state


class TurbineModel:
    """Advanced model using proper thermodynamics
    This module requires input temperature and pressure and outputs power and mass flow rate
    Per NRC documents, the turbine has a step function that allows an increase or decrease of
    steam flow rate up to a max of 10%





    """

    def __init__(self):
        self.fluid = 'Water'
        self.turbine_efficiency = 0.80
        self.generator_efficiency = 0.90
        self.outlet_p = 5000  # in Pascals
        self.inlet_t = cfg.T_SAT_SEC
        self.inlet_p = cfg.P_SEC_CONST
        self.m_dot_steam = cfg.M_DOT_SEC

    def turbine_power(self, inlet_p, inlet_t, outlet_p, m_dot_steam):
        """Calculate actual specific turbine work using isentropic thermodynamics"""
        # Use saturated vapor at inlet pressure for turbine inlet enthalpy to avoid
        # (P, T) saturation ambiguities and better reflect main steam conditions.
        h_in = CoolProp.PropsSI('H', 'P', inlet_p, 'Q', 1.0, self.fluid)

        # Retrieve entropy for saturated vapor at inlet pressure to avoid (P, T) saturation issues
        s_in = CoolProp.PropsSI('S', 'P', inlet_p, 'Q', 1.0, self.fluid)

        h_out_isentropic = CoolProp.PropsSI('H', 'P', outlet_p, 'S', s_in, self.fluid)

        # Actual work
        h_out_actual = h_in - (self.turbine_efficiency * (h_in - h_out_isentropic))

        specific_work = h_in - h_out_actual  # kJ/kg

        power_output = self.generator_efficiency * specific_work * m_dot_steam / 1000000

        return power_output  # Power = MJ/s = MWe

    def update_mass_flow_rate(self, m_dot_steam, power_supplied, power_dem, dt):

        m_dot_steam_max_step = m_dot_steam * 0.1 * (
                    dt / 1)  # max change is 10% step change, which is assumed to be a 1 second time interval

        m_dot_steam_max_ramp = m_dot_steam * 0.05 * (dt / 60)  # max ramp up/down rate is 5% per minute

        # Avoid division by zero when turbine power is near zero at startup.
        if abs(power_supplied) < 1e-6:
            # At startup (or if power is essentially zero), don't try to make a large
            # percentage-based correction in a single step; let flow ramp in later.
            power_percent_change = 0.0
        else:
            power_percent_change = (power_supplied - power_dem) / power_supplied

        if power_percent_change > 0.1:
            m_dot_steam_update = -(m_dot_steam_max_step + m_dot_steam_max_ramp)
        elif power_percent_change < -0.1:
            m_dot_steam_update = (m_dot_steam_max_step + m_dot_steam_max_ramp)
        else:
            # Proportional adjustment scaled by dt (dt/1 is equivalent to dt)
            m_dot_steam_update = -(power_percent_change * m_dot_steam * dt)

        new_m_dot_steam = m_dot_steam + m_dot_steam_update

        return new_m_dot_steam

    def step(self, inlet_t, inlet_p, m_dot_steam, power_supplied, power_dem, dt):

        power_output = self.turbine_power(
            inlet_p=inlet_p,
            inlet_t=inlet_t,
            outlet_p=self.outlet_p,
            m_dot_steam=m_dot_steam,
        )
        m_dot_steam_new = self.update_mass_flow_rate(
            m_dot_steam=m_dot_steam,
            power_supplied=power_supplied,
            power_dem=power_dem,
            dt=dt,
        )

        return power_output, m_dot_steam_new