import CoolProp.CoolProp as CoolProp
from config import Config as cfg


# from ic_system import ICSystem as ICSystem
# from plant_state import PlantState as plant_state


class TurbineModel:
    """Advanced model using proper thermodynamics
    This module requires input temperature and pressure and outputs power and mass flow rate
    Per NRC documents, the turbine has a step function that allows an increase or decrease of
    steam flow rate up to a max of 10% step rate or a ramp up/down rate of 5%/min
    """

    def __init__(self):
        self.fluid = 'Water'
        self.turbine_efficiency = 0.80      # reasonable efficiency rates, but on the lower side to allow the
        self.generator_efficiency = 0.90    # accounting for other heat losses elsewhere in the system, like conduction
                                            # losses in the pipes

        self.outlet_p = 5000                # in Pascals

    def turbine_power(self, inlet_p, inlet_t, outlet_p, m_dot_steam):
        """Calculate actual specific turbine work using isentropic thermodynamics"""
        # Retrieve the enthalpy value into the turbine
        h_in = CoolProp.PropsSI('H', 'P', inlet_p, 'T', inlet_t, self.fluid)

        # Retrieve the entropy value at the above enthalpy value to get to Isentropic expansion
        s_in = CoolProp.PropsSI('S', 'P', inlet_p, 'T', inlet_t, self.fluid)

        # use the entropy rate above to find an enthalpy rate out of the turbine using the set outlet pressure value
        h_out_isentropic = CoolProp.PropsSI('H', 'P', outlet_p, 'S', s_in, self.fluid)

        # estimated enthalpy value after accounting for isentropic turbine efficiency
        h_out_actual = h_in - (self.turbine_efficiency * (h_in - h_out_isentropic))

        # calculate specific work rate using standard equation of enthalpy into the turbine minus enthalpy out of the turbine
        specific_work = h_in - h_out_actual  # J/kg

        # Calculate power out using the passed parameter of steam mass flow rate times the specific work
        # rate times divided by 1 million (to convert to MegaWatts) and also included generator efficiency rate here,
        # as this is the scope of the project, but also means the output is an electric power
        power_output = self.generator_efficiency * specific_work * m_dot_steam / 1000000

        return power_output  # Power output by the generator to the grid in MegaWatt electric(MWe)

    def update_mass_flow_rate(self, m_dot_steam, power_supplied, power_dem, dt):

        # max change is 10% step change, which is assumed to be a single second time interval
        m_dot_steam_max_step = m_dot_steam * 0.1 * (dt / 1)

        # max ramp up/down rate is 5% per minute
        m_dot_steam_max_ramp = m_dot_steam * 0.05 * (dt / 60)

        # use standard percentage calculation to find the difference between power output above and power
        # demanded from the grid
        power_percent_change = (power_supplied - power_dem) / power_supplied

        if power_percent_change > 0.1:
            m_dot_steam_update = -(m_dot_steam_max_step + m_dot_steam_max_ramp)
        elif power_percent_change < -0.1:
            m_dot_steam_update = (m_dot_steam_max_step + m_dot_steam_max_ramp)
        else:
            m_dot_steam_update = -(power_percent_change * m_dot_steam * (dt / 1))

        # add the change in steam mass flow rate to current steam mass flow rate to
        new_m_dot_steam = m_dot_steam + m_dot_steam_update

        return new_m_dot_steam

    def step(self, inlet_t, inlet_p, m_dot_steam, power_dem, dt):

        # Passing variables to the power calculator
        power_output = self.turbine_power(
            inlet_p,
            inlet_t,
            self.outlet_p,
            m_dot_steam,
        )

        # Passing variables to the steam mass flow rate update function
        m_dot_steam_new = self.update_mass_flow_rate(
            m_dot_steam,
            power_output,
            power_dem,
            dt
        )
        return power_output, m_dot_steam_new



