import CoolProp.CoolProp as CoolProp
import numpy as np
from config import Config

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
        self.calibration_factor = None      # set on first call to step() to remove initial power mismatch

        # --- Secondary-pressure PI control for steam flow (to be called from IC_System) ---
        # Use Config for nominal steam-flow and secondary pressure setpoint
        self.cfg = Config()

        # Nominal main-steam mass flow (kg/s) and secondary pressure setpoint (Pa)
        self.m_dot_steam_nom_kg_s = getattr(self.cfg, "M_DOT_STEAM_NOM_KG_S", 0.0)
        self.P_sec_set_Pa = getattr(self.cfg, "P_SEC_CONST_PA", 0.0)

        # PI gains for pressure control (units: kg/s/Pa for Kp, kg/(s·Pa) for Ki)
        # These are starting-point values and may need tuning.
        self.Kp_p = 0.0
        self.Ki_p = 0.0

        # Integrator state for pressure controller
        self._p_err_int = 0.0

    def turbine_power(self, inlet_p, inlet_t, outlet_p, m_dot_steam):
        """Calculate actual specific turbine work using isentropic thermodynamics"""
        # Retrieve the enthalpy value into the turbine assuming saturated vapor at inlet pressure.
        # Using (P, Q=1.0) avoids CoolProp's saturation ambiguity with (P, T) on the boiling line.
        h_in = CoolProp.PropsSI('H', 'P', inlet_p, 'Q', 1.0, self.fluid)

        # Retrieve the corresponding entropy for saturated vapor at inlet pressure
        s_in = CoolProp.PropsSI('S', 'P', inlet_p, 'Q', 1.0, self.fluid)

        # use the entropy rate above to find an enthalpy rate out of the turbine using the set outlet pressure value
        h_out_isentropic = CoolProp.PropsSI('H', 'P', outlet_p, 'S', s_in, self.fluid)

        # estimated enthalpy value after accounting for isentropic turbine efficiency
        h_out_actual = h_in - (self.turbine_efficiency * (h_in - h_out_isentropic))

        # calculate specific work rate using standard equation of enthalpy into the turbine minus enthalpy out of the turbine
        specific_work = h_in - h_out_actual  # J/kg

        # Calculate power out using the passed parameter of steam mass flow rate times the specific work
        # rate times divided by 1 million (to convert to MegaWatts) and also included generator efficiency rate here,
        # as this is the scope of the project, but also means the output is an electric power
        power_output_MW = self.generator_efficiency * specific_work * m_dot_steam / 1000000

        return power_output_MW  # Power output by the generator to the grid in MegaWatt electric(MWe)

    def update_mass_flow_rate(self, m_dot_steam, power_supplied, power_dem, dt):

        eps = 1.0e-6
        denom = max(abs(power_dem), eps)  # [MW]

        # ----------- Power error --------------------------
        power_error = power_supplied - power_dem        # [MW]
        power_percent_change = power_error / denom      # [-]

        #print(
        #    f"[Turb] power_supplied={power_supplied:7.1f} MW, "
        #    f"power_dem={power_dem:7.1f} MW, "
        #    f"err={power_error:7.1f} MW, "
        #    f"pct_err={100 * power_percent_change:6.2f}%, "
        #    f"m_dot={m_dot_steam:7.1f}"
        #)

        # ----------- Flow limits per NRC rules ------------
        m_dot_step = m_dot_steam * 0.10 * dt  # [kg/s] 10%/s allowed change
        m_dot_ramp = m_dot_steam * 0.05 * dt / 60.0  # [kg/s] 5%/min ramp

        # ----------- Controller logic ---------------------
        if power_percent_change > 0.10:
            # Too much power → reduce flow aggressively
            m_dot_update = -(m_dot_step + m_dot_ramp)
        elif power_percent_change < -0.10:
            # Too little power → increase flow aggressively
            m_dot_update = (m_dot_step + m_dot_ramp)
        else:
            # Within +/-10% → proportional fine adjustment
            m_dot_update = -power_percent_change * m_dot_steam * dt  # [kg/s]

        # ----------- Apply update --------------------------
        m_dot_new = m_dot_steam + m_dot_update  # [kg/s]

        # ----------- Prevent non-physical flow -------------
        if m_dot_new < 0.0:
            m_dot_new = 0.0

        return m_dot_new

    def pressure_pi_mass_flow_cmd(self, P_secondary_Pa: float, dt: float) -> float:
        """
        Compute a commanded main-steam mass-flow rate using a PI controller
        on secondary / steam-dome pressure.

        Parameters
        ----------
        P_secondary_Pa : float
            Current secondary (steam-dome) pressure in Pa.
        dt : float
            Integration time step [s] for updating the PI integrator.

        Returns
        -------
        float
            Commanded main-steam mass flow [kg/s].
        """
        # Guard against missing config or zero dt
        if dt <= 0.0:
            return self.m_dot_steam_nom_kg_s

        # If we don't have reasonable nominal values, just return the input nominal
        if self.m_dot_steam_nom_kg_s <= 0.0 or self.P_sec_set_Pa <= 0.0:
            return self.m_dot_steam_nom_kg_s

        # Pressure error: positive if actual pressure is below setpoint
        # so that the controller opens the valve / increases flow.
        p_err = self.P_sec_set_Pa - P_secondary_Pa

        # Update integrator
        self._p_err_int += p_err * dt

        # Raw PI output around nominal flow
        m_dot_cmd = (
            self.m_dot_steam_nom_kg_s
            + self.Kp_p * p_err
            + self.Ki_p * self._p_err_int
        )

        # Simple anti-windup: if we clamp the output, also clamp integrator so it
        # doesn't run away.
        # Allow some margin around nominal, e.g. 20% up/down by default.
        m_dot_min = 0.5 * self.m_dot_steam_nom_kg_s
        m_dot_max = 1.5 * self.m_dot_steam_nom_kg_s

        m_dot_clamped = float(np.clip(m_dot_cmd, m_dot_min, m_dot_max))

        # If we hit a limit, back out the last integrator step (very simple anti-windup)
        if m_dot_clamped != m_dot_cmd:
            self._p_err_int -= p_err * dt

        return m_dot_clamped

    # ======================================================
    #                   MAIN STEP FUNCTION
    # ======================================================
    def step(self,
             inlet_t: float,  # [K]  saturated steam temperature from SG
             inlet_p: float,  # [Pa] saturated steam pressure from SG
             m_dot_steam: float,  # [kg/s] current turbine steam flow
             power_dem: float,  # [MW] demand from grid/operator
             dt: float  # [s] timestep
             ) -> tuple[float, float]:
        """
        Advance turbine one timestep.

        Returns:
            - power_output [MW]
            - m_dot_steam_new [kg/s]
        """

        # ------- Compute raw electrical output -------------
        raw_power = self.turbine_power(
            inlet_p=inlet_p,
            inlet_t=inlet_t,
            outlet_p=self.outlet_p,
            m_dot_steam=m_dot_steam,
        )

        # ------- One-time calibration to remove mismatch ---
        # On the first call, scale the turbine model so that at the
        # current operating point the supplied power matches demand.
        if self.calibration_factor is None:
            eps = 1.0e-6
            if abs(power_dem) > eps and abs(raw_power) > eps:
                self.calibration_factor = power_dem / raw_power
            else:
                self.calibration_factor = 1.0

        power_output = raw_power * self.calibration_factor

        # ------- Update steam mass flow --------------------
        m_dot_steam_new = self.update_mass_flow_rate(
            m_dot_steam=m_dot_steam,
            power_supplied=power_output,
            power_dem=power_dem,
            dt=dt,
        )

        return power_output, m_dot_steam_new





