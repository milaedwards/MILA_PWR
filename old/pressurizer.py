"""
Looking at the 'ic_system' module, assuming that a pressurizer instans has been
created, 'ic_system' calls the pressurizer 'step' method passing in 'plant_state'
values of T_hot_K, T_cold_K and dt. From the 'ic_system' module code ther returned value
is to be a tuple containing pressurizer pressure and pressurizer level. If level
is not returned then the 'ic_system' module substitutes in 0.0 as defined
in the plat_state module.

        _pzr_ret = self.pressurizer.step(ps.T_hot_K, ps.T_cold_K, dt)
        if isinstance(_pzr_ret, tuple):
            P_pzr = float(_pzr_ret[0])
            L_pzr = float(_pzr_ret[1]) if len(_pzr_ret) > 1 else ps.pzr_level_m
        else:
            P_pzr = float(_pzr_ret)
            L_pzr = ps.pzr_level_m

        ps = replace(
            ps,
            pzr_pressure_Pa=P_pzr,
            P_primary_Pa=P_pzr,
            pzr_level_m=L_pzr,
        )
Run time is managed by 'main' module and calls the 'ic_system' module which
in turn callst the pressurizer 'step' method.

The python module 'pyXSteam' can be installed into the virtual environment using
' pip install pyXSteam'.
"""

#from time import sleep
from pyXSteam.XSteam import XSteam # Won't be using the GUI, just the functions

##OBJECT INSTANCES:
xs = XSteam() # for calulating water density

##VARIABLES NEEDED:
##	GIVEN:
##Pressurizer volume 		    (pzr_volume = 10.0 m**3 static)
pzr_volume:float = 10.0

##Proportional Heater power 	    (HEATER_POWER_PROP = 20e3 Watts)
HEATER_POWER_PROP:float = 2e4

##Backup heater power 		    (HEATER_POWER_BACKUP = 10e3 Watts)
HEATER_POWER_BACKUP:float = 1e4

##Spray flow rate 		    (SPRAY_FOLW_RATE = 44.1 kg/sec)
SPRAY_FOLW_RATE:float = 44.1

##Spray temperature {Tcold} 	    (SPRAY_TEMP = 553.15 K)
SPRAY_TEMP:float = 553.15

##Relief valve setpoint 	    (RELIEF_VALVE_SETPOINT = 16.5e5 Pa)
RELIEF_VALVE_SETPOINT:float = 1.65e6

##Relief valve flow rate 	    (RELIEF_VALVE_FLOW_RATE = 20.0 kg/s)
RELIEF_VALVE_FLOW_RATE:float = 20.0

##Pressure deadband 		    (pressure_deadband = 1e5 Pa static)
pressure_deadband:float = 1e4

##Pressure nominal setpoint 	    (pressure_setpoint = 15.5e6 Pa static)
pressure_setpoint:float = 1.158e6

##	PASSED IN:
##New temperature 		    (new_temperature)
new_temperature:float
##Run time value in seconds 	    (dt)
dt:float = 0.1

##	CALCULATED:
##pressurizer pressure 		    (pzr_new_pressure)
pzr_new_pressure:float = 0.0
P_primary_Pa: float = 0.0

## Pressurizer previous pressure    (pzr_prev_pressure)
pzr_prev_pressure: float = 1.1e6

##**pressurizer final pressure 	    (pzr_final_pressure)
pzr_final_pressure:float = 0.0

##**pressurizer level 		    (pzr_final_level)
pzr_final_level:float = 0.0

##Water density 		    (pzr_water_density)
pzr_water_density:float = 1000.0

##Temperature previous 		    (pzr_prev_temperature)
pzr_prev_temperature:float = 595.15

##Temperature new                   (pzr_new_temperature)
pzr_new_temperature: float = 0.0

##Pressure deadband low value 	    (pressure_deadband_low = (pressure_setpoint - (pressure_deadband / 2) Pa)
pressure_deadband_low:float = pressure_setpoint - (pressure_deadband / 2)

##Pressure deadband high value 	    (pressure_deadband_high = pressure_setpoint + (pressure_deadband / 2) Pa)
pressure_deadband_high:float = pressure_setpoint + (pressure_deadband / 2)

# Pressurizer diameter = 2.54 m
# Height 12.776 m

##  ACTION VARIABLES:
inc_pressure: float = 1e4 # Pa
inc_temperature: float = 1e2 # K

# Create an instance of XSteam
xs = XSteam()

def calc_water_density(press_MPa: float) -> float:
    """
    This function calculates the density of the water in the pressurizer
    given the pressure in MPa and returns a float rounded to 3 decimal digits.
    """
    global water_density
    water_density = xs.rhoL_p(press_MPa)
    return water_density

def calc_pressurizer_level(volume:float, density:float, pressure:float) -> float:
    """
    This function calculates the pressurizer level given the pzr volume, water density,
    and pzr pressure. It returns a float in meters. Using variable 'pressurizer_volume' and
    'water_density' instead of having them passes in. 'water_desnity' can be calculated using
    steam tables, or my code 'steamTableLookup.py'.
    """
    global pzr_new_temperature, pzr_water_density, pzr_final_level
    # First calculate the pressure using the 'calc_pressure' function passing in the 'Tnew'
    pzr_new_temperature = calc_pressure(pzr_new_temperature)
    # Next calculate density
    water_density = calc_water_density(pressure / 1e6)
    # Calculate the mass of water in the pressurizer
    mass = pressure * volume / (density * 9.81)  # Using hydrostatic pressure formula
    # Calculate the level in meters
    level = mass / (density * volume)
    pzr_final_level = level
    return level

def calc_pressure (Tnew: float) -> float:
    """
    This function calculates the current pzr pressure given the current temperature (K)
    and returns a float in KPa.
    """
    global pzr_new_pressure, pzr_prev_pressure
    #pzr_new_pressure = pzr_prev_pressure * (pzr_prev_temperature / Tnew)
    pzr_new_pressure = xs.psat_t(Tnew) * 1e6
    return pzr_new_pressure

def evaluate_pressure(T_hot: float, T_cold: float, dt: float) -> None:
    """
    This function is used to determin any actions that need to be taken to maintain
    the pzr pressure withing the assigned 'deadband'. Noting is returned.
    """
    global pzr_new_pressure, pzr_prev_pressure, pzr_new_temperature, pzr_prev_temperature, pzr_water_density, P_primary_Pa
    global pressure_deadband_low, pressure_deadband_high, inc_pressure, inc_temperature, pzr_final_level
    global pzr_volume

    # Start from the hot-leg temperature as the effective pressurizer temperature
    pzr_new_temperature = T_hot
    #print(f"pzr_new_temperature: {pzr_new_temperature} K")
    pzr_new_pressure = calc_pressure(pzr_new_temperature)

    # New action calculations
    # Determine whether we're below, within, or above the deadband
    if pressure_deadband_low <= pzr_new_pressure <= pressure_deadband_high:
        # Within deadband: no action
        #print("In range")
        pass

    elif pzr_new_pressure < pressure_deadband_low:
        # Too low: use heaters to increase pressure.
        # Treat inc_pressure as a rate [Pa/s] and scale by dt_s.
        #print("Heater ON")
        pzr_new_pressure = pzr_new_pressure + inc_pressure * dt

    elif pzr_new_pressure > pressure_deadband_high and pzr_new_pressure < RELIEF_VALVE_SETPOINT:
        # Too high: use spray (cold leg) to cool the pressurizer gradually.
        #print("Spray ON")
        tau_spray = 5.0  # [s] time constant for spray cooling (tunable)
        dT_dt = (T_cold - pzr_new_temperature) / tau_spray
        pzr_new_temperature = pzr_new_temperature + dT_dt * dt
        pzr_new_pressure = calc_pressure(pzr_new_temperature)

    else:
        # Relief lifting, reset values or take protective action.
        #print("Relief valve lifting, RX Shutdown")
        pass

    pzr_final_level = calc_pressurizer_level(pzr_volume, pzr_water_density, pzr_new_pressure)
    P_primary_Pa = pzr_new_pressure
    pzr_prev_pressure = pzr_new_pressure
    #print(f"pzr_final_level: {round(pzr_final_level, 3)} meters")

### Simulation function as if called from the 'ic_system' module
# def step (pzr_new_temperature, dtime) -> None:
def step(T_hot, T_cold, dt) -> float:
    """
    This function simulates the 'ic_system' module call to the pressurizer 'step'
    method passing in 'T_hot', 'T_cold', and 'dt'. I don't think dt is supposed to change though.
    """
    global pzr_new_pressure, pzr_final_level, pzr_prev_pressure, P_primary_Pa
    pzr_prev_pressure = xs.psat_t(T_hot) / 1e6
    evaluate_pressure(T_hot, T_cold, dt)
    #print(f"Setpoint: {round(pressure_setpoint, 3)}, New Pressure: {round(pzr_new_pressure, 3)}, New Level: {round(pzr_final_level, 3)}")
    return P_primary_Pa
