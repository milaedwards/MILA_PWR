# system_constants.py
"""
This module contains some constants used in the simulation
of an operating nuclear poswer plant of the 'pressurized
water reactor' (PWR) design.
"""
# Written by TCS 20251107

class SystemConstants():
    # AP1000 pressurizer parameters (simplified, example values)
    # TCS  1 psia to pascal = 6894.75728 pascal
    
    def __init__(self):
        self.V_PRIMARY: float = 271.84 # [m^3]
        self.U = 7000 # w/m2*K
        self.P_SETPOINT = 15.5e6  # Pa (approx. 2250 psia)
        self.VOLUME = 59.47 # m^3 (pressurizer total volume)
        self.V_WATER_INITIAL = self.VOLUME / 2 #25.0  # m^3 (initial liquid volume)
        self.PRESSURE_TOLERANCE = 0.5e6  # Pa (control band)
        self.PRESSURIZER_LEVEL = 6.338 # m @ 50% of the total volume

        # Conversion 1 psi = 6.894757e3 Pa
        # Heater parameters
        self.HEATER_POWER_PROP = 1500e3  # W (proportional heater power)
        self.HEATER_POWER_BACKUP = 500e3  # W (backup heater power)
        #1.5 psia/kW = 1.5 * 6.894757e3 Pa

        # Spray parameters
        self.SPRAY_FLOW_RATE = 44.1  # kg/s (max spray flow)
        self.SPRAY_TEMP = 280 + 273.15  # K (spray water temperature)
        #1.25 psi/kg = 1.25 * 6.894757e3 Pa / kg
        # Max flowrate = 44.1 kg / sec

        # Relief valve parameters
        self.RELIEF_VALVE_SETPOINT = 16.5e6  # Pa
        self.RELIEF_VALVE_FLOW_RATE = 20.0  # kg/s

        # Steam/Water Properties (simplified, assume constant density)
        self.rho_steam = 50.0  # kg/m^3 (approximate)
        self.rho_water = 750.0  # kg/m^3 (approximate)
        self.cv_steam = 1500.0  # J/kg/K (approximate) TCS (100C = 1.86 kJ/kg*K)
        self.cp_water = 5000.0  # J/kg/K (approximate)

        # PID control parameters for pressure
        self.KP = 1.0e-8
        self.KI = 1.0e-10
        self.KD = 0.0

        self.T_avg = 574.039 #### TCS added as a module variable for passing around

        

