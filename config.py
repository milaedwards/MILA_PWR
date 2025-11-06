from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    # Simulation
    dt: float = 0.1  # time step [s]
    t_final: float = 300.0  # total simulated time [s]
    log_every_n: int = 1  # log every n steps

    # Primary plant constants
    P_PRI_SET: float = 15_513_000.0  # primary operating pressure setpoint [Pa] (~2250 psia)
    N_LOOPS: int = 2  # number of primary loops
    RHO_NOM_KG_M3: float = 726.51  # nominal primary density [kg/m^3] (@15.5 MPa, 300°C)
    CP_PRI_J_PER_KG_K: float = 5438.0  # primary heat capacity [J/(kg*K)]
    M_DOT_PRI = 14520.72  # [kg/s]

    # Secondary constants
    P_SEC_CONST: float = 5764017.09  # Secondary pressure [Pa] (836 psia)
    T_SAT_SEC_K: float = 546.13  # Saturation temp at P_sec [K] (523.37 °F)
    M_DOT_SEC: float = 1887.4 # kg/s steam mass flow rate at full power (1.498e7 lbm/hr)

    Q_CORE_NOMINAL_W: float = 3.400e9  # Nominal core thermal power [W] (3400 MWt)

    # Starting conditions (t=0)
    T_HOT_INIT_K: float = 595.59  # Hot-leg initial temperature [K] (612.2 °F)
    T_COLD_INIT_K: float = 552.59  # Cold-leg initial temperature [K] (535.0 °F)
    P_PRI_INIT_PA: float = P_PRI_SET  # Initial primary pressure [Pa]
    P_SEC_INIT_PA: float = P_SEC_CONST  # Initial secondary pressure [Pa]
    ROD_INSERT_INIT: float = 0.55  # Initial rod insertion (0 = withdrawn)

    # Time Constants
    TAU_COOLANT_CORE = 1.55 # [s]
    TAU_FUEL = 5.54 # [s]
    TAU_MODERATOR = 4.0 # [s]

    THETA_PRIMARY = 12.3 # [s]
    TAU_HOTLEG = 2.46 # [s]
    TAU_COLDLEG = 2.46 # [s]
    TAU_PF_SG_TUBES = 4.305 # [s]
    TAU_VESSEL_PLENUMS = 3.075 # [s]

    TAU_PF_SG_S = 1.6 # [s]
    TAU_MT_SG_S = 0.54 # [s]
    TAU_SF_SG_S = 2.8 # [s]