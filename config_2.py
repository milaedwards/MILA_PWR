from dataclasses import dataclass, field
import numpy as np

try:
    import CoolProp.CoolProp as clp
except ImportError:
    clp = None


@dataclass
class Config:
    # ---------------- Simulation ----------------
    dt: float = 0.1              # [s] integration time step
    t_final: float = 600.0       # [s] total simulation time
    log_every_n: int = 1         # [-] print/log interval

    SOLVER_METHOD_REACTOR: str = "Radau"  # [-] ODE solver method [8]
    SOLVER_RTOL_REACTOR: float = 1.0e-6   # [-] relative tolerance [8]
    SOLVER_ATOL_REACTOR: float = 1.0e-8   # [-] absolute tolerance [8]

    # ---------------- Plant-level RCS ----------------
    P_PRI_SET_Pa: float = 15_513_000.0  # [Pa] RCS setpoint pressure
    N_LOOPS: int = 2                 # [-] number of primary loops
    RHO_NOM: float = 726.51          # [kg/m^3] nominal RCS density
    CP_PRI: float = 5888.866           # [J/kg-K] RCS coolant heat capacity
    M_DOT_PRI: float = 14275.560    # [kg/s] total mass flow in core
    #M_DOT_VESSEL: float = 15170.145  # [kg/s] total RCS mass flow
    M_DOT_LOOP: float = 157500.000   # [gpm] best-estimate loop flow per loop
    #V_PRIMARY: float = 271.84        # [m^3] total RCS fluid volume
    Q_CORE_NOMINAL_W: float = 3.400e9  # [W] core thermal power
    Q_CORE_NOMINAL_MW: float = 3400.0  # [MW] core thermal power

    # ---------------- Secondary plant-level ----------------
    P_SEC_CONST_Pa: float = 5_764_017.09   # [Pa] SG secondary pressure
    T_SAT_SEC_K: float = 546.13           # [K] saturation temp at P_sec
    M_DOT_SEC_KG_S: float = 1887.4           # [kg/s] main steam / feed flow

    # Simple secondary pressure controller (turbine demand bias)
    Kp_PSEC: float = 0.5                # [-] proportional gain

    # ---------------- Initial conditions ----------------
    T_HOT_INIT_K: float = 610.0 * 5.0 / 9.0 + 273.15    # [K] initial hot-leg temp
    T_COLD_INIT_K: float = 537.2 * 5.0 / 9.0 + 273.15    # [K] initial cold-leg temp
    P_PRI_INIT_Pa: float = P_PRI_SET_Pa  # [Pa] initial primary pressure
    P_SEC_INIT_Pa: float = P_SEC_CONST_Pa  # [Pa] initial secondary pressure
    ROD_INSERT_INIT: float = 0.57  # [-] initial rod insertion fraction

    # Rated electrical turbine-generator power
    P_RATED_MWe: float = 1122.0  # [MW_e] nominal electrical output

    # RCS volumes
    V_RXI_m3: float = 47.0       # [m^3] lower plenum volume
    V_HOT_m3: float = 7.5        # [m^3] hot-leg piping
    V_COLD_m3: float = 11.5      # [m^3] cold-leg piping
    V_SGI_m3: float = 11.35      # [m^3] SG inlet plenum
    V_SGU_m3: float = 11.35      # [m^3] SG outlet plenum

    # SG secondary
    T_fw_K: float = 226.7 + 273.15        # [K] feedwater temperature
    mdot_fw_kg_s: float = M_DOT_SEC_KG_S  # [kg/s] feedwater mass flow

    # SG geometry
    N_tubes: int = 10025                  # [-] number of U-tubes
    L_tube_m: float = 20.63               # [m] tube length
    D_i_m: float = 0.0154                 # [m] inner diameter
    t_wall_m: float = 0.001016            # [m] tube wall thickness
    D_o_m: float = field(init=False)      # [m] outer diameter (computed)
    A_ht_m2: float = 10936.06             # [m^2] heat transfer area
    m_tube_kg: float = 1.28e5             # [kg] tube bundle metal mass
    m_sec_water_kg: float = 7.9722e4      # [kg] secondary inventory
    V_steam_m3: float = 147.9             # [m^3] steam dome volume

    # Heat transfer
    U_W_m2K: float = 7660.0               # [W/m^2-K] overall HTC
    h_s_W_m2K: float = 38794.0            # [W/m^2-K] secondary HTC

    # Tube metal properties
    k_690_W_mK: float = 13.5              # [W/m-K] Alloy 690 conductivity
    cp_690_J_kgK: float = 524.0           # [J/kg-K] metal cp

    # ----------- Computed fields (set in __post_init__) ----------
    rho_p_kg_m3: float = field(init=False)   # [kg/m^3] RCS density
    cp_p_J_kgK: float = field(init=False)    # [J/kg-K] RCS cp
    T_sat_sec_K: float = field(init=False)   # [K] saturated T at P_SG
    cp_s_J_kgK: float = field(init=False)    # [J/kg-K] secondary cp
    H_ws_minus_cpTfw_J_kg: float = field(init=False)  # [J/kg]

    V_p_m3: float = field(init=False)        # [m^3] primary tube volume
    m_p_kg: float = field(init=False)        # [kg] primary mass
    C_p_J_K: float = field(init=False)       # [J/K] primary heat capacity
    C_m_J_K: float = field(init=False)       # [J/K] tube metal heat capacity
    C_s_J_K: float = field(init=False)       # [J/K] secondary water capacity

    R_total_m2K_W: float = field(init=False) # [m^2-K/W] total thermal R
    R_wall_m2K_W: float = field(init=False)  # [m^2-K/W] wall thermal R
    R_s_m2K_W: float = field(init=False)     # [m^2-K/W] sec-side R
    R_p_m2K_W: float = field(init=False)     # [m^2-K/W] pri-side R
    h_p_W_m2K: float = field(init=False)     # [W/m^2-K] primary HTC

    G_pm_W_K: float = field(init=False)      # [W/K] pri→metal conductance
    G_ms_W_K: float = field(init=False)      # [W/K] metal→sec conductance

    tau_pm_s: float = field(init=False)      # [s] primary→metal tau
    tau_mp_s: float = field(init=False)      # [s] metal coupling tau
    tau_ms_s: float = field(init=False)      # [s] metal→secondary tau

    m_core_kg_s: float = field(init=False)   # [kg/s] core flow
    m_loop_kg_s: float = field(init=False)   # [kg/s] per-loop flow

    tau_hot_s: float = field(init=False)     # [s] hot-leg tau
    tau_sgi_s: float = field(init=False)     # [s] SG-inlet tau
    tau_sgu_s: float = field(init=False)     # [s] SG-outlet tau
    tau_cold_s: float = field(init=False)    # [s] cold-leg tau
    tau_rxi_s: float = field(init=False)     # [s] RPV inlet tau

    # =====================================================
    #                  POST-INIT CALCULATIONS
    # =====================================================
    def __post_init__(self):
        # Outer diameter
        object.__setattr__(self, "D_o_m", self.D_i_m + 2 * self.t_wall_m)

        # Primary tube volume [m^3]
        V_p = self.N_tubes * np.pi * (self.D_i_m/2)**2 * self.L_tube_m
        object.__setattr__(self, "V_p_m3", V_p)

        # Reference temperature for RCS property call
        T_ref = 0.5 * (self.T_HOT_INIT_K + self.T_COLD_INIT_K)

        # CoolProp properties
        rho_p = clp.PropsSI("D", "P", self.P_PRI_SET_Pa, "T", T_ref, "Water")  # [kg/m3]
        cp_p  = clp.PropsSI("C", "P", self.P_PRI_SET_Pa, "T", T_ref, "Water")  # [J/kg-K]
        object.__setattr__(self, "rho_p_kg_m3", rho_p)
        object.__setattr__(self, "cp_p_J_kgK", cp_p)

        # SG saturation properties
        T_sat = clp.PropsSI("T", "P", self.P_SEC_CONST_Pa, "Q", 0.0, "Water")  # [K]
        cp_s  = clp.PropsSI("C", "P", self.P_SEC_CONST_Pa, "Q", 0.0, "Water")  # [J/kg-K]
        object.__setattr__(self, "T_sat_sec_K", T_sat)
        object.__setattr__(self, "cp_s_J_kgK", cp_s)

        # Feedwater enthalpy correction term
        cp_fw  = clp.PropsSI("C", "P", self.P_SEC_CONST_Pa, "T", self.T_fw_K, "Water")  # [J/kg-K]
        h_steam = clp.PropsSI("H", "P", self.P_SEC_CONST_Pa, "Q", 1.0, "Water")         # [J/kg]
        H = h_steam - cp_fw * self.T_fw_K  # [J/kg]
        object.__setattr__(self, "H_ws_minus_cpTfw_J_kg", H)

        # Masses [kg] and heat capacities [J/K]
        m_p = rho_p * V_p
        C_p = m_p * cp_p
        C_m = self.m_tube_kg * self.cp_690_J_kgK
        C_s = self.m_sec_water_kg * cp_s
        object.__setattr__(self, "m_p_kg", m_p)
        object.__setattr__(self, "C_p_J_K", C_p)
        object.__setattr__(self, "C_m_J_K", C_m)
        object.__setattr__(self, "C_s_J_K", C_s)

        # Thermal resistances
        R_total = 1/self.U_W_m2K
        R_wall = self.t_wall_m / self.k_690_W_mK
        R_s = 1/self.h_s_W_m2K
        R_p = R_total - R_wall - R_s
        object.__setattr__(self, "R_total_m2K_W", R_total)
        object.__setattr__(self, "R_wall_m2K_W", R_wall)
        object.__setattr__(self, "R_s_m2K_W", R_s)
        object.__setattr__(self, "R_p_m2K_W", R_p)
        object.__setattr__(self, "h_p_W_m2K", 1/R_p)

        # Conductances [W/K]
        G_pm = (1/R_p) * self.A_ht_m2
        G_ms = self.h_s_W_m2K * self.A_ht_m2
        object.__setattr__(self, "G_pm_W_K", G_pm)
        object.__setattr__(self, "G_ms_W_K", G_ms)

        # SG taus [s]
        object.__setattr__(self, "tau_pm_s", self.C_p_J_K / self.G_pm_W_K)
        object.__setattr__(self, "tau_mp_s", self.C_m_J_K * (1/self.G_pm_W_K + 1/self.G_ms_W_K))
        object.__setattr__(self, "tau_ms_s", self.C_s_J_K / self.G_ms_W_K)

        # RCS flow
        m_core = self.M_DOT_PRI / 3600.0   # [kg/s]
        m_loop = m_core / 2                       # [kg/s per loop]
        object.__setattr__(self, "m_core_kg_s", m_core)
        object.__setattr__(self, "m_loop_kg_s", m_loop)

        # Hydraulic taus [s]
        rho = rho_p

        def tau_h(V, md): return rho * V / md

        object.__setattr__(self, "tau_hot_s",  tau_h(self.V_HOT_m3,  m_loop))
        object.__setattr__(self, "tau_sgi_s",  tau_h(self.V_SGI_m3,  m_loop))
        object.__setattr__(self, "tau_sgu_s",  tau_h(self.V_SGU_m3,  m_loop))
        object.__setattr__(self, "tau_cold_s", tau_h(self.V_COLD_m3, m_loop))
        object.__setattr__(self, "tau_rxi_s",  tau_h(self.V_RXI_m3,  m_core))
