from dataclasses import dataclass, field
import math
import numpy as np

try:
    import CoolProp.CoolProp as clp
except ImportError:
    clp = None


@dataclass
class Config:
    """
    Unified AP1000 configuration:
    - RCS + core + SG
    - Kinetics + feedback
    - Rod worth + control
    - Turbine + scenario

    Units and sources are in the comments.
    """

    # =========================================================
    # 1. SIMULATION / SOLVER CONTROL
    # =========================================================
    PLANT_DT_S: float = 0.1          # [s] integration time step [8]
    PLANT_T_FINAL_S: float = 300.0   # [s] total simulation time [8]
    PLANT_LOG_EVERY_N: int = 1       # [-] log interval [8]

    SOLVER_METHOD_REACTOR: str = "Radau"  # [-] ODE solver method [8]
    SOLVER_RTOL_REACTOR: float = 1.0e-6   # [-] relative tolerance [8]
    SOLVER_ATOL_REACTOR: float = 1.0e-8   # [-] absolute tolerance [8]

    # =========================================================
    # 2. PRIMARY SYSTEM (RCS) – DCD CH. 5.1
    # =========================================================
    # Table 5.1-2: RCS design parameters
    RCS_P_SET_PA: float = 15_513_000.0
    # [Pa] RCS pressure (2250 psia) [1], DCD Table 5.1-2

    RCS_T_HOT_K: float = 610.0 * 5.0 / 9.0 + 273.15
    # [K] RV outlet temperature (hot leg), 610 °F [1]

    RCS_T_COLD_K: float = 537.2 * 5.0 / 9.0 + 273.15
    # [K] RV inlet temperature (cold leg), 537.2 °F [1]

    # Table 5.1-3: flows
    RCS_M_DOT_VESSEL_KG_S: float = 15_170.10
    # [kg/s] vessel flow, 120.4×10^6 lb/hr [2]

    RCS_M_DOT_CORE_KG_S: float = 14_275.60
    # [kg/s] core flow, 113.3×10^6 lb/hr [2]

    RCS_M_DOT_LOOP_GPM: float = 157_500.0
    # [gpm] best-estimate loop flow per loop [2]

    RCS_N_LOOPS: int = 2
    # [-] number of primary loops [1]

    RCS_RHO_NOM_KG_M3: float = 726.51
    # [kg/m^3] nominal RCS density at HFP (DCD + CoolProp) [3],[7]

    RCS_CP_NOM_J_KG_K: float = 5442.0
    # [J/kg-K] nominal RCS cp at HFP [3],[7]

    # Simple RCS segment volumes (geometry reconstruction / assumptions)
    RCS_V_RPV_INLET_M3: float = 47.0    # [m^3] downcomer / lower plenum [8]
    RCS_V_HOTLEG_M3: float = 7.5        # [m^3] hot leg per loop [8]
    RCS_V_COLDLEG_M3: float = 11.5      # [m^3] cold leg per loop [8]
    SG_V_PLENUM_IN_M3: float = 11.35    # [m^3] SG inlet plenum per SG [8]
    SG_V_PLENUM_OUT_M3: float = 11.35   # [m^3] SG outlet plenum per SG [8]

    # =========================================================
    # 3. CORE THERMAL DESIGN – DCD CH. 4.4 + MANN MODEL
    # =========================================================
    P_RATED_MWT: float = 3400.0
    # [MWt] rated core thermal power [3], DCD Table 4.4-1

    Q_CORE_NOMINAL_W: float = 3.40e9
    # [W] same as P_RATED_MWT [3]

    CORE_M_DOT_EFFECTIVE_KG_S: float = 13_456.6
    # [kg/s] effective flow for heat transfer, 106.8×10^6 lb/hr [3]

    CORE_M_FUEL_KG: float = 95_974.7
    # [kg] UO2 fuel mass, 211,727 lbm [3]

    CORE_CP_FUEL_J_KG_K: float = 313.0
    # [J/kg-K] fuel cp (ORNL UO2 data) [6]

    CORE_M_COOLANT_KG: float = 11_923.53
    # [kg] coolant mass in core (from project geometry) [8]

    CORE_CP_COOLANT_J_KG_K: float = 5442.0
    # [J/kg-K] coolant cp (same as RCS cp) [3],[7]

    CORE_U_FUEL_COOLANT_W_M2K: float = 1143.0
    # [W/m^2-K] fuel-to-coolant HTC (Kerlin + Masoud) [7],[8]

    CORE_A_FUEL_COOLANT_M2: float = 5267.6
    # [m^2] active core heat-transfer area [3], DCD Table 4.4-1

    CORE_F_FUEL_POWER_FRAC: float = 0.974
    # [-] fraction of power generated in fuel [3]

    CORE_F_BYPASS_FRAC: float = 0.059
    # [-] upper-plenum bypass flow fraction (5.9%) [2]

    # Vajpayee plenum time constants (used by thermal model)
    CORE_TAU_RXI_VAJ_S: float = 2.145
    # [s] T_cold → T_core_inlet lag (downcomer + lower plenum) [8]

    CORE_TAU_RXU_VAJ_S: float = 2.517
    # [s] Tc2 → T_hot_leg lag (upper plenum) [8]

    # =========================================================
    # 4. SECONDARY / STEAM SIDE – DCD CH. 5.1
    # =========================================================
    P_SG_SEC_PA: float = 5_764_000.0
    # [Pa] SG exit pressure, 836 psia [1], DCD Table 5.1-2

    M_DOT_STEAM_TOTAL_KG_S: float = 1886.21
    # [kg/s] total steam flow, 14.97×10^6 lb/hr [1]

    M_DOT_FEEDWATER_TOTAL_KG_S: float = 1886.21
    # [kg/s] total feedwater flow (same as steam at steady state) [1]

    T_FEEDWATER_K: float = 226.67 + 273.15
    # [K] feedwater temperature, 440 °F [1]

    # =========================================================
    # 5. STEAM GENERATOR GEOMETRY & INVENTORIES – DCD CH. 5.4
    # =========================================================
    SG_N_TUBES: int = 10025
    # [-] number of U-tubes per SG [4], DCD Table 5.4-4

    SG_TUBE_LEN_M: float = 20.64
    # [m] tube length, 67.7 ft [4]

    SG_TUBE_DI_M: float = 0.01544
    # [m] inner diameter, 0.608 in [4]

    SG_TUBE_T_WALL_M: float = 0.001016
    # [m] wall thickness, 0.040 in [4]

    SG_TUBE_DO_M: float = field(init=False)
    # [m] outer diameter (computed) [4]

    SG_A_HT_TOTAL_M2: float = 11_477.0
    # [m^2] total heat-transfer surface area, 123,538 ft^2 [4]

    SG_M_SEC_WATER_KG: float = 79_722.0
    # [kg] secondary water mass per SG, 175,758 lbm [5]

    SG_V_DOME_M3: float = 147.90
    # [m^3] steam dome volume per SG, 5,222 ft^3 [5]

    SG_U_TOTAL_W_M2K: float = 7660.0
    # [W/m^2-K] overall HTC [8]

    SG_H_SEC_W_M2K: float = 38_794.0
    # [W/m^2-K] secondary-side HTC [8]

    SG_K_TUBEMETAL_W_MK: float = 13.5
    # [W/m-K] Alloy 690 thermal conductivity [8]

    SG_CP_TUBEMETAL_J_KG_K: float = 524.0
    # [J/kg-K] Alloy 690 cp [8]

    # =========================================================
    # 6. ELECTRICAL / TURBINE – PLANT + PROJECT
    # =========================================================
    GEN_P_RATED_MWE_NAMEPLATE: float = 1200.0
    # [MWe] generator active (nameplate) power [6], Plant Description Sec. 8

    GEN_P_RATED_MWE_NET: float = 1117.0
    # [MWe] nominal net electrical output at ≈33% efficiency (project) [8]

    # For backward compatibility, use net rating for reactor control scenarios
    P_RATED_MWe: float = GEN_P_RATED_MWE_NET
    # [MWe] used in global kinetics/control code [8]

    # Turbine thermodynamic model parameters
    TURB_FLUID_NAME: str = "Water"      # [-] CoolProp fluid [8]
    TURB_EFF_ISEN_FRAC: float = 0.80    # [-] turbine isentropic efficiency [8]
    GEN_EFF_FRAC: float = 0.90          # [-] generator efficiency [8]

    TURB_P_OUTLET_PA: float = 5_000.0
    # [Pa] exhaust/condenser pressure (simplified) [8]

    TURB_M_DOT_STEP_MAX_FRAC: float = 0.10
    # [-] 10% step limit for steam flow [8]

    TURB_M_DOT_RAMP_MAX_FRAC_PER_MIN: float = 0.05
    # [-] 5%/min ramp limit for steam flow [8]

    TURB_POWER_ERR_LIMIT_FRAC: float = 0.10
    # [-] ±10% power error threshold used in mass-flow update [8]

    PLANT_W_TO_MW: float = 1.0e-6
    # [MW/W] conversion factor [8]

    # =========================================================
    # 7. KINETICS & REACTIVITY – DCD CH. 4.3 + KERLIN
    # =========================================================
    PCM_TO_DK_FRAC: float = 1.0e-5
    # [-] conversion factor pcm → Δk/k [3]

    KIN_LAMBDA_PROMPT_S: float = 19.8e-6
    # [s] prompt neutron generation time [3], DCD Table 4.3-2

    KIN_BETA_EFF: float = 0.0075
    # [-] effective delayed neutron fraction [3]

    KIN_BETA_SHAPE_U235: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.000221, 0.001467, 0.001313,
             0.002647, 0.000771, 0.000281],
            dtype=float
        )
    )
    # [-] β_i shape for U-235 (Kerlin) [7]

    KIN_LAMBDA_I_INV_S: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.0124, 0.0305, 0.111,
             0.301, 1.14, 3.01],
            dtype=float
        )
    )
    # [1/s] decay constants λ_i for 6 delayed groups [7]

    # Reactivity step used by scenarios (e.g., fixed ρ insertion)
    REACTIVITY_STEP_PCM_INIT: float = 0.0
    # [pcm] default fixed reactivity step [8]

    REACTIVITY_DEBUG_DEFAULT: bool = False
    # [-] flag for verbose reactivity debug [8]

    # =========================================================
    # 8. TEMPERATURE FEEDBACK – DCD CH. 4.3 + PROJECT
    # =========================================================
    FB_ALPHA_D_PCM_PER_F: float = -1.4
    # [pcm/°F] Doppler coefficient [3]

    FB_ALPHA_M1_PCM_PER_F: float = -14.0
    # [pcm/°F] moderator coefficient (inlet/avg) [3],[7]

    FB_ALPHA_M2_PCM_PER_F: float = -6.0
    # [pcm/°F] moderator coefficient (outlet) [3],[7]

    FB_COEFF_UNITS: str = "F"
    # [-] coefficients expressed per °F [3]

    FB_T_F0_K: float = 1128.0
    # [K] reference fuel temperature at HFP [8]

    FB_T_M0_K: float = 576.54
    # [K] reference average moderator temperature [8]

    FB_TC1_0_K: float = 577.91
    # [K] reference first coolant lump temperature [8]

    FB_TC2_0_K: float = 601.12
    # [K] reference second coolant lump temperature [8]

    # =========================================================
    # 9. ROD WORTH & MECHANICS – DCD CH. 4.3
    # =========================================================
    ROD_STROKE_IN: float = 166.755
    # [in] full rod travel length [5]

    ROD_V_MAX_IN_PER_MIN: float = 45.0
    # [in/min] maximum rod speed [5]

    ROD_X_INIT_FRAC: float = 0.57
    # [-] initial rod insertion fraction at BOC/HFP [8]

    ROD_X_MIN_FRAC: float = 0.0
    # [-] fully withdrawn [8]

    ROD_X_MAX_FRAC: float = 1.0
    # [-] fully inserted [8]

    ROD_RHO_TOTAL_PCM: float = 10_490.0
    # [pcm] total rod worth (all rods except MOST) [3], Table 4.3-3 item 3b

    ROD_LUT_X_POINTS: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.00, 0.05, 0.10, 0.15, 0.20,
             0.30, 0.40, 0.50, 0.60, 0.70,
             0.80, 0.90, 0.95, 1.00],
            dtype=float
        )
    )
    # [-] normalized insertion fraction x for rod worth LUT [8]

    ROD_LUT_WORTH_F_POINTS: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.0000, 0.0001, 0.0003, 0.0020, 0.0055,
             0.0200, 0.0600, 0.1200, 0.2400, 0.4600,
             0.7000, 0.9000, 0.9700, 1.0000],
            dtype=float
        )
    )
    # [-] normalized worth f(x) for rod worth LUT [8]

    # =========================================================
    # 10. CONTROL SYSTEM CONSTANTS – PROJECT
    # =========================================================
    CTRL_KP_TEMP: float = 0.003     # [-] temp-loop proportional gain [8]
    CTRL_KI_TEMP: float = 0.0       # [1/s] temp-loop integral gain [8]

    CTRL_KI_POWER: float = 0.001    # [1/s] outer-loop power integrator [8]
    CTRL_KFF_POWER: float = 0.0     # [-] load-change feedforward gain [8]

    CTRL_K_AW: float = 2.0          # [-] anti-windup back-calculation gain [8]
    CTRL_LEAK_INNER_INV_S: float = 0.10   # [1/s] inner-loop integrator leak [8]
    CTRL_LEAK_OUTER_INV_S: float = 0.10   # [1/s] outer-loop integrator leak [8]

    CTRL_DB_TEMP_K: float = 0.3     # [K] temperature deadband [8]
    CTRL_DB_POWER_PU: float = 0.04  # [pu] power deadband (4%) [8]

    CTRL_U_MAX_INV_S: float = 0.04  # [1/s] max controller output (rod speed) [8]
    CTRL_Z_BIAS_MAX_K: float = 0.5  # [K] max temperature bias from outer loop [8]

    # =========================================================
    # 11. SCENARIO / SLIDING TAVG PROGRAM – PROJECT
    # =========================================================
    SCEN_P_TURB_INIT_PU: float = 1.0
    # [pu] initial turbine power demand [8]

    SCEN_LOAD_STEP_MWE: float = -50.0
    # [MWe] turbine load change (negative = load reduction) [8]

    SCEN_STEP_TIME_S: float = 10.0
    # [s] time of load change or rod insertion [8]

    SCEN_P_TURB_STEP_PU: float = field(init=False)
    # [pu] P_turb after load step [8]

    FB_S_TAVG_K_PER_PU: float = 19.0
    # [K/pu] sliding T_avg program slope [8]

    # =========================================================
    # 12. DERIVED / RUNTIME VALUES
    # =========================================================
    RCS_M_DOT_LOOP_KG_S: float = field(init=False)   # [kg/s] per-loop mass flow
    rho_p_kg_m3: float = field(init=False)           # [kg/m^3] RCS density at T_ref
    cp_p_J_kgK: float = field(init=False)            # [J/kg-K] RCS cp at T_ref

    T_sat_sec_K: float = field(init=False)           # [K] sat T at P_SG_SEC_PA
    cp_s_J_kgK: float = field(init=False)            # [J/kg-K] cp of sat liquid
    H_ws_minus_cpTfw_J_kg: float = field(init=False) # [J/kg] h_steam - cp_fw*T_fw

    SG_V_TUBE_PRI_M3: float = field(init=False)      # [m^3] primary volume in tubes
    SG_M_PRI_TUBES_KG: float = field(init=False)     # [kg] primary mass in tubes

    SG_C_PRI_J_K: float = field(init=False)          # [J/K] primary-side HT capacity
    SG_C_METAL_J_K: float = field(init=False)        # [J/K] metal HT capacity
    SG_C_SEC_J_K: float = field(init=False)          # [J/K] secondary HT capacity

    SG_G_PM_W_K: float = field(init=False)           # [W/K] primary→metal conductance
    SG_G_MS_W_K: float = field(init=False)           # [W/K] metal→secondary conductance

    SG_TAU_PM_S: float = field(init=False)           # [s] primary lump tau
    SG_TAU_MP_S: float = field(init=False)           # [s] metal lump tau
    SG_TAU_MS_S: float = field(init=False)           # [s] secondary lump tau

    CORE_UA_FUEL_COOLANT_W_K: float = field(init=False)    # [W/K]
    CORE_TAU_FUEL_S: float = field(init=False)             # [s]
    CORE_TAU_COOLANT_S: float = field(init=False)          # [s]
    CORE_TAU_RESIDENCE_S: float = field(init=False)        # [s]
    CORE_H_FUEL_K_PER_S_PER_PU: float = field(init=False)  # [K/s per pu]
    CORE_H_COOLANT_K_PER_S_PER_PU: float = field(init=False)  # [K/s per pu]

    RCS_TAU_HOTLEG_S: float = field(init=False)      # [s] hot leg
    RCS_TAU_SGI_S: float = field(init=False)         # [s] SG inlet plenum
    RCS_TAU_SGU_S: float = field(init=False)         # [s] SG outlet plenum
    RCS_TAU_COLDLEG_S: float = field(init=False)     # [s] cold leg
    RCS_TAU_RPV_INLET_S: float = field(init=False)   # [s] RPV inlet (downcomer)

    ROD_X_RATE_MAX_INV_S: float = field(init=False)  # [1/s] max rod speed

    KIN_BETA_I: np.ndarray = field(init=False)       # [-] scaled β_i to β_eff
    KIN_BETA_TOTAL: float = field(init=False)        # [-] sum of β_i

    # =========================================================
    # POST-INIT CALCULATIONS
    # =========================================================
    def __post_init__(self) -> None:
        if clp is None:
            raise RuntimeError("CoolProp is required to initialize Config thermo properties.")

        # ---- SG tube outer diameter and primary volume ----
        D_o = self.SG_TUBE_DI_M + 2.0 * self.SG_TUBE_T_WALL_M
        object.__setattr__(self, "SG_TUBE_DO_M", D_o)

        V_p = self.SG_N_TUBES * math.pi * (self.SG_TUBE_DI_M / 2.0) ** 2 * self.SG_TUBE_LEN_M
        object.__setattr__(self, "SG_V_TUBE_PRI_M3", V_p)

        # ---- Convert loop flow from gpm to kg/s using nominal density ----
        loop_vol_m3_s = (self.RCS_M_DOT_LOOP_GPM * 0.00378541) / 60.0
        loop_mdot_kg_s = loop_vol_m3_s * self.RCS_RHO_NOM_KG_M3
        object.__setattr__(self, "RCS_M_DOT_LOOP_KG_S", loop_mdot_kg_s)

        # ---- Reference temperature and primary thermo ----
        T_ref = 0.5 * (self.RCS_T_HOT_K + self.RCS_T_COLD_K)
        rho_p = clp.PropsSI("D", "P", self.RCS_P_SET_PA, "T", T_ref, "Water")
        cp_p = clp.PropsSI("C", "P", self.RCS_P_SET_PA, "T", T_ref, "Water")
        object.__setattr__(self, "rho_p_kg_m3", rho_p)
        object.__setattr__(self, "cp_p_J_kgK", cp_p)

        # ---- SG saturation properties ----
        T_sat = clp.PropsSI("T", "P", self.P_SG_SEC_PA, "Q", 0.0, "Water")
        cp_s = clp.PropsSI("C", "P", self.P_SG_SEC_PA, "Q", 0.0, "Water")
        object.__setattr__(self, "T_sat_sec_K", T_sat)
        object.__setattr__(self, "cp_s_J_kgK", cp_s)

        cp_fw = clp.PropsSI("C", "P", self.P_SG_SEC_PA, "T", self.T_FEEDWATER_K, "Water")
        h_steam = clp.PropsSI("H", "P", self.P_SG_SEC_PA, "Q", 1.0, "Water")
        H_corr = h_steam - cp_fw * self.T_FEEDWATER_K
        object.__setattr__(self, "H_ws_minus_cpTfw_J_kg", H_corr)

        # ---- Thermal capacities ----
        m_p = rho_p * V_p
        C_p = m_p * cp_p
        C_m = self.SG_CP_TUBEMETAL_J_KG_K * self.SG_A_HT_TOTAL_M2 * self.SG_TUBE_T_WALL_M / self.SG_K_TUBEMETAL_W_MK
        C_s = self.SG_M_SEC_WATER_KG * cp_s

        object.__setattr__(self, "SG_M_PRI_TUBES_KG", m_p)
        object.__setattr__(self, "SG_C_PRI_J_K", C_p)
        object.__setattr__(self, "SG_C_METAL_J_K", C_m)
        object.__setattr__(self, "SG_C_SEC_J_K", C_s)

        # ---- SG thermal resistances and conductances ----
        R_wall = self.SG_TUBE_T_WALL_M / self.SG_K_TUBEMETAL_W_MK
        R_s = 1.0 / self.SG_H_SEC_W_M2K
        R_total = 1.0 / self.SG_U_TOTAL_W_M2K
        R_p = R_total - R_wall - R_s

        G_pm = self.SG_A_HT_TOTAL_M2 / R_p
        G_ms = self.SG_H_SEC_W_M2K * self.SG_A_HT_TOTAL_M2

        object.__setattr__(self, "SG_G_PM_W_K", G_pm)
        object.__setattr__(self, "SG_G_MS_W_K", G_ms)

        tau_pm = C_p / G_pm
        tau_mp = C_m * (1.0 / G_pm + 1.0 / G_ms)
        tau_ms = C_s / G_ms

        object.__setattr__(self, "SG_TAU_PM_S", tau_pm)
        object.__setattr__(self, "SG_TAU_MP_S", tau_mp)
        object.__setattr__(self, "SG_TAU_MS_S", tau_ms)

        # ---- Core UA, taus, and H-factors ----
        UA = self.CORE_U_FUEL_COOLANT_W_M2K * self.CORE_A_FUEL_COOLANT_M2
        object.__setattr__(self, "CORE_UA_FUEL_COOLANT_W_K", UA)

        tau_f = self.CORE_M_FUEL_KG * self.CORE_CP_FUEL_J_KG_K / UA
        tau_c = self.CORE_M_COOLANT_KG * self.CORE_CP_COOLANT_J_KG_K / UA
        tau_r = self.CORE_M_COOLANT_KG / self.CORE_M_DOT_EFFECTIVE_KG_S

        object.__setattr__(self, "CORE_TAU_FUEL_S", tau_f)
        object.__setattr__(self, "CORE_TAU_COOLANT_S", tau_c)
        object.__setattr__(self, "CORE_TAU_RESIDENCE_S", tau_r)

        H_f = self.CORE_F_FUEL_POWER_FRAC * self.Q_CORE_NOMINAL_W / (
            self.CORE_M_FUEL_KG * self.CORE_CP_FUEL_J_KG_K
        )
        H_c = (1.0 - self.CORE_F_FUEL_POWER_FRAC) * self.Q_CORE_NOMINAL_W / (
            self.CORE_M_COOLANT_KG * self.CORE_CP_COOLANT_J_KG_K
        )

        object.__setattr__(self, "CORE_H_FUEL_K_PER_S_PER_PU", H_f)
        object.__setattr__(self, "CORE_H_COOLANT_K_PER_S_PER_PU", H_c)

        # ---- Hydraulic taus (RCS) using best-estimate flows ----
        def tau(vol_m3: float, m_dot_kg_s: float) -> float:
            return rho_p * vol_m3 / m_dot_kg_s

        tau_hot = tau(self.RCS_V_HOTLEG_M3, loop_mdot_kg_s)
        tau_sgi = tau(self.SG_V_PLENUM_IN_M3, loop_mdot_kg_s)
        tau_sgu = tau(self.SG_V_PLENUM_OUT_M3, loop_mdot_kg_s)
        tau_cold = tau(self.RCS_V_COLDLEG_M3, loop_mdot_kg_s)
        tau_rxi = tau(self.RCS_V_RPV_INLET_M3, self.RCS_M_DOT_CORE_KG_S)

        object.__setattr__(self, "RCS_TAU_HOTLEG_S", tau_hot)
        object.__setattr__(self, "RCS_TAU_SGI_S", tau_sgi)
        object.__setattr__(self, "RCS_TAU_SGU_S", tau_sgu)
        object.__setattr__(self, "RCS_TAU_COLDLEG_S", tau_cold)
        object.__setattr__(self, "RCS_TAU_RPV_INLET_S", tau_rxi)

        # ---- Rod maximum speed (fraction of stroke per second) ----
        x_rate_max = (self.ROD_V_MAX_IN_PER_MIN / 60.0) / self.ROD_STROKE_IN
        object.__setattr__(self, "ROD_X_RATE_MAX_INV_S", x_rate_max)

        # ---- Scenario turbine power after load step ----
        P_turb_step = 1.0 + self.SCEN_LOAD_STEP_MWE / self.P_RATED_MWe
        object.__setattr__(self, "SCEN_P_TURB_STEP_PU", P_turb_step)

        # ---- Kinetics β_i scaled to β_eff ----
        beta_shape_sum = float(self.KIN_BETA_SHAPE_U235.sum())
        beta_i = self.KIN_BETA_SHAPE_U235 * (self.KIN_BETA_EFF / beta_shape_sum)
        object.__setattr__(self, "KIN_BETA_I", beta_i)
        object.__setattr__(self, "KIN_BETA_TOTAL", float(beta_i.sum()))


# =========================================================
# REFERENCES (FOR [n] TAGS ABOVE)
# =========================================================
# [1], AP1000 DCD, Rev. 19, Chapter 5, Table 5.1-2
#      "Reactor Coolant System Design Parameters" – RCS pressure,
#      hot/cold leg temperatures, steam and feedwater conditions.
#
# [2], AP1000 DCD, Rev. 19, Chapter 5, Table 5.1-3
#      "Reactor Vessel and Internals: Summary of Hydraulic Parameters" –
#      vessel flow, core flow, loop gpm, core bypass fraction.
#
# [3], AP1000 DCD, Rev. 19, Chapter 4, Table 4.4-1 and Table 4.3-2
#      Thermal design parameters (3400 MWt, effective flow, fuel mass)
#      and core kinetics parameters (β_eff, prompt lifetime).
#
# [4], AP1000 DCD, Rev. 19, Chapter 5, Table 5.4-4
#      "Steam Generator Design Data" – tube count and dimensions,
#      tube length, heat-transfer surface area.
#
# [5], AP1000 DCD, Rev. 19, Chapter 5, Table 5.4-5
#      "Steam Generator Water Inventories" – primary water volume,
#      secondary water mass, steam volume.
#
# [6], AP1000 Plant Description, Section 8
#      Steam and Power Conversion System – generator active (nameplate)
#      power of 1200 MWe.
#
# [7], CoolProp / IAPWS-IF97 water-steam properties
#      for density, enthalpy, and specific heat at AP1000 conditions.
#
# [8], Project-specific modeling assumptions and constants
#      not directly tabulated in the DCD (controller tuning, sliding
#      T_avg slope, time step, initial rod fraction, etc.).
