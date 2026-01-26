# config.py
# Clean and modern AP1000 thermal-hydraulic configuration

from dataclasses import dataclass, field
import numpy as np

try:
    import CoolProp.CoolProp as CP
except ImportError:
    CP = None


@dataclass
class Config:
    """
    Minimal, clean, and complete configuration for the AP1000
    reactor + primary loop + Delta-125 SG subsystem.
    Only includes values actually required in the physics models.
    """

    # ---------------------------------------------------------
    # Simulation
    # ---------------------------------------------------------
    dt: float = 0.1      # [s]
    t_final: float = 300 # [s]
    startup_warm_time_s: float = 0.0  # [s] pre-sim warmup to settle loops

    # ---------------------------------------------------------
    # Primary loop nominal conditions
    # ---------------------------------------------------------
    P_pri_nom_Pa: float = 15.513e6           # [Pa]
    m_dot_primary_nom_kg_s: float = 15170.145  # [kg/s]
    m_dot_core_kg_s = 14275.56
    T_hot_nom_K: float = 594.261             # [K]
    T_cold_nom_K: float = 553.817            # [K]

    # ---------------------------------------------------------
    # Secondary loop nominal conditions
    # ---------------------------------------------------------
    P_sec_nom_Pa: float = 5.764e6            # [Pa]
    T_fw_K: float = 228.35 + 273.15           # [K]
    m_dot_steam_nom_kg_s: float = 1886.188   # [kg/s]

    P_sec_set_MPa = 5.764
    P_sec_set_Pa = P_sec_set_MPa * 1e6
    Kp_P_mdot_per_MPa = 150.0  # kg/s per MPa (start here)
    Ki_P_mdot_per_MPa_s = 5.0  # kg/s per (MPa*s)
    P_int_limit = 5.0  # MPa*s (clamps integral)
    m_dot_cmd_min = 0.2 * m_dot_steam_nom_kg_s
    m_dot_cmd_max = 1.2 * m_dot_steam_nom_kg_s
    m_dot_cmd_rate_limit = 30.0  # kg/s per second (optional)

    # ---------------------------------------------------------
    # Rated powers and nominal control state
    # ---------------------------------------------------------
    P_core_nom_MWt: float = 3400.0           # [MWt]
    P_e_nom_MWe: float = 1122.0              # [MWe]
    rod_insert_nom: float = 0.57             # [-]
    load_demand_max_pu: float = 1.2          # [-] governor clamp for load demand
    steam_flow_pressure_exp: float = 0.5     # [-] how strongly dome pressure limits SG flow
    steamgen_settle_time_s: float = 30.0     # [s] SG internal warmup duration
    turbine_power_scale: float | None = None # [-] optional manual turbine scaling

    # ---------------------------------------------------------
    # REACTOR CORE (lumped fuel + coolant + kinetics)
    # ---------------------------------------------------------
    # Core thermal masses
    m_fuel_core_kg: float = 95974.7  # UO2 mass
    cp_fuel_core_J_kgK: float = 313.0  # fuel cp
    cp_coolant_core_J_kgK: float = 5541.398

    # Core reference temperatures
    T_f0: float = 1111.680  # Fuel temperature at steady state with bypass flow
    T_m0: float = 575.305  # Kelvin, average coolant temperature (measured with bypass flow effect)
    Tc10: float = 575.305  # Kelvin, first coolant lump temperature at steady state
    Tc20: float = 596.793  # Kelvin, second coolant lump temperature at steady state

    m_coolant_core_kg: float = 12649.23  # kg of coolant in core

    # Fuel-to-coolant heat transfer (lumped)
    U_fc_W_m2K: float = 1172.078  # fuel→coolant HTC
    A_fc_m2: float = 5267.6  # active HT area

    # Reactor power controller gains
    Kp_rc: float = 0.00030 # proportional gain
    Ki_rc: float = 0.000015 # integral gain
    Kpow_i_rc: float = 0.30 # power gain
    Kff_rc: float = 0.10 # feedforward gain
    K_AW_rc: float = 2.0 # Anti-windup back-calculation gain
    LEAK_rc: float = 0.10 # Inner loop integral leak [1/s] - high for stability
    LEAK_OUTER_rc: float = 0.0 # Outer loop integral leak [1/s] - high for stability
    DB_rc_K: float = 0.3 # Temperature deadband [K] - large to prevent hunting
    P_DEADBAND_rc: float = 0.005 # Power deadband [pu] (4%)
    U_MAX_rc: float = 0.0045 # Maximum controller output [1/s] based on physical limits
    Z_BIAS_MAX_rc: float = 8.0 # Maximum power bias [K]

    # Power split / residence time
    f_fuel_fraction: float = 0.974  # fraction of power to fuel
    bypass_fraction: float = 0.059  # 5.9% bypass, if you want it here

    tau_rxi_s: float = 1.68  # vessel inlet → core inlet
    tau_rxu_s: float = 2.06  # core exit → vessel exit
    tau_f = (m_fuel_core_kg * cp_fuel_core_J_kgK) / (U_fc_W_m2K * A_fc_m2)  # fuel-to-coolant heat transfer time constant
    tau_c = (m_coolant_core_kg * cp_coolant_core_J_kgK) / (U_fc_W_m2K * A_fc_m2)  # time constant  for fuel-to-coolant heat transfer
    tau_r = m_coolant_core_kg / m_dot_core_kg_s  # residence time of coolant in the core

    # Rod mechanics
    stroke_in: float = 166.755
    vmax_rod_in_per_min: float = 45.0

    # Feedback coefficients (pcm/°F)
    alpha_D_pcm_per_F: float = -1.4  # Doppler
    alpha_M_pcm_per_F: float = -20.0  # moderator

    # Point kinetics
    Lambda_prompt_s: float = 19.8e-6
    beta_eff: float = 0.0075
    beta_shape_tuple: tuple = (
        0.000221, 0.001467, 0.001313, 0.002647, 0.000771, 0.000281
    )
    lambda_i_tuple: tuple = (
        0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01
    )

    # Rod worth
    rho_rod_tot_pcm: float = 10490.0

    # Nominal metal and secondary temperatures (computed later)
    T_metal_nom_K: float = field(init=False)  # [K]
    T_sec_nom_K: float = field(init=False)    # [K]

    # ---------------------------------------------------------
    # SG geometry / masses / areas
    # ---------------------------------------------------------
    V_steam_m3: float = 147.9             # [m^3] steam dome volume
    m_sec_water_kg: float = 7.9722e4      # [kg] secondary inventory

    m_tube_kg: float = 1.28e5             # [kg] SG metal mass
    cp_metal_J_kgK: float = 524.0         # [J/kg-K]

    A_ht_m2: float = 10936.06             # [m^2] heat transfer area
    #U_W_m2K: float = 7660.0              # [W/m^2-K] primary-side HTC
    U_W_m2K: float = 16000.0 * 1.19

    h_s_W_m2K: float = 38794.0            # [W/m^2-K] secondary HTC (geometric)

    # Primary tube geometry
    D_i_m: float = 0.0154                 # [m]
    L_tube_m: float = 20.63               # [m]
    t_wall_m: float = 0.001016            # [m]
    k_wall_W_mK: float = 13.5             # [W/m-K]

    # Hydraulic volumes (primary-side segments)
    V_hot_m3: float = 7.5
    V_sgi_m3: float = 11.35
    V_sgu_m3: float = 11.35
    V_cold_m3: float = 11.5

    # ---------------------------------------------------------
    # SG CALIBRATED METAL→STEAM CONDUCTANCE KNOB
    # ---------------------------------------------------------
    # This is the value found by sg_calibration:
    #   G_ms_calib_W_K ≈ 1.919e8 W/K
    # which yields:
    #   p_s_eq ≈ 5.764 MPa and
    #   q_ms_eq ≈ m_dot * Δh_fg ≈ 3.417 GW
    #
    # All derived G_ms_W_K and taus below use this calibrated value.
    G_ms_calib_W_K: float = 3.24e8 * 1.025       # [W/K] calibrated knob

    # ---------------------------------------------------------
    # AUTO-COMPUTED FIELDS
    # ---------------------------------------------------------
    rho_primary_kg_m3: float = field(init=False)
    cp_primary_J_kgK: float = field(init=False)

    # Steam / feedwater properties
    T_sat_sec_K: float = field(init=False)
    cp_fw_J_kgK: float = field(init=False)
    h_fw_J_kg: float = field(init=False)
    h_steam_J_kg: float = field(init=False)
    delta_h_steam_fw_J_kg: float = field(init=False)
    delta_h_eff_J_kg: float = field(init=False)
    # compatibility aliases
    h_steam_nom_J_kg: float = field(init=False)
    delta_h_steam_fw_nom_J_kg: float = field(init=False)

    # Primary tube volume
    V_p_m3: float = field(init=False)

    # Heat capacities
    C_p_J_K: float = field(init=False)
    C_m_J_K: float = field(init=False)
    C_s_J_K: float = field(init=False)

    # Conductances actually used by SGCore
    G_pm_W_K: float = field(init=False)
    G_ms_W_K: float = field(init=False)

    # Time constants (taus)
    tau_pm_s: float = field(init=False)
    tau_mp_s: float = field(init=False)
    tau_ms_s: float = field(init=False)
    tau_ms_metal_s: float = field(init=False)
    tau_mp_metal_s: float = field(init=False)

    tau_hot_s: float = field(init=False)
    tau_sgi_s: float = field(init=False)
    tau_sgu_s: float = field(init=False)
    tau_cold_s: float = field(init=False)

    # ---------------------------------------------------------
    #                 POST-INIT COMPUTATION
    # ---------------------------------------------------------
    def __post_init__(self):
        # ---------------------------------------------
        # Primary coolant properties (ρ, cp)
        # ---------------------------------------------
        T_ref = 0.5 * (self.T_hot_nom_K + self.T_cold_nom_K)

        if CP:
            rho = CP.PropsSI("D", "P", self.P_pri_nom_Pa, "T", T_ref, "Water")
            cp_p = CP.PropsSI("C", "P", self.P_pri_nom_Pa, "T", T_ref, "Water")
        else:
            rho = 720.0
            cp_p = 5500.0

        object.__setattr__(self, "rho_primary_kg_m3", rho)
        object.__setattr__(self, "cp_primary_J_kgK", cp_p)

        # Nominal metal temperature ~ average of hot and cold legs
        T_metal_nom = 0.5 * (self.T_hot_nom_K + self.T_cold_nom_K)
        object.__setattr__(self, "T_metal_nom_K", T_metal_nom)

        # ---------------------------------------------
        # Secondary saturation temperature
        # ---------------------------------------------
        if CP:
            T_sat = CP.PropsSI("T", "P", self.P_sec_nom_Pa, "Q", 0, "Water")
        else:
            T_sat = 546.15  # [K] approximate
        object.__setattr__(self, "T_sat_sec_K", T_sat)
        object.__setattr__(self, "T_sec_nom_K", T_sat)

        # ---------------------------------------------
        # Steam & feedwater enthalpies
        # ---------------------------------------------
        if CP:
            h_s = CP.PropsSI("H", "P", self.P_sec_nom_Pa, "Q", 1, "Water")
            h_fw = CP.PropsSI("H", "P", self.P_sec_nom_Pa, "T", self.T_fw_K, "Water")
            cp_fw = CP.PropsSI("C", "P", self.P_sec_nom_Pa, "T", self.T_fw_K, "Water")
        else:
            h_fw = 0.0
            cp_fw = 4200.0
            h_s = 1.803e6


        #delta_fw = h_s - h_fw
        delta_fw = 1.803e6
        delta_eff = h_s - cp_fw * self.T_fw_K

        object.__setattr__(self, "h_fw_J_kg", h_fw)
        object.__setattr__(self, "h_steam_J_kg", h_s)
        object.__setattr__(self, "cp_fw_J_kgK", cp_fw)
        object.__setattr__(self, "delta_h_steam_fw_J_kg", delta_fw)
        object.__setattr__(self, "delta_h_eff_J_kg", delta_eff)
        # compatibility aliases
        object.__setattr__(self, "h_steam_nom_J_kg", h_s)
        object.__setattr__(self, "delta_h_steam_fw_nom_J_kg", delta_fw)

        # ---------------------------------------------
        # Primary tube volume
        # ---------------------------------------------
        A_i = np.pi * (self.D_i_m / 2.0) ** 2
        # scale tube count to match total heat transfer area
        V_p = A_i * self.L_tube_m * (self.A_ht_m2 / (np.pi * self.D_i_m * self.L_tube_m))
        object.__setattr__(self, "V_p_m3", V_p)

        # Heat capacities
        C_p = rho * V_p * cp_p
        C_m = self.m_tube_kg * self.cp_metal_J_kgK
        C_s = self.m_sec_water_kg * 4200.0  # [J/kg-K] water approx

        object.__setattr__(self, "C_p_J_K", C_p)
        object.__setattr__(self, "C_m_J_K", C_m)
        object.__setattr__(self, "C_s_J_K", C_s)

        # ---------------------------------------------
        # Conductances
        # ---------------------------------------------
        # Primary side: from geometry + U
        G_pm = self.U_W_m2K * self.A_ht_m2

        # Metal → steam: USE CALIBRATED VALUE, NOT h_s*A
        G_ms = self.G_ms_calib_W_K

        object.__setattr__(self, "G_pm_W_K", G_pm)
        object.__setattr__(self, "G_ms_W_K", G_ms)

        # ---------------------------------------------
        # Time constants
        # ---------------------------------------------
        object.__setattr__(self, "tau_pm_s", C_p / G_pm)
        object.__setattr__(self, "tau_mp_s", C_m * (1.0 / G_pm + 1.0 / G_ms))
        object.__setattr__(self, "tau_ms_s", C_s / G_ms)
        object.__setattr__(self, "tau_ms_metal_s", C_m / max(G_ms, 1e-12))
        object.__setattr__(self, "tau_mp_metal_s", C_m / max(G_pm, 1e-12))

        # Hydraulic taus
        def tau_h(V):
            return rho * V / self.m_dot_primary_nom_kg_s

        object.__setattr__(self, "tau_hot_s", tau_h(self.V_hot_m3))
        object.__setattr__(self, "tau_sgi_s", tau_h(self.V_sgi_m3))
        object.__setattr__(self, "tau_sgu_s", tau_h(self.V_sgu_m3))
        object.__setattr__(self, "tau_cold_s", tau_h(self.V_cold_m3))
