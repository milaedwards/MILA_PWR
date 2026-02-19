from dataclasses import dataclass, field

@dataclass
class Config:
    # GLOBALS
    dt: float = 0.1                                   # [s] integration time step
    t_final: float = 3600.0                           # [s] default run length

    # PRIMARY SIDE
    P_pri_nom_Pa: float = 15.513e6                    # [Pa] nominal RCS pressure
    T_hot_nom_K: float = 586.76
    T_cold_nom_K: float = 545.89
    m_dot_primary_nom_kg_s: float = 15170.145         # [kg/s] total RCS flow (all loops)
    m_dot_core_kg_s: float = 14275.6                  # [kg/s] core flow (DCD)
    rho_primary_nom_kg_m3: float = 720.0              # [kg/m^3] nominal primary coolant density

    # SECONDARY SIDE
    P_sec_nom_Pa: float = 5.764e6                     # [Pa] nominal SG secondary pressure
    T_sec_nom_K: float = 546.129                      # [K] nominal steam saturation temperature (at P_sec_nom)
    T_metal_nom_K: float = 574.039                    # [K] nominal SG metal temperature
    T_fw_nom_K: float = 226.7 + 273.15                # [K] nominal feedwater temperature

    m_dot_steam_nom_override_kg_s: float = 1774.9198  # [kg/s] nominal steam mass flow

    cp_fw_J_kgK: float = 4200.0                       # [J/kg-K] feedwater cp (lumped)

    # CORE / PLANT NOMINAL POWERS
    P_core_nom_MWt: float = 3417.8                    # [MWt] rated core thermal power
    P_e_nom_MWe: float = 1127.87                      # [MWe] net electric power
    rod_insert_nom: float = 0.57                      # [-] nominal rod insertion

    # MATERIAL PROPERTIES
    cp_fuel_core_J_kgK: float = 313.0                 # [J/kg-K] effective fuel cp
    cp_coolant_core_J_kgK: float = 5511.4             # [J/kg-K] effective coolant cp

    V_core_coolant_m3: float = 18.0                   # [m^3] coolant volume in core (effective)
    m_fuel_core_kg: float = 95974.7                   # [kg] effective fuel mass (lumped)
    m_coolant_core_kg: float = 12649.23               # [kg] effective core coolant mass (fixed)

    # Core reference temperatures
    T_f0: float = 1104.0                              # [K] steady-state fuel temperature
    T_m0: float = 567.6                               # [K] average coolant temperature
    Tc10: float = 567.6                               # [K] coolant lump 1 temperature
    Tc20: float = 589.3                               # [K] coolant lump 2 temperature

    # CORE HEAT-TRANSFER AREAS AND COEFFICIENTS
    A_fc_m2: float = 5267.6                           # [m^2] fuel-to-coolant area (lumped)
    U_fc_W_m2K: float = 1178.214                      # [W/m^2-K] fuel-to-coolant U

    # PRIMARY LOOP VOLUMES
    V_hot_m3: float = 7.5                             # [m^3] hot leg effective volume
    V_sg_plenums_m3: float = 16.65                    # [m^3] SG primary plenums total (inlet+outlet)
    V_sgi_m3: float = 8.325                           # [m^3] SG inlet plenum volume
    V_sgu_m3: float = 8.325                           # [m^3] SG outlet plenum volume
    V_cold_m3: float = 11.5                           # [m^3] cold leg effective volume

    # SG GEOMETRY (FROM DCD TABLE – SG ONLY)
    V_sg_tubes_m3: float = 42.16                      # [m^3] primary water in SG tubes
    V_sec_water_m3: float = 103.24                    # [m^3] secondary water volume
    V_steam_m3: float = 147.87                        # [m^3] steam/dome volume
    m_sec_water_kg: float = 79728.0                   # [kg] secondary water mass (effective)

    # SG CALIBRATED METAL TO STEAM CONDUCTANCE + SG PI GAINS
    G_ms_calib_W_K: float = 1.218199e8                # [W/K] calibrated conductance knob
    G_ms1_frac: float = 0.5                           # [-] fraction of G_ms applied to metal node 1
    G_ms2_frac: float = 0.5                           # [-] fraction of G_ms applied to metal node 2

    Kp_flow_per_Pa: float = 1.2e-4                    # [kg/s per Pa] steam-flow trim proportional gain
    Ki_flow_per_Pa_s: float = 4.0e-5                  # [kg/s per (Pa*s)] steam-flow trim integral gain

    # STEAM GENERATOR SECONDARY PRESSURE DYNAMICS (Level 3)
    sg_KpP_Pa_per_kg_s: float = 3000.0                # [Pa/(kg/s)] pressure sensitivity to steam mismatch
    sg_tauP_s: float = 5.0                            # [s] pressure lag time constant
    sg_dPmax_Pa: float = 0.5e6                        # [Pa] clamp around nominal (+/-)
    sg_Pmin_Pa: float = 1.0e5                         # [Pa] absolute min pressure clamp
    sg_Pmax_Pa: float = 25.0e6                        # [Pa] absolute max pressure clamp
    sg_tau_sep_s: float = 8.0                         # [s] steam separator / dome lag (delivered steam inertia)
    sg_tau_valve_s: float = 2.0                       # [s] steam valve / demand lag time constant

    # REACTOR CONTROL CONSTANTS
    Kp_rc: float = 0.00030                            # [pcm/K] proportional gain
    Ki_rc: float = 0.000015                           # [pcm/(K*s)] integral gain
    Kpow_i_rc: float = 0.30                           # [-] inner power-loop integral gain
    Kff_rc: float = 0.10                              # [-] feedforward gain
    K_AW_rc: float = 2.0                              # [-] anti-windup gain
    LEAK_rc: float = 0.10                             # [-] inner-loop integral leak
    LEAK_OUTER_rc: float = 0.01                       # [-] outer-loop integral leak

    DB_rc_K: float = 0.3                              # [K] temperature deadband
    P_DEADBAND_rc: float = 0.005                      # [pu] power deadband
    U_MAX_rc: float = 0.0045                          # [1/s] max controller output
    Z_BIAS_MAX_rc: float = 8.0                        # [K] max power bias

    f_fuel_fraction: float = 0.974                    # [-] fraction of power to fuel
    bypass_fraction: float = 0.059                    # [-] bypass fraction

    tau_rxi_s: float = 1.68                           # [s] vessel inlet → core inlet
    tau_rxu_s: float = 2.06                           # [s] core exit → vessel exit

    # Rod mechanics
    stroke_in: float = 166.755                        # [in] total stroke
    vmax_rod_in_per_min: float = 45.0                 # [in/min] max speed

    # Feedback coefficients (pcm/°F)
    alpha_D_pcm_per_F: float = -1.4                   # [pcm/°F] Doppler
    alpha_M_pcm_per_F: float = -20.0                  # [pcm/°F] moderator

    # Point kinetics
    Lambda_prompt_s: float = 19.8e-6                  # [s] prompt generation time
    beta_eff: float = 0.0075                          # [-] effective delayed neutron fraction
    beta_shape_tuple: tuple = (
        0.000221, 0.001467, 0.001313, 0.002647, 0.000771, 0.000281
    )                                                 # [-] delayed group fractions
    lambda_shape_tuple: tuple = (
        0.0124, 0.0305, 0.1110, 0.3010, 1.1400, 3.0100
    )                                                 # [1/s] delayed group decay constants

    rho_rod_tot_pcm: float = 10490.0                  # [pcm] total rod worth

    # AUTO-COMPUTED FIELDS (set in __post_init__)
    # Aliases
    cp_fuel_J_kgK: float = field(init=False)          # [J/kg-K]
    cp_coolant_J_kgK: float = field(init=False)       # [J/kg-K]

    # Nominal average temperature
    Tavg_nom_K: float = field(init=False)             # [K]

    # Steam / feed enthalpies (fixed constants, no CP)
    h_steam_J_kg: float = field(init=False)           # [J/kg] nominal steam enthalpy
    delta_h_steam_fw_J_kg: float = field(init=False)  # [J/kg] nominal enthalpy rise
    m_dot_steam_nom_kg_s: float = field(init=False)   # [kg/s] nominal steam flow actually used

    # Conductances used by SGCore
    G_ms_W_K: float = field(init=False)               # [W/K]

    # Time constants (taus)
    tau_pm_s: float = field(init=False)               # [s] primary → metal node 1
    tau_mp_s: float = field(init=False)               # [s] metal node 1 → metal node 2 (effective)
    tau_ms_s: float = field(init=False)               # [s] metal → secondary/steam (legacy tau)

    tau_hot_s: float = field(init=False)              # [s] RX upper → hot leg residence/transport
    tau_sgi_s: float = field(init=False)              # [s] hot leg → SG inlet residence/transport
    tau_sgu_s: float = field(init=False)              # [s] SG outlet → cold leg residence/transport (tube node 2 → SG outlet)
    tau_cold_s: float = field(init=False)             # [s] SG outlet → reactor inlet residence/transport

    # Core thermal time constants
    tau_f: float = field(init=False)                  # [s] fuel thermal time constant
    tau_c: float = field(init=False)                  # [s] coolant thermal time constant
    tau_r: float = field(init=False)                  # [s] coolant residence time

    # Legacy alias
    lambda_i_tuple: tuple = field(init=False)         # [1/s] same as lambda_shape_tuple

    # POST-INIT COMPUTATION
    def __post_init__(self):
        # Aliases / backward-compatible names
        self.m_dot_steam_nom_kg_s = float(self.m_dot_steam_nom_override_kg_s)
        self.cp_fuel_J_kgK = float(self.cp_fuel_core_J_kgK)
        self.cp_coolant_J_kgK = float(self.cp_coolant_core_J_kgK)

        # Nominal average temperature
        self.Tavg_nom_K = 0.5 * (self.T_hot_nom_K + self.T_cold_nom_K)

        # Legacy alias
        self.lambda_i_tuple = self.lambda_shape_tuple

        # Steam / feed enthalpies (FIXED CONSTANTS, no CP)
        self.h_steam_J_kg = 2.787e6                   # [J/kg] nominal saturated vapor enthalpy (approx at 5.764 MPa)
        h_fw = 9.75e5                                 # [J/kg] nominal feedwater enthalpy (approx)
        self.delta_h_steam_fw_J_kg = self.h_steam_J_kg - h_fw  # [J/kg]

        # Nominal steam flow used everywhere
        self.m_dot_steam_nom_kg_s = float(self.m_dot_steam_nom_override_kg_s)  # [kg/s]

        # Conductances
        self.G_ms_W_K = float(self.G_ms_calib_W_K)    # [W/K]

        # SG node masses / capacitances (lumped, DCD-based intent)
        # Primary mass in SG tubes (fixed density assumption)
        rho_sg_primary_kg_m3 = 726.0                           # [kg/m^3] fixed density used for SG tube primary water
        m_p = self.V_sg_tubes_m3 * rho_sg_primary_kg_m3        # [kg]
        m_m = 0.5 * self.m_fuel_core_kg                        # [kg] effective metal mass (lumped)
        m_s = self.m_sec_water_kg                              # [kg] secondary water mass (effective)

        # Primary to metal conductance roughly tied to core U·A
        G_pm_W_K = 0.25 * self.U_fc_W_m2K * self.A_fc_m2       # [W/K]
        G_ms_W_K = self.G_ms_W_K                               # [W/K]

        # Capacitances
        C_p_J_K = m_p * self.cp_coolant_J_kgK                  # [J/K]
        C_m_J_K = m_m * self.cp_fuel_J_kgK                     # [J/K]
        C_s_J_K = m_s * 4200.0                                 # [J/K] secondary-side effective cp assumption

        # Time constants
        self.tau_pm_s = C_p_J_K / G_pm_W_K if G_pm_W_K > 0.0 else 0.0    # [s]
        if (G_pm_W_K > 0.0) and (G_ms_W_K > 0.0):
            self.tau_mp_s = C_m_J_K * (1.0 / G_pm_W_K + 1.0 / G_ms_W_K)  # [s]
        else:
            self.tau_mp_s = 0.0
        self.tau_ms_s = C_s_J_K / G_ms_W_K if G_ms_W_K > 0.0 else 0.0    # [s]

        # Hydraulic taus (residence/transport), fixed density
        rho = float(self.rho_primary_nom_kg_m3)                # [kg/m^3]

        def tau_h(V_m3: float) -> float:
            if self.m_dot_primary_nom_kg_s <= 0.0:
                return 0.0
            return (rho * V_m3) / self.m_dot_primary_nom_kg_s  # [s]

        self.tau_hot_s = tau_h(self.V_hot_m3)   # [s]
        self.tau_sgi_s = tau_h(self.V_sgi_m3)   # [s]
        self.tau_sgu_s = tau_h(self.V_sgu_m3)   # [s]
        self.tau_cold_s = tau_h(self.V_cold_m3) # [s]

        # Core thermal time constants (Holbert-style), derived here
        den_W_K = self.U_fc_W_m2K * self.A_fc_m2               # [W/K]
        if den_W_K > 0.0:
            self.tau_f = (self.m_fuel_core_kg * self.cp_fuel_J_kgK) / den_W_K        # [s]
            self.tau_c = (self.m_coolant_core_kg * self.cp_coolant_J_kgK) / den_W_K  # [s]
        else:
            self.tau_f = 0.0
            self.tau_c = 0.0

        self.tau_r = (
            self.m_coolant_core_kg / self.m_dot_primary_nom_kg_s
            if self.m_dot_primary_nom_kg_s > 0.0 else 0.0
        )