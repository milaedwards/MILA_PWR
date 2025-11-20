import numpy as np

try:
    import CoolProp.CoolProp as clp
except ImportError:
    clp = None


class TauConfig:
    """
    AP1000 RCS + Delta-125 SG configuration and derived constants.
    Only fields used by SG are kept.
    """

    def __init__(self, use_coolprop: bool = True):
        if not (use_coolprop and clp is not None):
            raise ValueError("CoolProp is required for TauConfig in this setup.")

        # ---------- RCS conditions ----------
        self.P_RCS_Pa = 15.5e6  # primary pressure [Pa]
        self.T_cold_K = 280.7 + 273.15  # cold leg temperature [K]
        self.T_hot_K = 321.1 + 273.15  # hot leg temperature [K]
        self.m_core_kg_per_hr = 5.148e7  # core mass flow [kg/hr]

        # ---------- RCS volumes ----------
        self.V_RXU_m3 = 17.5  # upper plenum [m³] (not used in SG)
        self.V_RXI_m3 = 47.0  # lower plenum [m³]
        self.V_HOT_m3 = 7.5  # hot leg per loop [m³]
        self.V_COLD_m3 = 11.5  # cold leg per loop [m³]
        self.V_SGI_m3 = 11.35  # SG inlet plenum per loop [m³]
        self.V_SGU_m3 = 11.35  # SG outlet plenum per loop [m³]

        # ---------- SG secondary ----------
        self.P_SG_sec_Pa = 5.76e6  # steam pressure [Pa]
        self.T_fw_K = 226.7 + 273.15  # feedwater temperature [K]
        self.mdot_fw_kg_s = 1887.0  # feedwater nominal flow [kg/s]

        # ---------- SG geometry ----------
        self.N_tubes = 10025  # number of tubes [-]
        self.L_tube_m = 20.63  # tube length [m]
        self.D_i_m = 0.0154  # inner diameter [m]
        self.t_wall_m = 0.001016  # wall thickness [m]
        self.D_o_m = self.D_i_m + 2.0 * self.t_wall_m
        self.A_ht_m2 = 10936.06  # heat-transfer area [m²]
        self.m_tube_kg = 1.28e5  # tube metal mass [kg]
        self.m_sec_water_kg = 7.9722e4  # secondary mass [kg]
        self.V_steam_m3 = 147.9  # steam dome volume [m³]

        # ---------- Heat transfer ----------
        self.U_W_m2K = 7660.0  # overall HTC [W/m²-K]
        self.h_s_W_m2K = 38794.0  # secondary HTC [W/m²-K]

        # ---------- Metal properties ----------
        self.k_690_W_mK = 13.5  # Inconel-690 k [W/m-K]
        self.cp_690_J_kgK = 524.0  # cp [J/kg-K]

        # ---------- Primary volume ----------
        self.V_p_m3 = (
            self.N_tubes
            * np.pi
            * (self.D_i_m / 2.0) ** 2
            * self.L_tube_m
        )  # primary volume in tubes [m³]

        # ---------- Fluid properties (CoolProp) ----------
        T_ref_p = 0.5 * (self.T_hot_K + self.T_cold_K)  # primary average [K]

        self.rho_p_kg_m3 = clp.PropsSI("D", "P", self.P_RCS_Pa, "T", T_ref_p, "Water")
        self.cp_p_J_kgK = clp.PropsSI("C", "P", self.P_RCS_Pa, "T", T_ref_p, "Water")

        self.T_sat_sec_K = clp.PropsSI("T", "P", self.P_SG_sec_Pa, "Q", 0.0, "Water")
        self.cp_s_J_kgK = clp.PropsSI("C", "P", self.P_SG_sec_Pa, "T", self.T_sat_sec_K, "Water")

        cp_fw_J_kgK = clp.PropsSI("C", "P", self.P_SG_sec_Pa, "T", self.T_fw_K, "Water")
        h_steam_J_kg = clp.PropsSI("H", "P", self.P_SG_sec_Pa, "Q", 1.0, "Water")

        # h_ws - cp_fw * T_fw [J/kg]
        self.H_ws_minus_cpTfw_J_kg = h_steam_J_kg - cp_fw_J_kgK * self.T_fw_K

        # ---------- Masses & heat capacities ----------
        self.m_p_kg = self.rho_p_kg_m3 * self.V_p_m3  # primary mass [kg]
        self.C_p_J_K = self.m_p_kg * self.cp_p_J_kgK  # primary C [J/K]
        self.C_m_J_K = self.m_tube_kg * self.cp_690_J_kgK  # metal C [J/K]
        self.C_s_J_K = self.m_sec_water_kg * self.cp_s_J_kgK  # secondary C [J/K]

        # ---------- Thermal resistances ----------
        self.R_total_m2K_W = 1.0 / self.U_W_m2K  # total [m²-K/W]
        self.R_wall_m2K_W = self.t_wall_m / self.k_690_W_mK
        self.R_s_m2K_W = 1.0 / self.h_s_W_m2K

        self.R_p_m2K_W = self.R_total_m2K_W - self.R_wall_m2K_W - self.R_s_m2K_W
        if self.R_p_m2K_W <= 0.0:
            raise ValueError("Computed primary thermal resistance is non-positive.")

        self.h_p_W_m2K = 1.0 / self.R_p_m2K_W  # primary HTC [W/m²-K]

        # ---------- Conductances ----------
        self.G_pm_W_K = self.h_p_W_m2K * self.A_ht_m2  # primary→metal [W/K]
        self.G_ms_W_K = self.h_s_W_m2K * self.A_ht_m2  # metal→secondary [W/K]

        # ---------- Global SG taus ----------
        self.tau_pm_s = self.C_p_J_K / self.G_pm_W_K  # primary↔metal [s]
        self.tau_mp_s = self.C_m_J_K * (
            1.0 / self.G_pm_W_K + 1.0 / self.G_ms_W_K
        )  # metal↔primary [s]
        self.tau_ms_s = self.C_s_J_K / self.G_ms_W_K  # metal↔secondary [s]

        # ---------- RCS hydraulic taus ----------
        self.m_core_kg_s = self.m_core_kg_per_hr / 3600.0  # core flow [kg/s]
        self.m_loop_kg_s = self.m_core_kg_s / 2.0  # loop flow [kg/s]
        self.rho_core_kg_m3 = self.rho_p_kg_m3  # coolant density [kg/m³]

        def tau_h(V_m3: float, m_dot_kg_s: float) -> float:
            return self.rho_core_kg_m3 * V_m3 / m_dot_kg_s  # [s]

        self.tau_hot_s = tau_h(self.V_HOT_m3, self.m_loop_kg_s)  # hot leg [s]
        self.tau_cold_s = tau_h(self.V_COLD_m3, self.m_loop_kg_s)  # cold leg [s]
        self.tau_sgi_s = tau_h(self.V_SGI_m3, self.m_loop_kg_s)  # SG inlet [s]
        self.tau_sgu_s = tau_h(self.V_SGU_m3, self.m_loop_kg_s)  # SG outlet [s]
        self.tau_rxi_s = tau_h(self.V_RXI_m3, self.m_core_kg_s)  # lower plenum [s]
