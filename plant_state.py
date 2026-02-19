from dataclasses import dataclass, replace
from config import Config

cfg = Config()

@dataclass
class PlantState:
    # Time
    t_s: float = 0.0

    # Primary loop
    T_hot_K: float = cfg.T_hot_nom_K
    T_cold_K: float = cfg.T_cold_nom_K
    P_primary_Pa: float = cfg.P_pri_nom_Pa
    m_dot_primary_kg_s: float = cfg.m_dot_primary_nom_kg_s

    # Pressurizer
    P_pzr_Pa: float = cfg.P_pri_nom_Pa
    pzr_level_m: float = 0.0
    pzr_heater_pu: float = 0.0
    pzr_spray_pu: float = 0.0
    pzr_heater_frac: float = 0.0               # continuous heater output 0-1 (after lag)
    pzr_heater_kW: float = 0.0                 # heater thermal power [kW]
    pzr_surge_direction: str = "NEUTRAL"       # "IN-SURGE", "OUT-SURGE", or "NEUTRAL"
    pzr_pressure_setpoint_Pa: float = 15.50e6  # shifted pressure setpoint [Pa]

    # Steam generator temperatures (for plotting)
    # Primary side around the SG
    T_sg_in_K: float = cfg.T_hot_nom_K
    T_sg_out_K: float = cfg.T_cold_nom_K

    T_metal_K: float = cfg.T_metal_nom_K  # average metal temperature
    T_sec_K: float = cfg.T_sec_nom_K      # steam dome temperature

    m_dot_steam_cmd_kg_s: float = cfg.m_dot_steam_nom_kg_s

    # Steam/secondary
    P_secondary_Pa: float = cfg.P_sec_nom_Pa
    m_dot_steam_kg_s: float = cfg.m_dot_steam_nom_kg_s
    steam_h_J_kg: float = cfg.h_steam_J_kg
    sg_power_limited: bool = False

    # Power + reactivity (stored in MW)
    P_core_MW: float = cfg.P_core_nom_MWt
    P_turbine_MW: float = cfg.P_e_nom_MWe * 1.0 # ~1095 MWe
    rod_pos_pu: float = cfg.rod_insert_nom      # 0 = withdrawn, 1 = inserted
    rho_reactivity_dk: float = 0.0              # Δk/k

    # user commands
    load_demand_pu: float = 1.0
    rod_mode: str = "auto"
    rod_cmd_manual_pu: float = 0.0              # used when rod_mode = 'manual'

    @property
    def Tavg_K(self) -> float:
        return 0.5 * (self.T_hot_K + self.T_cold_K)

    def copy_advance_time(self, dt: float) -> "PlantState":
        return replace(self, t_s=self.t_s + dt)

    def clip_invariants(self) -> "PlantState":
        T_hot = float(max(250.0, min(1100.0, self.T_hot_K)))
        T_cold = float(max(250.0, min(1100.0, self.T_cold_K)))
        P_pri = float(max(1.0e5, self.P_primary_Pa))
        P_pzr = float(max(1.0e5, self.P_pzr_Pa))
        P_sec = float(max(1.0e5, self.P_secondary_Pa))
        mdotp = float(max(0.0, self.m_dot_primary_kg_s))
        mdots = float(max(0.0, self.m_dot_steam_kg_s))
        Tsteam = float(max(250.0, min(1100.0, self.T_sec_K)))
        Tmetal = float(max(250.0, min(1100.0, self.T_metal_K)))
        rod = float(max(0.0, min(1.0, self.rod_pos_pu)))
        ld = float(max(0.0, min(1.2, self.load_demand_pu)))
        rodm = float(max(-1.0, min(1.0, self.rod_cmd_manual_pu)))
        hsteam = float(self.steam_h_J_kg)

        pzr_level = float(max(0.0, min(20.0, self.pzr_level_m)))
        pzr_heater = float(max(0.0, min(1.0, self.pzr_heater_pu)))
        pzr_spray = float(max(0.0, min(1.0, self.pzr_spray_pu)))
        pzr_heater_frac = float(max(0.0, min(1.0, self.pzr_heater_frac)))
        pzr_heater_kW = float(max(0.0, self.pzr_heater_kW))

        return replace(
            self,
            T_hot_K=T_hot,
            T_cold_K=T_cold,
            P_primary_Pa=P_pri,
            P_pzr_Pa=P_pzr,
            P_secondary_Pa=P_sec,
            m_dot_primary_kg_s=mdotp,
            m_dot_steam_kg_s=mdots,
            steam_h_J_kg=hsteam,
            sg_power_limited=self.sg_power_limited,
            T_sec_K=Tsteam,
            T_metal_K=Tmetal,
            rod_pos_pu=rod,
            load_demand_pu=ld,
            rod_cmd_manual_pu=rodm,
            pzr_level_m=pzr_level,
            pzr_heater_pu=pzr_heater,
            pzr_spray_pu=pzr_spray,
            pzr_heater_frac=pzr_heater_frac,
            pzr_heater_kW=pzr_heater_kW,
            pzr_surge_direction=self.pzr_surge_direction,
            pzr_pressure_setpoint_Pa=self.pzr_pressure_setpoint_Pa,
        )