from dataclasses import dataclass, replace
from config_2 import Config

cfg = Config()

@dataclass
class PlantState:
    # Time
    t_s: float = 0.0

    # Primary loop
    T_hot_K: float = cfg.T_HOT_INIT_K
    T_cold_K: float = cfg.T_COLD_INIT_K
    P_primary_Pa: float = cfg.P_PRI_INIT_Pa
    m_dot_primary_kg_s: float = cfg.M_DOT_PRI

    # Steam/secondary
    P_secondary_Pa: float = cfg.P_SEC_INIT_Pa
    T_steam_K: float = cfg.T_SAT_SEC_K
    m_dot_steam_kg_s: float = cfg.M_DOT_SEC_KG_S
    sg_level_m: float = 0.0

    # Power + reactivity (stored in MW)
    P_core_MW: float = cfg.Q_CORE_NOMINAL_MW
    P_turbine_MW: float = cfg.P_RATED_MWe
    rod_pos_pu: float = cfg.ROD_INSERT_INIT      # 0=withdrawn, 1=inserted
    rho_reactivity_dk: float = 0.0               # Δk/k

    # user commands
    load_demand_pu: float = 1.0                  # 0..1
    rod_mode: str = "auto"
    rod_cmd_manual_pu: float = 0.55              # used when rod_mode='manual'

    @property
    def Tavg_K(self) -> float:
        return 0.5 * (self.T_hot_K + self.T_cold_K)

    def copy_advance_time(self, dt: float) -> "PlantState":
        return replace(self, t_s=self.t_s + dt)

    def clip_invariants(self) -> "PlantState":
        # bounds (broad; move to Config later if needed)
        T_hot  = float(max(250.0, min(1100.0, self.T_hot_K)))
        T_cold = float(max(250.0, min(1100.0, self.T_cold_K)))
        P_pri  = float(max(1.0e5, self.P_primary_Pa))
        P_sec  = float(max(1.0e5, self.P_secondary_Pa))
        mdotp  = float(max(0.0, self.m_dot_primary_kg_s))
        mdots  = float(max(0.0, self.m_dot_steam_kg_s))
        Tsteam = float(max(250.0, min(1100.0, self.T_steam_K)))
        # pzrL   = float(max(0.0, self.pzr_level_m))
        sgL    = float(max(0.0, self.sg_level_m))
        rod    = float(min(1.0, max(0.0, self.rod_pos_pu)))
        ld     = float(min(1.0, max(0.0, self.load_demand_pu)))
        rodm   = float(min(1.0, max(-1.0, self.rod_cmd_manual_pu)))

        return replace(self,
            T_hot_K=T_hot, T_cold_K=T_cold, P_primary_Pa=P_pri, P_secondary_Pa=P_sec,
            m_dot_primary_kg_s=mdotp, m_dot_steam_kg_s=mdots,
            T_steam_K=Tsteam, sg_level_m=sgL,
            rod_pos_pu=rod, load_demand_pu=ld, rod_cmd_manual_pu=rodm
        )

    @staticmethod
    def init_default() -> "PlantState":
        return PlantState()

    @staticmethod
    def init_from_config(custom_cfg) -> "PlantState":
        # Start from current defaults
        base = PlantState()

        mapping = {
            "T_HOT_INIT_K": "T_hot_K",
            "T_COLD_INIT_K": "T_cold_K",
            "P_PRI_INIT_PA": "P_primary_Pa",
            "M_DOT_PRI": "m_dot_primary_kg_s",
            "P_SEC_INIT_PA": "P_secondary_Pa",
            "T_SAT_SEC_K": "T_steam_K",
            "M_DOT_SEC": "m_dot_steam_kg_s",
            "Q_CORE_NOMINAL_MW": "P_core_MW",
            "ROD_INSERT_INIT": "rod_pos_pu",
        }
        data = base.__dict__.copy()
        for cfg_key, field_name in mapping.items():
            if hasattr(custom_cfg, cfg_key):
                data[field_name] = getattr(custom_cfg, cfg_key)
        return PlantState(**data).clip_invariants()