from dataclasses import dataclass, replace

@dataclass
class PlantState:
    # Time
    t_s: float = 0.0

    # Primary loop
    T_hot_K: float = 595.0
    T_cold_K: float = 553.0
    P_primary_Pa: float = 15.5e6
    m_dot_primary_kg_s: float = 0.0

    # Pressurizer
    pzr_pressure_Pa: float = 15.5e6
    pzr_level_m: float = 0.0

    # Steam/secondary
    P_secondary_Pa: float = 6.0e6
    T_steam_K: float = 546.0
    m_dot_steam_kg_s: float = 0.0
    sg_level_m: float = 0.0

    # Power & reactivity
    P_core_W: float = 3.4e9
    P_turbine_W: float = 0.0
    rod_pos_pu: float = 0.55  # 0=withdrawn, 1=inserted
    rho_reactivity_dk: float = 0.0  # Δk/k

    # External/user commands (for logging & plotting)
    load_demand_pu: float = 1.0  # 0..1
    rod_mode: str = "auto"  # 'auto' | 'manual'
    rod_cmd_manual_pu: float = 0.55  # used when rod_mode='manual'

    # ----------- helpers -----------
    @property
    def Tavg_K(self) -> float:
        return 0.5 * (self.T_hot_K + self.T_cold_K)

    def copy_advance_time(self, dt: float) -> "PlantState":
        return replace(self, t_s=self.t_s + dt)

    def clip_invariants(self) -> "PlantState":
        # bounds (broad; move to Config later if needed)
        T_hot = float(max(250.0, min(1100.0, self.T_hot_K)))
        T_cold = float(max(250.0, min(1100.0, self.T_cold_K)))
        P_pri = float(max(1.0e5, self.P_primary_Pa))
        P_sec = float(max(1.0e5, self.P_secondary_Pa))
        pzrP = float(max(1.0e5, self.pzr_pressure_Pa))
        mdotp = float(max(0.0, self.m_dot_primary_kg_s))
        mdots = float(max(0.0, self.m_dot_steam_kg_s))
        Tsteam = float(max(250.0, min(1100.0, self.T_steam_K)))
        pzrL = float(max(0.0, self.pzr_level_m))
        sgL = float(max(0.0, self.sg_level_m))
        rod = float(min(1.0, max(0.0, self.rod_pos_pu)))
        ld = float(min(1.0, max(0.0, self.load_demand_pu)))
        rodm = float(min(1.0, max(0.0, self.rod_cmd_manual_pu)))
        return replace(self,
                       T_hot_K=T_hot, T_cold_K=T_cold, P_primary_Pa=P_pri, P_secondary_Pa=P_sec,
                       pzr_pressure_Pa=pzrP, m_dot_primary_kg_s=mdotp, m_dot_steam_kg_s=mdots,
                       T_steam_K=Tsteam, pzr_level_m=pzrL, sg_level_m=sgL,
                       rod_pos_pu=rod, load_demand_pu=ld, rod_cmd_manual_pu=rodm
                       )


@staticmethod
def init_default() -> "PlantState":
    from config import Config
    cfg = Config()
    return PlantState.init_from_config(cfg)


@staticmethod
def init_from_config(cfg) -> "PlantState":
    return PlantState(
        t_s=0.0,
        T_hot_K=getattr(cfg, "T_HOT_INIT_K", 595.0),
        T_cold_K=getattr(cfg, "T_COLD_INIT_K", 553.0),
        P_primary_Pa=getattr(cfg, "P_PRI_INIT_PA", 15.5e6),
        m_dot_primary_kg_s=getattr(cfg, "M_DOT_PRI", 1.0e4),
        pzr_pressure_Pa=getattr(cfg, "P_PRI_INIT_PA", 15.5e6),
        pzr_level_m=0.0,
        P_secondary_Pa=getattr(cfg, "P_SEC_INIT_PA", 6.0e6),
        T_steam_K=getattr(cfg, "T_SAT_SEC_K", 546.0),
        m_dot_steam_kg_s=getattr(cfg, "M_DOT_SEC", 9.0e3),
        sg_level_m=0.0,
        P_core_W=getattr(cfg, "Q_CORE_NOMINAL_W", 3.4e9),
        P_turbine_W=0.0,
        rod_pos_pu=getattr(cfg, "ROD_INSERT_INIT", 0.55),
        rho_reactivity_dk=0.0,
        load_demand_pu=1.0,
        rod_mode="auto",
        rod_cmd_manual_pu=0.55,
    )
