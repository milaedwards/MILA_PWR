from dataclasses import dataclass, replace
from typing import Optional
from config import Config
from plant_state import PlantState
from final_sim.stubs import ReactorCore, Pressurizer, SteamGenerator, TurbineCondenser

@dataclass
class ICSystem:
    reactor: ReactorCore
    pressurizer: Pressurizer
    steamgen: SteamGenerator
    turbine: TurbineCondenser
    Config: object | None = None


def __post_init__(self):
    if self.cfg is None:
        self.cfg = Config()

def step(self, ps: PlantState) -> PlantState:

        cfg = self.cfg
        dt = getattr(cfg, "dt", 0.1)

        # Validate dependencies (explicit failures help catch wiring mistakes)
        assert self.reactor is not None, "ICSystem.reactor is required"
        assert self.pressurizer is not None, "ICSystem.pressurizer is required"
        assert self.steamgen is not None, "ICSystem.steamgen is required"
        assert self.turbine is not None, "ICSystem.turbine is required"

        # 1) Reactor
        manual_cmd = ps.rod_cmd_manual_pu if ps.rod_mode == "manual" else None
        Th_out, P_core, rod_pos, rho_dk = self.reactor.step(
            Tc_in=ps.T_cold_K, dt=dt, P_turb=ps.load_demand_pu, manual_rod_cmd=manual_cmd
        )
        ps = replace(ps, T_hot_K=float(Th_out), P_core_W=float(P_core),
                         rod_pos_pu=float(rod_pos), rho_reactivity_dk=float(rho_dk))

        # 2) Steam Generator
        cp = float(getattr(cfg, "CP_PRI_J_PER_KG_K", 5200.0))
        Tc_next, m_dot_steam, P_sec = self.steamgen.step(ps.T_hot_K, ps.P_core_W, ps.m_dot_primary_kg_s, cp, dt)
        ps = replace(ps, T_cold_K=float(Tc_next), m_dot_steam_kg_s=float(m_dot_steam), P_secondary_Pa=float(P_sec))

        # 3) Pressurizer
        P_pzr, L_pzr, heater_cmd, spray_cmd = self.pressurizer.step(
            dt, ps.P_primary_Pa, ps.T_hot_K, float(getattr(cfg, "T_SPRAY_K", 300.0))
        )
        ps = replace(ps, pzr_pressure_Pa=float(P_pzr), P_primary_Pa=float(P_pzr), pzr_level_m=float(L_pzr))

        # 4) Turbine/Condenser
        P_turb, P_sec2 = self.turbine.step(
            ps.m_dot_steam_kg_s,
            float(getattr(cfg, "T_SAT_SEC_K", ps.T_steam_K)),
            ps.P_secondary_Pa,
            float(getattr(cfg, "P_BACKPRESSURE_PA", ps.P_secondary_Pa)),
            ps.load_demand_pu,
            dt
        )
        ps = replace(ps, P_turbine_W=float(P_turb), P_secondary_Pa=float(P_sec2))

        return ps.clip_invariants()
