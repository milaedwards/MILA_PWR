from dataclasses import dataclass, replace
from typing import Optional, Any
from config import Config
from plant_state import PlantState

@dataclass
class ICSystem:
    reactor: Any
    # pressurizer: Any
    steamgen: Any
    turbine: Any
    cfg: Optional[Config] = None

    def __post_init__(self):
        if self.cfg is None:
            self.cfg = Config()

        c = self.cfg

        # Internal Steam Generator state for dict-based SG API
        # NOTE: adjust the Config attribute names (T_HOT_INIT, etc.)
        # to match your actual Config class.
        self.sg_state = {
            # Primary-side nodes
            "T_rxu":  c.T_HOT_INIT,
            "T_hot":  c.T_HOT_INIT,
            "T_sgi":  c.T_HOT_INIT,
            "T_p1":   c.T_HOT_INIT,
            "T_p2":   c.T_COLD_INIT,
            "T_m1":   c.T_HOT_INIT,
            "T_m2":   c.T_COLD_INIT,
            "T_sgu":  c.T_COLD_INIT,
            "T_cold": c.T_COLD_INIT,
            "T_rxi":  c.T_COLD_INIT,

            # Secondary / steam side
            "p_s":       c.P_SEC_INIT,
            "T_s":       c.T_SAT_SEC,
            "M_dot_stm": c.M_DOT_SEC,
        }

    def step(self, ps: PlantState) -> PlantState:
        cfg = self.cfg
        dt = getattr(cfg, "dt", 0.1)

        assert self.reactor is not None, "ICSystem.reactor is required"
        # assert self.pressurizer is not None, "ICSystem.pressurizer is required"
        assert self.steamgen is not None, "ICSystem.steamgen is required"
        assert self.turbine is not None, "ICSystem.turbine is required"

        # Reactor Core
        manual_cmd = ps.rod_cmd_manual_pu if ps.rod_mode == "manual" else None
        T_hot, P_core = self.reactor.step(
            Tc_in=ps.T_cold_K,
            P_turb=ps.load_demand_pu,
            control_mode=ps.rod_mode,
            manual_u=manual_cmd,
            dt=dt,
        )

        ps = replace(
            ps,
            T_hot_K=T_hot,
            P_core_W=P_core,
        )

        # Steam Generator (dict-based API)
        # Build SG input dict from last SG state plus current PlantState interface values
        sg_in = self.sg_state.copy()
        sg_in.update({
            # Interface with reactor / primary loop
            "T_rxu": ps.T_hot_K,
            # If you want SG to own its own T_hot/T_cold, do not override them here.
            # "T_hot": ps.T_hot_K,
            # "T_cold": ps.T_cold_K,

            # Interface with secondary / turbine
            "p_s":       ps.P_secondary_Pa,
            "T_s":       ps.T_steam_K,
            "M_dot_stm": ps.m_dot_steam_kg_s,

            # Time step
            "dt": dt,
        })

        # Call the dict-based steam generator model
        sg_out = self.steamgen.step(sg_in)

        # Collapse *_new keys back into the internal SG state dict
        sg_state_new = self.sg_state.copy()
        for key, val in sg_out.items():
            if key.endswith("_new"):
                base = key[:-4]  # strip "_new"
                sg_state_new[base] = val
            else:
                sg_state_new[key] = val

        self.sg_state = sg_state_new

        # Push SG outputs back into PlantState
        ps = replace(
            ps,
            T_cold_K=self.sg_state["T_cold"],
            T_steam_K=self.sg_state["T_s"],
            P_secondary_Pa=self.sg_state["p_s"],
            m_dot_steam_kg_s=self.sg_state["M_dot_stm"],
        )

        # Pressurizer
        #P_pzr = float(self.pressurizer.step(
        #    T_hot=ps.T_hot_K,
        #    T_cold=ps.T_cold_K,
        #    dt=dt
        #))

        #ps = replace(
        #    ps,
        #    P_primary_Pa=P_pzr,
        #)

        ps = replace(ps, P_primary_Pa=self.cfg.P_PRI_INIT)

        # Turbine Condenser (all powers in MW)
        # Current turbine power in MW (PlantState stores MW)
        P_turb_supplied_MW = ps.P_turbine_W

        # Demanded electrical power in MW from per-unit load demand
        P_dem_MW = ps.load_demand_pu * self.cfg.P_RATED_MWe

        P_turb_MW, m_dot_steam = self.turbine.step(
            inlet_t=ps.T_steam_K,
            inlet_p=ps.P_secondary_Pa,
            m_dot_steam=ps.m_dot_steam_kg_s,
            power_supplied=P_turb_supplied_MW,
            power_dem=P_dem_MW,
            dt=dt,
        )

        ps = replace(
            ps,
            P_turbine_W=float(P_turb_MW),  # still stored in MW
            m_dot_steam_kg_s=float(m_dot_steam),
        )

        return ps.clip_invariants()
