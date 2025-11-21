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

        # Internal state for manual rod "step" commands
        # (user says: insert/withdraw by X%, ICSystem converts to a move-to-target).
        self._manual_step_active = False
        self._manual_step_target_x = 0.0
        self._manual_step_dir = 0.0

    def step(self, ps: PlantState) -> PlantState:
        cfg = self.cfg
        dt = getattr(cfg, "dt", 0.1)

        assert self.reactor is not None, "ICSystem.reactor is required"
        # assert self.pressurizer is not None, "ICSystem.pressurizer is required"
        assert self.steamgen is not None, "ICSystem.steamgen is required"
        assert self.turbine is not None, "ICSystem.turbine is required"

        # Reactor Core
        # Interpret ps.rod_cmd_manual_pu as a one-shot "insert / withdraw by X fraction"
        # (e.g. +0.05 = insert 5%, -0.05 = withdraw 5%).
        manual_u: Optional[float]

        if ps.rod_mode == "manual":
            dx_cmd = getattr(ps, "rod_cmd_manual_pu", 0.0) or 0.0

            # Start a new manual step if we have a non-zero command
            # and are not already executing a previous step.
            if (not self._manual_step_active) and abs(dx_cmd) > 1e-6:
                # Current rod position in fraction of stroke [0, 1]
                x_curr = getattr(self.reactor, "x", None)
                if x_curr is not None:
                    # Target position from "insert/withdraw by X%"
                    x_target = x_curr + dx_cmd
                    # Clip to physical limits
                    x_target = max(0.0, min(1.0, x_target))

                    if abs(x_target - x_curr) > 1e-6:
                        direction = 1.0 if x_target > x_curr else -1.0
                        self._manual_step_active = True
                        self._manual_step_target_x = x_target
                        self._manual_step_dir = direction
                else:
                    # Fallback: if we cannot read reactor.x, at least move one step
                    self._manual_step_active = True
                    self._manual_step_target_x = None
                    self._manual_step_dir = 1.0 if dx_cmd > 0.0 else -1.0

            # If a manual step is active, drive rods toward the target at max speed
            if self._manual_step_active:
                x_curr = getattr(self.reactor, "x", None)
                x_rate_max = getattr(getattr(self.reactor, "ctrl", None), "x_rate_max", 0.0)

                if self._manual_step_target_x is not None and x_curr is not None:
                    # Check if we've reached (or slightly passed) the target
                    reached = (
                        (self._manual_step_dir > 0 and x_curr >= self._manual_step_target_x - 1e-4) or
                        (self._manual_step_dir < 0 and x_curr <= self._manual_step_target_x + 1e-4)
                    )
                    if reached or x_rate_max <= 0.0:
                        manual_u = 0.0
                        self._manual_step_active = False
                        self._manual_step_target_x = 0.0
                        self._manual_step_dir = 0.0
                    else:
                        manual_u = self._manual_step_dir * x_rate_max
                else:
                    # No position feedback: move once at max speed, then stop
                    if x_rate_max <= 0.0:
                        manual_u = 0.0
                    else:
                        manual_u = self._manual_step_dir * x_rate_max
                    self._manual_step_active = False
                    self._manual_step_target_x = 0.0
                    self._manual_step_dir = 0.0
            else:
                manual_u = 0.0
        else:
            # Not in manual mode: cancel any pending manual step
            self._manual_step_active = False
            self._manual_step_target_x = 0.0
            self._manual_step_dir = 0.0
            manual_u = None

        T_hot, P_core = self.reactor.step(
            Tc_in=ps.T_cold_K,
            P_turb=ps.load_demand_pu,
            control_mode=ps.rod_mode,
            manual_u=manual_u,
            dt=dt,
        )

        # Pull rod position and total reactivity from the reactor for diagnostics
        try:
            diag = self.reactor.get_diagnostics()
        except AttributeError:
            diag = {}

        rod_pos = float(diag.get("x_rods", getattr(ps, "rod_pos_pu", 0.0)))
        rho_dk = float(getattr(self.reactor, "rho_dk", getattr(ps, "rho_reactivity_dk", 0.0)))

        # After consuming the one-shot manual rod command, clear it in PlantState
        ps = replace(
            ps,
            T_hot_K=T_hot,
            P_core_MW=P_core,
            rod_cmd_manual_pu=0.0,
            rod_pos_pu=rod_pos,
            rho_reactivity_dk=rho_dk,
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

        ps = replace(ps, P_primary_Pa=self.cfg.P_PRI_INIT_PA)

        # Turbine Condenser (all powers in MW)
        # Current turbine power in MW (PlantState stores MW)
        P_turb_supplied_MW = ps.P_turbine_MW

        # Demanded electrical power in MW from per-unit load demand
        P_dem_MW = ps.load_demand_pu * self.cfg.P_RATED_MWe

        P_turb_MW, m_dot_steam = self.turbine.step(
            inlet_t=ps.T_steam_K,
            inlet_p=ps.P_secondary_Pa,
            m_dot_steam=ps.m_dot_steam_kg_s,
            power_dem=P_dem_MW,
            dt=dt,
        )

        ps = replace(
            ps,
            P_turbine_MW=float(P_turb_MW),  # still stored in MW
            m_dot_steam_kg_s=float(m_dot_steam),
        )

        return ps.clip_invariants()
