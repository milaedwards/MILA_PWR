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
            "T_rxu":  c.T_HOT_INIT_K,
            "T_hot":  c.T_HOT_INIT_K,
            "T_sgi":  c.T_HOT_INIT_K,
            "T_p1":   c.T_HOT_INIT_K,
            "T_p2":   c.T_COLD_INIT_K,
            "T_m1":   c.T_HOT_INIT_K,
            "T_m2":   c.T_COLD_INIT_K,
            "T_sgu":  c.T_COLD_INIT_K,
            "T_cold": c.T_COLD_INIT_K,
            "T_rxi":  c.T_COLD_INIT_K,

            # Secondary / steam side
            "p_s":       c.P_SEC_INIT_PA,
            "T_s":       c.T_SAT_SEC_K,
            "M_dot_stm": c.M_DOT_SEC_KG_S,
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

        # Use actual turbine power (per-unit) as the input to the reactor
        # power program, rather than the operator load demand.
        #P_rated_MWe = getattr(self.cfg, "P_RATED_MWe", 1000.0)
        #if P_rated_MWe > 0.0:
        #    P_turb_pu = float(ps.load_demand_pu)
        #else:
        #    P_turb_pu = 1.0

        # Drive the reactor against the operator's load demand (per-unit)
        # rather than the currently produced turbine power. Using the demand
        # avoids a self-reinforcing loop where falling turbine power lowers
        # the temperature setpoint, causing the rods to insert further instead
        # of restoring power.
        #P_turb_pu = float(ps.load_demand_pu)

        # Drive the reactor temperature program primarily from the operator's
        # load demand, but do not let the reference collapse below what the
        # turbine is actually producing. If turbine power temporarily sags,
        # falling all the way to the measured output would recreate the
        # runaway insertion loop; if the turbine briefly overshoots, allowing
        # the higher feedback prevents the controller from demanding less than
        # what the grid is already getting.
        #P_turb_feedback_pu = ps.P_turbine_MW / max(self.cfg.P_RATED_MWe, 1.0)
        #P_turb_pu = max(float(ps.load_demand_pu), float(P_turb_feedback_pu))

        # --- Load signal for reactor temperature program (per-unit) ---
        # Use the turbine's own nominal MWe rating for scaling if available,
        # otherwise fall back to the Config value.
        try:
            P_rated_raw = float(getattr(self.turbine, "P_nom_MWe"))
        except (AttributeError, TypeError, ValueError):
            P_rated_raw = float(getattr(self.cfg, "P_RATED_MWe", 1000.0))
        P_rated = max(P_rated_raw, 1.0)

        # Operator demand (what the grid is asking for)
        P_demand_pu = float(ps.load_demand_pu)

        # Feedback from actual turbine power (can be useful later)
        P_turb_feedback_pu = ps.P_turbine_MW / P_rated

        # Use demand, but don't let the reference fall below actual output
        P_turb_pu = max(P_demand_pu, P_turb_feedback_pu)

        T_hot, P_core = self.reactor.step(
            Tc_in=ps.T_cold_K,
            P_turb=P_turb_pu,
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
        rho_dk = float(
            getattr(self.reactor, "rho_dk",
                    getattr(ps, "rho_reactivity_dk", 0.0))
        )

        # ---------- DEBUG: reactor thermal / neutronic state ----------
        if ps.t_s < 5.0:  # only spam for the first few seconds
            T_avg_rc = diag.get(
                "T_avg",
                0.5 * (ps.T_hot_K + ps.T_cold_K)  # fallback if missing
            )
            print(
                "[RC_STATE] "
                f"t={ps.t_s:6.3f}s  "
                f"P_core={P_core:7.1f} MWt  "
                f"rho={rho_dk * 1e5:7.1f} pcm  "
                f"Tf={diag.get('Tf', float('nan')):7.2f} K  "
                f"Tc1={diag.get('Tc1', float('nan')):7.2f} K  "
                f"Tc2={diag.get('Tc2', float('nan')):7.2f} K  "
                f"T_in={diag.get('T_core_inlet', ps.T_cold_K):7.2f} K  "
                f"T_hot={diag.get('T_hot_leg', T_hot):7.2f} K  "
                f"T_avg={T_avg_rc:7.2f} K  "
                f"rod_pos={rod_pos:5.3f}"
            )
        # --------------------------------------------------------------

        # --- Enforce simple energy balance between core power and ΔT across core ---
        # P_core is in MW; convert to W.
        P_core_W = P_core * 1.0e6

        # Primary mass flow & cp
        m_dot_p = getattr(self.cfg, "M_DOT_PRI", ps.m_dot_primary_kg_s)
        cp_p = getattr(self.cfg, "CP_PRI", 5458.0)

        N_loops = float(getattr(self.cfg, "N_LOOPS", 2))
        Q_core_loop_W = P_core_W / max(N_loops, 1.0)

        ps = replace(
            ps,
            T_hot_K=T_hot,  # <- use the reactor's own outlet temp
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
            "Q_core_W": Q_core_loop_W,

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

        self.sg_state = sg_state_new

        # --- Map SG outputs back into PlantState ---

        # Reactor inlet / primary cold leg temperature from SG
        # Prefer T_rxi (reactor inlet) if present; fall back to T_cold.
        T_rxi = float(self.sg_state.get("T_rxi", ps.T_cold_K))
        T_cold_loop = float(self.sg_state.get("T_cold", T_rxi))

        # Metal node temps for convenience
        T_m1 = float(self.sg_state.get("T_m1", ps.T_sg_m1_K))
        T_m2 = float(self.sg_state.get("T_m2", ps.T_sg_m2_K))

        ps = replace(
            ps,

            # ---- PRIMARY SIDE ----
            # What the reactor actually sees as inlet temperature
            T_cold_K=T_rxi,

            # Summary SG primary temps for plotting
            T_sg_primary_in_K=float(self.sg_state.get("T_hot", ps.T_sg_primary_in_K)),
            T_sg_primary_out_K=T_cold_loop,
            T_sg_metal_K=0.5 * (T_m1 + T_m2),

            # Detailed SG primary temps
            T_sg_hot_K=float(self.sg_state.get("T_hot", ps.T_sg_hot_K)),
            T_sg_sgi_K=float(self.sg_state.get("T_sgi", ps.T_sg_sgi_K)),
            T_sg_p1_K=float(self.sg_state.get("T_p1", ps.T_sg_p1_K)),
            T_sg_p2_K=float(self.sg_state.get("T_p2", ps.T_sg_p2_K)),
            T_sg_sgu_K=float(self.sg_state.get("T_sgu", ps.T_sg_sgu_K)),
            T_sg_cold_K=T_cold_loop,
            T_sg_m1_K=T_m1,
            T_sg_m2_K=T_m2,

            # Secondary / steam dome temperature
            T_sg_steam_K=float(self.sg_state.get("T_s", ps.T_sg_steam_K)),

            # ---- SECONDARY SIDE (already mostly correct) ----
            P_secondary_Pa=float(self.sg_state.get("p_s", ps.P_secondary_Pa)),
            T_steam_K=float(self.sg_state.get("T_s", ps.T_steam_K)),
            m_dot_steam_kg_s=float(self.sg_state.get("M_dot_stm",
                                                     ps.m_dot_steam_kg_s)),
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

        # For now, do not force primary pressure to the nominal value each step.
        # This allows a future pressurizer model (or other dynamics) to own P_primary_Pa.
        # ps = replace(ps, P_primary_Pa=self.cfg.P_PRI_INIT_PA)

        # Turbine Condenser (all powers in MW)
        # Current turbine power in MW (PlantState stores MW)
        P_turb_supplied_MW = ps.P_turbine_MW

        # Base demand from the operator (per-unit)
        P_dem_pu = float(ps.load_demand_pu)

        # --- Simple secondary pressure controller ---
        # If P_secondary > P_SEC_CONST: increase turbine demand (open valves)
        # If P_secondary < P_SEC_CONST: decrease turbine demand (close valves)
        P_sec_nom = getattr(self.cfg, "P_SEC_CONST_PA", 5.764e6)
        if P_sec_nom > 0.0:
            # error > 0 means pressure is BELOW nominal
            p_err = (P_sec_nom - ps.P_secondary_Pa) / P_sec_nom

            # Proportional correction on demand
            Kp = getattr(self.cfg, "Kp_PSEC", 0.0)
            P_dem_pu -= Kp * p_err

        # Clamp demand to a reasonable range
        P_dem_pu = max(0.0, min(1.2, P_dem_pu))

        # Base demand from operator (scale by the same rated turbine power)
        P_dem_MW = P_dem_pu * P_rated

        # Call the turbine model
        P_turb_MW, m_dot_steam = self.turbine.step(
            inlet_t=ps.T_steam_K,
            inlet_p=ps.P_secondary_Pa,
            m_dot_steam=ps.m_dot_steam_kg_s,
            power_dem=P_dem_MW,
            dt=dt,
        )

        ps = replace(
            ps,
            P_turbine_MW=float(P_turb_MW),
            m_dot_steam_kg_s=float(m_dot_steam),
        )

        if ps.t_s < 5.0:  # only spam for the first few seconds
            print(f"[DBG] t={ps.t_s:.1f}s  T_hot={ps.T_hot_K:.2f} K  "
                  f"T_cold={ps.T_cold_K:.2f} K")

        return ps.clip_invariants()