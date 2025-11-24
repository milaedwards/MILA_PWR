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
        # Ensure a reactor instance is provided before we try to use it
        assert self.reactor is not None, "ICSystem.reactor must be constructed before ICSystem"

        # If no Config was explicitly passed to ICSystem, try to reuse the reactor's Config.
        # This keeps reactor, SG, and PlantState all on the same configuration instance.
        if self.cfg is None:
            reactor_cfg = getattr(self.reactor, "cfg", None)
            if reactor_cfg is not None:
                self.cfg = reactor_cfg
            else:
                self.cfg = Config()

        c = self.cfg
        ps0 = PlantState.init_from_config(c)

        # Per-loop steam flow (secondary side)
        n_loops = max(float(getattr(c, "N_LOOPS", 1.0)), 1.0)
        m_dot_loop = getattr(c, "M_DOT_SEC_KG_S", ps0.m_dot_steam_kg_s) / n_loops

        # Use the plant-level nominal hot/cold temps from PlantState / Config
        T_hot = float(ps0.T_hot_K)  # 596.483 K
        T_cold = float(ps0.T_cold_K)  # 553.817 K

        # Primary fluid lumps:
        #   T_p1 near the hot end (core outlet side),
        #   T_p2 near the cold end (return to core / cold leg).
        T_p1 = T_hot
        T_p2 = T_cold

        # Steam dome temperature reference for metal nodes
        T_steam = float(getattr(c, "T_SAT_SEC_K", ps0.T_steam_K))

        # Metal nodes lag between primary fluid and the steam dome; start them
        # midway so there is an initial temperature gradient for heat flow.
        T_m1 = 0.5 * (T_p1 + T_steam)
        T_m2 = 0.5 * (T_p2 + T_steam)

        # Steam generator internal state (per loop)
        self.sg_state = {
            "T_rxu":  T_hot,
            "T_hot":  T_hot,
            "T_sgi":  T_hot,
            "T_p1":   T_p1,
            "T_p2":   T_p2,
            "T_sgu":  T_cold,
            "T_cold": T_cold,
            "T_rxi":  T_cold,
            "T_m1":   T_m1,
            "T_m2":   T_m2,
            "p_s":    getattr(c, "P_SEC_INIT_PA", ps0.P_secondary_Pa),
            "T_s":    T_steam,
            "M_dot_stm": m_dot_loop,
        }

        self._manual_step_active = False
        self._manual_step_target_x = 0.0
        self._manual_step_dir = 0.0
        self._psec_int = 0.0

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
        #if ps.t_s < 5.0:  # only spam for the first few seconds
        #    T_avg_rc = diag.get(
        #        "T_avg",
        #        0.5 * (ps.T_hot_K + ps.T_cold_K)  # fallback if missing
        #    )
        #    print(
        #        "[RC_STATE] "
        #        f"t={ps.t_s:6.3f}s  "
        #        f"P_core={P_core:7.1f} MWt  "
        #        f"rho={rho_dk * 1e5:7.1f} pcm  "
        #        f"Tf={diag.get('Tf', float('nan')):7.2f} K  "
        #        f"Tc1={diag.get('Tc1', float('nan')):7.2f} K  "
        #        f"Tc2={diag.get('Tc2', float('nan')):7.2f} K  "
        #        f"T_in={diag.get('T_core_inlet', ps.T_cold_K):7.2f} K  "
        #        f"T_hot={diag.get('T_hot_leg', T_hot):7.2f} K  "
        #        f"T_avg={T_avg_rc:7.2f} K  "
        #        f"rod_pos={rod_pos:5.3f}"
        #    )
        # --------------------------------------------------------------

        # --- Enforce simple energy balance between core power and ΔT across core ---
        # P_core is in MW; convert to W.
        P_core_W = P_core * 1.0e6

        # Primary mass flow & cp
        m_dot_p = getattr(self.cfg, "M_DOT_PRI", ps.m_dot_primary_kg_s)
        cp_p = getattr(self.cfg, "CP_PRI", 5458.0)

        N_loops = max(float(getattr(self.cfg, "N_LOOPS", 1.0)), 1.0)
        Q_core_loop_W = P_core_W / N_loops

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

        # Per-loop scaling: each SG sees only its share of core power and steam flow
        N_loops = max(float(getattr(self.cfg, "N_LOOPS", 1.0)), 1.0)
        sg_in.update({
            # Interface with reactor / primary loop
            "T_rxu": ps.T_hot_K,
            # If you want SG to own its own T_hot/T_cold, do not override them here.
            # "T_hot": ps.T_hot_K,
            # "T_cold": ps.T_cold_K,
            "Q_core_W": Q_core_loop_W,            # per-loop power

            # Interface with secondary / turbine
            "p_s":       ps.P_secondary_Pa,
            "T_s":       ps.T_steam_K,
            "M_dot_stm": ps.m_dot_steam_kg_s / N_loops,     # per-loop steam flow

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
                                                     ps.m_dot_steam_kg_s)) * N_loops,
        )

        # === DEBUG: temperature snapshot for early-time behavior ===
        if ps.t_s <= 50.0:  # only print first 50 s to avoid spamming
            try:
                rdiag = self.reactor.get_diagnostics()
            except AttributeError:
                rdiag = {}

            # Steam generator temperatures (primary + metal + steam)
            print(
                f"[DBG_SG]   t={ps.t_s:5.2f}s  "
                f"T_sg_in={getattr(ps, 'T_sg_primary_in_K', float('nan')):7.2f}K  "
                f"T_sg_out={getattr(ps, 'T_sg_primary_out_K', float('nan')):7.2f}K  "
                f"T_m1={getattr(ps, 'T_sg_m1_K', float('nan')):7.2f}K  "
                f"T_m2={getattr(ps, 'T_sg_m2_K', float('nan')):7.2f}K  "
                f"T_mavg={getattr(ps, 'T_sg_metal_K', float('nan')):7.2f}K  "
                f"T_steam={getattr(ps, 'T_sg_steam_K', float('nan')):7.2f}K"
            )

            # Reactor-core / RCS temperatures
            print(
                f"[DBG_RC]   t={ps.t_s:5.2f}s  "
                f"T_hot={ps.T_hot_K:7.2f}K  "
                f"T_cold={ps.T_cold_K:7.2f}K  "
                f"Tavg={ps.Tavg_K:7.2f}K  "
                f"RC_Tin={rdiag.get('T_in', float('nan')):7.2f}K  "
                f"RC_Thot={rdiag.get('T_hot', float('nan')):7.2f}K  "
                f"RC_Tc1={rdiag.get('Tc1', float('nan')):7.2f}K  "
                f"RC_Tc2={rdiag.get('Tc2', float('nan')):7.2f}K"
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

        # --- Secondary pressure PI controller (acts like a main-steam valve) ---
        # Define error such that:
        #   p_err > 0  => P_secondary ABOVE nominal  => open valve (increase demand)
        #   p_err < 0  => P_secondary BELOW nominal  => close valve (decrease demand)
        P_sec_nom = getattr(self.cfg, "P_SEC_CONST_PA", 5.764e6)
        if P_sec_nom > 0.0:
            # Normalized pressure error
            p_err = (ps.P_secondary_Pa - P_sec_nom) / P_sec_nom

            # PI gains (dimensionless)
            Kp = float(getattr(self.cfg, "Kp_PSEC", 0.0))
            Ki = float(getattr(self.cfg, "Ki_PSEC", 0.0))

            # Integrator update with simple anti-windup clamp
            self._psec_int += p_err * Ki * dt
            self._psec_int = max(-0.2, min(0.2, self._psec_int))

            # Apply PI correction on top of operator demand
            P_dem_pu += Kp * p_err + self._psec_int

        # Clamp demand to a reasonable range
        P_dem_pu = max(0.0, min(1.2, P_dem_pu))

        # Base demand from operator (scale by the same rated turbine power)
        P_dem_MW = P_dem_pu * P_rated

        # Call the turbine model
        P_turb_MW, m_dot_steam_per_loop = self.turbine.step(
            inlet_t=ps.T_steam_K,
            inlet_p=ps.P_secondary_Pa,
            m_dot_steam=ps.m_dot_steam_kg_s,
            power_dem=P_dem_MW,
            dt=dt,
        )

        ps = replace(
            ps,
            P_turbine_MW=float(P_turb_MW),
            m_dot_steam_kg_s=float(m_dot_steam_per_loop),
        )

        #if ps.t_s < 5.0:  # only spam for the first few seconds
        #    print(f"[DBG] t={ps.t_s:.1f}s  T_hot={ps.T_hot_K:.2f} K  "
        #          f"T_cold={ps.T_cold_K:.2f} K")

        # === DEBUG: simple energy balance / power-flow snapshot ===
        # Print for early-time behavior only to avoid spamming the console.
        if ps.t_s <= 50.0:
            # Core thermal power (total), MWt
            Q_core_MWt = float(P_core)

            # Approximate heat carried to the secondary by steam flow,
            # based on the SG's stored (h_steam - cp * T_fw) term.
            H_ws_minus_cpTfw = float(getattr(self.steamgen, "H_ws_minus_cpTfw_J_kg", 0.0))
            if H_ws_minus_cpTfw > 0.0:
                Q_to_SG_MWt = ps.m_dot_steam_kg_s * H_ws_minus_cpTfw / 1.0e6
            else:
                Q_to_SG_MWt = 0.0

            print(
                f"[DBG_ENERGY] t={ps.t_s:7.2f}s  "
                f"P_core={Q_core_MWt:8.1f} MWt  "
                f"Q_core={Q_core_MWt:8.1f} MWt  "
                f"Q_to_SG={Q_to_SG_MWt:8.1f} MW  "
                f"m_dot_steam={ps.m_dot_steam_kg_s:8.1f} kg/s  "
                f"P_turb={ps.P_turbine_MW:8.1f} MWe"
            )

        return ps.clip_invariants()
