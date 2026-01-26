from dataclasses import dataclass, field, replace
from typing import Optional, Any

from config import Config
from plant_state import PlantState
from reactor_core import ReactorSimulator
from steam_generator import SteamGenerator
from turbine_condenser import TurbineModel


@dataclass
class ICSystem:
    cfg: Config = field(default_factory=Config)
    reactor: ReactorSimulator | None = None
    steamgen: SteamGenerator | None = None
    turbine: TurbineModel | None = None

    # Internal state for "move by Δx" manual rod steps
    _manual_step_active: bool = False
    _manual_step_target_x: float = 0.0
    _manual_step_dir: float = 0.0

    def __post_init__(self) -> None:
        # Make sure we actually have a reactor
        assert self.reactor is not None, "ICSystem.reactor must be provided"

        # Reuse reactor cfg if ICSystem.cfg wasn't set explicitly
        if self.cfg is None:
            reactor_cfg = getattr(self.reactor, "cfg", None)
            self.cfg = reactor_cfg if reactor_cfg is not None else Config()

        # Build SG and turbine if they weren't passed in
        if self.steamgen is None:
            self.steamgen = SteamGenerator(self.cfg)
        if self.turbine is None:
            self.turbine = TurbineModel(self.cfg)

        # Manual-step fields already have dataclass defaults above,
        # just keep them.

    # ------------------------------------------------------------------
    # Helpers for manual "step by Δx" rod motion
    # ------------------------------------------------------------------
    def start_manual_step(self, current_x: float, delta_x: float) -> None:
        self._manual_step_active = True
        self._manual_step_target_x = float(current_x + delta_x)
        self._manual_step_dir = 1.0 if delta_x > 0.0 else -1.0

    def update_manual_step(self, current_x: float) -> Optional[float]:
        if not self._manual_step_active:
            return None

        # If we overshot the target, clamp and stop
        if (self._manual_step_dir > 0 and current_x >= self._manual_step_target_x) or (
            self._manual_step_dir < 0 and current_x <= self._manual_step_target_x
        ):
            self._manual_step_active = False
            return self._manual_step_target_x

        # Otherwise, keep commanding movement in the chosen direction
        return self._manual_step_dir

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, ps: PlantState, dt: float) -> PlantState:
        """
        Advance the tightly-coupled reactor + SG + turbine system by one time-step.
        """
        assert self.reactor is not None
        assert self.steamgen is not None
        assert self.turbine is not None

        # 1) Advance reactor core
        manual_u: Optional[float] = None
        # 1) Advance reactor core
        if ps.rod_mode == "manual":
            manual_u = float(ps.rod_cmd_manual_pu)  # <-- use user input directly
        else:
            manual_u = 0.0  # keep a float so reactor_core never sees None

        T_hot_K, P_core_MWt = self.reactor.step(
            Tc_in=ps.T_cold_K,
            dt=dt,
            P_turb=ps.load_demand_pu,
            control_mode=ps.rod_mode,
            manual_u=manual_u,
        )

        # Grab total reactivity from the reactor, if exposed
        rho = getattr(self.reactor, "rho_total", getattr(self.reactor, "rho_tot", 0.0))

        # Update plant state with new core outlet temperature and power
        ps = replace(
            ps,
            T_hot_K=float(T_hot_K),
            P_core_MW=float(P_core_MWt),
            rho_reactivity_dk=float(rho),
            rod_pos_pu=float(getattr(self.reactor, "x", ps.rod_pos_pu)),
        )

        # 2) Advance steam generator
        Q_core_W = ps.P_core_MW * 1e6
        T_cold_K, m_dot_steam_actual_kg_s, P_sec_Pa, T_sec_K, sg_limited, h_steam = self.steamgen.step(
            ps.T_hot_K,
            Q_core_W,
            ps.m_dot_steam_cmd_kg_s,
            dt,
        )

        if ps.t_s <= dt:
            print(
                f"[IC t={ps.t_s:6.3f}] SG: "
                f"cmd={ps.m_dot_steam_cmd_kg_s:8.2f} -> act={m_dot_steam_actual_kg_s:8.2f} kg/s | "
                f"Psec={P_sec_Pa / 1e6:6.3f} MPa | "
                f"h={h_steam / 1e6:6.3f} MJ/kg | "
                f"limited={sg_limited}"
            )

        if ps.t_s <= 1.0:
            caps = getattr(self.steamgen, "_dbg_caps", None)
            if caps:
                print(f"[IC t={ps.t_s:6.3f}] SG caps: "
                      f"heat_cap={caps['heat_cap']:.2f} "
                      f"pressure_cap={caps['pressure_cap']:.2f} "
                      f"cmd={caps['cmd']:.2f} "
                      f"q_ms={caps['q_ms_MW']:.2f}MW "
                      f"dh={caps['delta_h_MJkg']:.3f}MJ/kg")

        ps = replace(
            ps,
            # primary loop
            T_cold_K=T_cold_K,
            T_sg_in_K=ps.T_hot_K,
            T_sg_out_K=T_cold_K,
            T_metal_K=self.steamgen.T_metal_K,
            T_sec_K=T_sec_K,
            P_secondary_Pa=P_sec_Pa,
            # Store the actual steam flow delivered by the SG
            m_dot_steam_kg_s=m_dot_steam_actual_kg_s,
            steam_h_J_kg=h_steam,
            sg_power_limited=bool(sg_limited),
        )

        # 3) (Optional) Pressurizer / RCS pressure dynamics would go here
        # (currently not modeled – primary pressure held at nominal via cfg)

        # 4) Advance turbine generator
        P_dem_MW = ps.load_demand_pu * self.cfg.P_e_nom_MWe

        # Use the actual steam flow from the SG as the turbine inlet flow
        P_turb_MW, m_dot_steam_cmd_new = self.turbine.step(
            inlet_h=ps.steam_h_J_kg,
            inlet_p=ps.P_secondary_Pa,
            #m_dot_steam=ps.m_dot_steam_kg_s,
            m_dot_cmd=ps.m_dot_steam_cmd_kg_s,
            m_dot_actual=ps.m_dot_steam_kg_s,
            power_dem=P_dem_MW,
            dt=dt,
        )

        if ps.t_s <= dt:
            print(
                f"[IC t={ps.t_s:6.3f}] TB: "
                f"Pturb={P_turb_MW:8.2f} MWe | "
                f"cmd_prev={ps.m_dot_steam_cmd_kg_s:8.2f} -> cmd_next={m_dot_steam_cmd_new:8.2f} | "
                f"m_act={ps.m_dot_steam_kg_s:8.2f} | "
                f"Psec_in={ps.P_secondary_Pa / 1e6:6.3f} MPa"
            )

        #if sg_limited:
            # If SG is limiting, clamp turbine command to actual SG flow
        #    m_dot_steam_cmd_new = ps.m_dot_steam_kg_s

        # don't force cmd down when SG is limiting
        # just keep cmd bounded
        m_dot_steam_cmd_new = max(0.0, min(m_dot_steam_cmd_new, 1.25 * self.cfg.m_dot_steam_nom_kg_s))

        # --- DEBUG: demand vs measured turbine vs steam cmd ---
        # Initialize one-time helper
        if not hasattr(self, "_dbg_next_print_s"):
            self._dbg_next_print_s = 0.0

        # Print for first 5 seconds, then every 10 seconds
        """
        if ps.t_s < 5.0 or ps.t_s >= self._dbg_next_print_s:
            power_dem_MWe = ps.load_demand_pu * self.cfg.P_e_nom_MWe  # demand in MWe

            print(
                f"[t={ps.t_s:6.2f}s] "
                f"power_dem={power_dem_MWe:8.2f} MWe | "
                f"P_turb={P_turb_MW:8.2f} MWe | "
                f"m_dot_cmd={m_dot_steam_cmd_new:8.2f} kg/s"
            )

            if ps.t_s >= self._dbg_next_print_s:
                self._dbg_next_print_s += 10.0
        """
        # -------------------------------------------------------

        ps = replace(
            ps,
            P_turbine_MW=float(P_turb_MW),
            # Store the newly commanded flow for use on the next SG step
            m_dot_steam_cmd_kg_s=float(m_dot_steam_cmd_new),
        )

        return ps.clip_invariants()
