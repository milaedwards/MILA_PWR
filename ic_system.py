from dataclasses import dataclass, field, replace
from typing import Optional, Any

from config import Config
from plant_state import PlantState
from reactor_core import ReactorSimulator
from steam_generator import SteamGenerator
from turbine_condenser import TurbineModel
from pressurizer import PressurizerModel

@dataclass
class ICSystem:
    cfg: Config = field(default_factory=Config)
    reactor: ReactorSimulator | None = None
    steamgen: SteamGenerator | None = None
    turbine: TurbineModel | None = None
    pressurizer: PressurizerModel | None = None

    # Internal state for "move by delta x" manual rod steps
    _manual_step_active: bool = False
    _manual_step_target_x: float = 0.0
    _manual_step_dir: float = 0.0

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

    # Main step
    def step(self, ps: PlantState, dt: float) -> PlantState:
        assert self.reactor is not None
        assert self.steamgen is not None
        assert self.turbine is not None
        assert self.pressurizer is not None

        # manual rod speed command (fraction of stroke per second)
        manual_u: Optional[float] = None
        if ps.rod_mode == "manual":
            manual_u = float(ps.rod_cmd_manual_pu)  # use user input
        else:
            manual_u = None

        P_turb_pu_ref = ps.P_turbine_MW / self.cfg.P_e_nom_MWe

        # 1) Advance reactor core
        T_hot_K, P_core_MWt = self.reactor.step(
            Tc_in=ps.T_cold_K,
            dt=dt,
            P_turb = P_turb_pu_ref,
            control_mode=ps.rod_mode,
            manual_u=manual_u,
        )

        rod_pos = float(getattr(self.reactor, "x", ps.rod_pos_pu))  # rod position (0..1)
        rho = float(getattr(self.reactor, "rho_tot", 0.0))  # reactivity Δk/k

        ps = replace(
            ps,
            T_hot_K=float(T_hot_K),
            P_core_MW=float(P_core_MWt),
            rod_pos_pu=rod_pos,
            rho_reactivity_dk=rho,
        )

        # 2) Advance steam generator
        T_cold_K, m_dot_steam_actual_kg_s, P_sec_Pa, T_sec_K, sg_limited, h_steam = self.steamgen.step(
            ps.T_hot_K,
            ps.m_dot_steam_cmd_kg_s,
            dt,
        )

        ps = replace(
            ps,
            T_cold_K=T_cold_K,
            T_sg_in_K=ps.T_hot_K,
            T_sg_out_K=T_cold_K,
            T_metal_K=self.steamgen.T_metal_K,
            T_sec_K=T_sec_K,
            P_secondary_Pa=P_sec_Pa,
            m_dot_steam_kg_s=m_dot_steam_actual_kg_s,
            steam_h_J_kg=h_steam,
            sg_power_limited=bool(sg_limited),
        )

        P_pzr_Pa, pzr_level_m, pzr_heater, pzr_spray = self.pressurizer.step(
            ps.T_hot_K, ps.T_cold_K, dt
        )

        # Extract additional pressurizer internals for GUI display
        pzr_heater_frac = float(getattr(self.pressurizer, "heater_frac", 0.0))
        pzr_heater_kW = pzr_heater_frac * self.pressurizer.HEATER_POWER / 1.0e3
        pzr_surge_dir = str(getattr(self.pressurizer, "surge_direction", "NEUTRAL"))
        pzr_p_setpoint = float(getattr(self.pressurizer, "P_SETPOINT", 15.50e6)) \
                       + float(getattr(self.pressurizer, "dP0", 0.0))

        ps = replace(
            ps,
            P_pzr_Pa=float(P_pzr_Pa),
            pzr_level_m=float(pzr_level_m),
            pzr_heater_pu=float(pzr_heater),
            pzr_spray_pu=float(pzr_spray),
            pzr_heater_frac=float(pzr_heater_frac),
            pzr_heater_kW=float(pzr_heater_kW),
            pzr_surge_direction=pzr_surge_dir,
            pzr_pressure_setpoint_Pa=float(pzr_p_setpoint),
        )

        # 4) Advance turbine generator
        P_dem_MW = ps.load_demand_pu * self.cfg.P_e_nom_MWe

        P_turb_MW, m_dot_steam_cmd_new = self.turbine.step(
            inlet_h=ps.steam_h_J_kg,
            inlet_p=ps.P_secondary_Pa,
            m_dot_steam=ps.m_dot_steam_kg_s,
            power_dem=P_dem_MW,
            dt=dt,
        )

        if sg_limited:
            # If SG is limiting, clamp turbine command to actual SG flow
            m_dot_steam_cmd_new = ps.m_dot_steam_kg_s

        ps = replace(
            ps,
            P_turbine_MW=float(P_turb_MW),
            m_dot_steam_cmd_kg_s=float(m_dot_steam_cmd_new),
        )

        return ps.clip_invariants()