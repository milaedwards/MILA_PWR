from typing import Protocol, Tuple, Optional

class ReactorCoreLike(Protocol):
    def step(self, Tc_in: float, dt: float, P_turb: float, manual_rod_cmd: Optional[float]) -> Tuple[float, float, float, float]:
        """
        Args:
            Tc_in: cold-leg temperature [K]
            dt: step size [s]
            P_turb: turbine load demand [pu]
            manual_rod_cmd: if not None, desired rod position [0..1 pu]
        Returns:
            Th_out_K, P_core_W, rod_pos_pu, rho_reactivity_dk
        """
        ...

class PressurizerLike(Protocol):
    def step(self, dt: float, P_primary_Pa: float, T_hot_K: float, T_spray_K: float) -> Tuple[float, float, float, float]:
        """
        Returns:
            pzr_pressure_Pa, pzr_level_m, heater_cmd_pu, spray_cmd_pu
        """
        ...

class SteamGeneratorLike(Protocol):
    def step(self, T_hot_K: float, P_core_W: float, m_dot_primary_kg_s: float, cp_J_per_kgK: float, dt: float) -> Tuple[float, float, float]:
        """
        Returns:
            T_cold_K, m_dot_steam_kg_s, P_secondary_Pa
        """
        ...

class TurbineCondenserLike(Protocol):
    def step(self, m_dot_steam_kg_s: float, T_steam_K: float, P_inlet_Pa: float, P_back_Pa: float, load_cmd_pu: float, dt: float) -> Tuple[float, float]:
        """
        Returns:
            P_turbine_W, P_secondary_Pa (updated/backpressure proxy)
        """
        ...

