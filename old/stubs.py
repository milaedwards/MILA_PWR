from typing import Optional, Tuple
from old_config import Config


class ReactorStub:
    """
    Minimal reactor stub with simple but stable dynamics:
      - Core power tracks turbine load and rod position with a first-order lag
      - Hot-leg temperature tracks a nominal setpoint scaled by power
      - No xenon/kinetics; returns only (T_hot, P_core)
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Nominal values from config, with safe fallbacks
        self.P_nom = float(getattr(cfg, "Q_CORE_NOMINAL_W", 3.4e9))
        self.T_hot_nom = float(getattr(cfg, "T_HOT_NOM_K", 600.0))
        self.rod_pos = float(getattr(cfg, "ROD_INSERT_INIT", 0.55))
        # Time constants [s] for smooth behavior
        self.tau_power = float(getattr(cfg, "TAU_CORE_POWER_S", 5.0))
        self.tau_temp = float(getattr(cfg, "TAU_CORE_TEMP_S", 10.0))
        # State variables
        self.P_core = self.P_nom
        self.T_hot = self.T_hot_nom

    def step(
        self,
        *,
        Tc_in: float,
        P_turb: float,
        rod_mode: str,
        manual_rod_cmd: Optional[float],
        dt: float,
    ) -> Tuple[float, float]:
        """
        Matches ICSystem call:
            self.reactor.step(Tc_in=..., P_turb=..., rod_mode=..., manual_rod_cmd=..., dt=...)
        Returns:
            (T_hot_K, P_core_W)
        Notes:
            Tc_in is currently unused in the dynamic law; it's included only
            so the API matches the real model.
        """
        # --- Rod position update (very simple) ---
        if rod_mode == "manual" and manual_rod_cmd is not None:
            # Treat manual_rod_cmd as a delta (per unit / s) and integrate
            self.rod_pos += float(manual_rod_cmd) * dt
            self.rod_pos = max(0.0, min(1.0, self.rod_pos))

        # --- Target core power as function of load and rods ---
        pu_load = max(0.0, min(1.0, float(P_turb)))
        # Rod gain: small deviation around nominal insertion
        rod_gain = 1.0 - 0.3 * (self.rod_pos - 0.55)
        P_target = self.P_nom * pu_load * rod_gain

        # --- First-order power lag ---
        if self.tau_power > 1e-6:
            self.P_core += (P_target - self.P_core) * dt / self.tau_power
        else:
            self.P_core = P_target

        # --- Hot-leg temperature tracks power around a nominal value ---
        T_target = self.T_hot_nom * (self.P_core / max(1e-6, self.P_nom))
        if self.tau_temp > 1e-6:
            self.T_hot += (T_target - self.T_hot) * dt / self.tau_temp
        else:
            self.T_hot = T_target

        return float(self.T_hot), float(self.P_core)


class SGStub:
    """
    Minimal steam-generator stub with simple first-order behavior:
      - Cold-leg temperature tracks a fixed ΔT below hot-leg temperature
      - Steam flow tracks the primary mass flow with a lag
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Nominal values and time constant
        self.dT_nom = float(getattr(cfg, "SG_DELTA_T_K", 30.0))
        self.m_dot_nom = float(getattr(cfg, "M_DOT_PRI_KG_S", 1.0e4))
        self.tau = float(getattr(cfg, "TAU_SG_S", 10.0))
        # State variables
        self.T_cold = float(getattr(cfg, "T_COLD_NOM_K", 560.0))
        self.m_dot_steam = self.m_dot_nom

    def step(
        self,
        *,
        T_hot: float,
        m_dot_primary: float,
        dt: float,
    ) -> Tuple[float, float]:
        """
        Matches ICSystem call:
            self.steamgen.step(T_hot=..., m_dot_primary=..., dt=...)
        Returns:
            (T_cold_K, m_dot_steam_kg_s)
        """
        # Steam flow smoothly follows primary flow
        m_target = float(m_dot_primary)
        if self.tau > 1e-6:
            self.m_dot_steam += (m_target - self.m_dot_steam) * dt / self.tau
        else:
            self.m_dot_steam = m_target

        # Cold-leg temperature tracks a fixed ΔT below T_hot
        T_cold_target = float(T_hot) - self.dT_nom
        if self.tau > 1e-6:
            self.T_cold += (T_cold_target - self.T_cold) * dt / self.tau
        else:
            self.T_cold = T_cold_target

        return float(self.T_cold), float(self.m_dot_steam)


class PressurizerStub:
    """
    Minimal pressurizer stub:
      - Holds primary pressure at setpoint (no dynamics)
      - Returns zero level/commands (placeholders)
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.P = float(getattr(cfg, "P_PRI_INIT_PA", 15.5e6))
        self.P_set = float(getattr(cfg, "P_PRI_SET_Pa", self.P))

    def step(
        self,
        *,
        T_hot: float,
        T_cold: float,
        dt: float,
    ) -> float:
        """
        Matches ICSystem call:
            self.pressurizer.step(T_hot=..., T_cold=..., dt=...)
        Returns:
            P_primary_Pa
        Notes:
            Minimal stub: holds pressure at a fixed setpoint and ignores temperatures.
        """
        # In this stub we simply clamp primary pressure to the setpoint.
        self.P = float(self.P_set)
        return float(self.P)


class TurbineStub:
    """
    Minimal turbine/condenser stub:
      - Electrical power ~ load_cmd * eta * nominal core power
      - Returns inlet pressure as 'secondary out' pressure
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.P_e = 0.0

    def step(
        self,
        *,
        m_dot_steam: float,
        T_steam: float,
        P_secondary: float,
        load_demand: float,
        dt: float,
    ):
        """
        Matches ICSystem call:
            self.turbine.step(
                m_dot_steam=...,
                T_steam=...,
                P_secondary=...,
                load_demand=...,
                dt=...
            )
        Returns:
            (P_turbine_W, P_secondary_out_Pa)
        Notes:
            Minimal stub: electrical power is proportional to load_demand, with
            no real dependence on steam flow or temperature. Secondary pressure
            is simply passed through.
        """
        eta = float(getattr(self.cfg, "ETA_ELEC_PU", 0.33))
        P_nom = float(getattr(self.cfg, "Q_CORE_NOMINAL_W", 3.4e9))
        self.P_e = float(load_demand) * eta * P_nom
        # For this stub, just pass P_secondary straight through as the outlet pressure.
        return float(self.P_e), float(P_secondary)
