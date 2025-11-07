from typing import Optional, Tuple
from config import Config


class ReactorStub:
    """
    Minimal reactor stub:
      - Tracks a simple first-order power response to load and rod position
      - Computes hot-leg outlet temperature from energy balance
      - No xenon/kinetics; returns rho=0
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.P = float(getattr(cfg, "Q_CORE_NOMINAL_W", 3.4e9))
        self.rod_pos = float(getattr(cfg, "ROD_INSERT_INIT", 0.55))

    def step(
        self,
        Tc_in: float,
        dt: float,
        P_turb: float,
        manual_rod_cmd: Optional[float],
    ) -> Tuple[float, float, float, float]:
        # Update rod if manual command present
        if manual_rod_cmd is not None:
            self.rod_pos = max(0.0, min(1.0, float(manual_rod_cmd)))

        # Target core power ~ nominal * load * rod scaling
        P_nom = float(getattr(self.cfg, "Q_CORE_NOMINAL_W", 3.4e9))
        rod_gain = 1.0 - 0.3 * (self.rod_pos - 0.55)
        target = P_nom * max(0.0, min(1.0, float(P_turb))) * rod_gain

        # First-order power lag (very light dynamics)
        tau = 4.0  # s
        a = max(0.0, min(1.0, dt / tau))
        self.P += (target - self.P) * a

        # Hot-leg temperature from energy balance
        cp = float(getattr(self.cfg, "CP_PRI_J_PER_KG_K", 5200.0))
        mdot = float(getattr(self.cfg, "M_DOT_PRI", 1.0e4))
        if mdot > 1e-12:
            Th_out = float(Tc_in + self.P / (mdot * cp))
        else:
            Th_out = float(Tc_in)

        rho_dk = 0.0  # no reactivity model in stub
        return float(Th_out), float(self.P), float(self.rod_pos), float(rho_dk)


class SGStub:
    """
    Minimal steam-generator stub:
      - Computes cold-leg temperature by removing core power from primary flow
      - Returns a constant secondary pressure and nominal steam flow
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def step(
        self,
        T_hot_K: float,
        P_core_W: float,
        m_dot_primary_kg_s: float,
        cp_J_per_kgK: float,
        dt: float,
    ):
        if m_dot_primary_kg_s > 1e-12:
            T_cold = float(T_hot_K - P_core_W / (m_dot_primary_kg_s * cp_J_per_kgK))
        else:
            T_cold = float(T_hot_K)

        m_dot_steam = float(getattr(self.cfg, "M_DOT_SEC", 9.0e3))
        P_secondary = float(getattr(self.cfg, "P_SEC_INIT_PA", 6.0e6))
        return float(T_cold), float(m_dot_steam), float(P_secondary)


class PressurizerStub:
    """
    Minimal pressurizer stub:
      - Holds primary pressure at setpoint (no dynamics)
      - Returns zero level/commands (placeholders)
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.P = float(getattr(cfg, "P_PRI_INIT_PA", 15.5e6))
        self.P_set = float(getattr(cfg, "P_PRI_SET", self.P))

    def step(
        self,
        dt: float,
        P_primary_Pa: float,
        T_hot_K: float,
        T_spray_K: float,
    ):
        self.P = float(self.P_set)
        level = 0.0
        heater_cmd = 0.0
        spray_cmd = 0.0
        return float(self.P), float(level), float(heater_cmd), float(spray_cmd)


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
        m_dot_steam_kg_s: float,
        T_steam_K: float,
        P_inlet_Pa: float,
        P_back_Pa: float,
        load_cmd_pu: float,
        dt: float,
    ):
        eta = float(getattr(self.cfg, "ETA_ELEC_PU", 0.33))
        P_nom = float(getattr(self.cfg, "Q_CORE_NOMINAL_W", 3.4e9))
        self.P_e = float(load_cmd_pu) * eta * P_nom
        return float(self.P_e), float(P_inlet_Pa)
