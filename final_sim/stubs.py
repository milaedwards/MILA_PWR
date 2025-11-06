# final_sim/stubs.py
from dataclasses import dataclass
from typing import Optional, Tuple

# --- Reactor ---
@dataclass
class ReactorCore:
    Tc: float
    P: float
    rod: float
    mode: str = "auto"

    def __init__(self, Tc_init: float, P_turb_init: float, control_mode: str = "auto"):
        self.Tc = Tc_init
        self.P = 3.4e9 * max(0.2, P_turb_init)
        self.rod = 0.55
        self.mode = control_mode

    def step(self, Tc_in: float, dt: float, P_turb: float, manual_rod_cmd: Optional[float]) -> Tuple[float, float, float, float]:
        self.Tc = 0.98*self.Tc + 0.02*Tc_in
        if manual_rod_cmd is not None:
            self.rod = float(max(0.0, min(1.0, manual_rod_cmd)))
        P_target = 3.4e9 * max(0.2, P_turb) * (1.0 - 0.3*(self.rod-0.55))
        self.P += (P_target - self.P) * min(1.0, dt/4.0)  # ~4 s tau
        cp = 5200.0; mdot = 1.0e4
        Th = self.Tc + self.P/(mdot*cp)
        rho_dk = 0.0
        return Th, self.P, self.rod, rho_dk

# --- Steam Generator ---
class SteamGenerator:
    def step(self, T_hot_K: float, P_core_W: float, m_dot_primary_kg_s: float, cp_J_per_kgK: float, dt: float):
        if m_dot_primary_kg_s <= 1e-9:
            return T_hot_K, 0.0, 6.0e6
        T_cold = T_hot_K - P_core_W/(m_dot_primary_kg_s*cp_J_per_kgK)
        m_dot_steam = 0.9*m_dot_primary_kg_s
        P_sec = 6.0e6
        return T_cold, m_dot_steam, P_sec

# --- Pressurizer (simple PI + 1st-order plant) ---
class Pressurizer:
    def __init__(self, P_set_Pa=15.5e6, KP=1e-7, KI=1e-9, deadband_Pa=2e5, tau_P_s=3.0):
        self.P_set = P_set_Pa; self.KP = KP; self.KI = KI; self.db = deadband_Pa; self.tau = max(1e-6, tau_P_s)
        self.I = 0.0; self.P = P_set_Pa; self.level = 0.0
    def step(self, dt, P_primary_Pa, T_hot_K, T_spray_K):
        e = self.P_set - P_primary_Pa
        if abs(e) < self.db: e = 0.0
        self.I += self.KI*e*dt
        u = self.KP*e + self.I
        heater = max(0.0, min(1.0, u))
        spray = max(0.0, min(1.0, -u))
        dP_target = heater*1e-6 - spray*2e5
        self.P += (dP_target - (self.P - P_primary_Pa)/self.tau) * dt
        return float(self.P), float(self.level), float(heater), float(spray)

# --- Turbine ---
class TurbineCondenser:
    def __init__(self, eta_e=0.328, P_back=6.0e6):
        self.eta = eta_e; self.P_back = P_back; self.P = 0.0
    def step(self, m_dot_steam_kg_s, T_steam_K, P_inlet_Pa, P_back_Pa, load_cmd_pu, dt):
        # final_sim.ic_system will compute power via the turbine; stub returns backpressure unchanged
        return self.P, self.P_back
