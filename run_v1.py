import argparse
import os
from typing import Optional, Tuple
import numpy as np

# Choose a Matplotlib backend before importing pyplot (PyCharm's helper backend can break on some versions)
import matplotlib
def _select_backend():
    # If running inside PyCharm, pick a known-good interactive backend; otherwise fall back to Agg
    if os.environ.get("PYCHARM_HOSTED") == "1":
        for candidate in ("MacOSX", "TkAgg", "Qt5Agg"):
            try:
                matplotlib.use(candidate, force=True)
                return
            except Exception:
                continue
    # As a safe default for headless or incompatible setups
    try:
        matplotlib.use("TkAgg", force=True)
    except Exception:
        matplotlib.use("Agg", force=True)

_select_backend()
import matplotlib.pyplot as plt

"""
Self-contained test harness for the final simulator orchestration.

Goals:
- Run ICSystem with simple stub subsystems (no heavy physics).
- Verify timing (dt), call order, and that commands/values are passed correctly.
- Optionally plot & CSV-log time series.
- More-physical mode: add light dynamics (taus, PI pressurizer, evap balance) for realistic plots.

Usage:
  # Default plots
  python run.py

  # Non-plot run with CSV
  python run.py --no-plots --csv sim.csv

  # Dial in smoother behavior
  python run.py \
    --steps 600 \
    --tau-reactor 8 --tau-th 5 \
    --tau-sg 8 --tau-evap 10 --hfg 1.8e6 --eta-evap 0.95 \
    --pzr-kp 2e-7 --pzr-ki 1e-9 --pzr-deadband 2e5 --pzr-tau 8 --pzr-gain 2e6 \
    --eta-e 0.33 --tau-turb 4 --tau-sec 8 --ksec 500
"""

from dataclasses import dataclass

from config import Config
from final_sim.plant_state import PlantState
from final_sim.ic_system import ICSystem


# ---------------------------
# More-physical stub subsystems (spies)
# ---------------------------

class ReactorStub:
    """
    Minimal reactor with:
      - Power first-order response (tau_reactor)
      - Outlet temperature first-order lag (tau_th)
      - Mild negative temperature feedback: rho_T = alpha_T * (Tavg - Tref)
      - Manual rod command adjusts target power via (1 - 0.3*(rod-0.55))
    """
    def __init__(self, cfg: Config, tau_reactor_s: float = None, tau_th_s: float = None, alpha_T_dk_per_K: float = None):
        self.cfg = cfg
        self.call_count = 0
        self.last_args = {}

        # State
        self.rod_pos = float(getattr(cfg, "ROD_INSERT_INIT", 0.55))
        self.P = float(getattr(cfg, "Q_CORE_NOMINAL_W", 3.4e9))
        self.Th = float(getattr(cfg, "T_HOT_INIT_K", 595.0))

        # Params
        self.tau_reactor = float(tau_reactor_s if tau_reactor_s is not None else getattr(cfg, "TAU_REACTOR_S", 6.0))
        self.tau_th = float(tau_th_s if tau_th_s is not None else getattr(cfg, "TAU_REACTOR_TH_S", 3.0))
        self.alpha_T = float(alpha_T_dk_per_K if alpha_T_dk_per_K is not None else getattr(cfg, "ALPHA_T_DK_PER_K", -1.0e-5))
        self.Tref = float(getattr(cfg, "TAVG_REF_BASE_K", 580.0))

    def step(self, Tc_in: float, dt: float, P_turb: float, manual_rod_cmd: Optional[float]) -> Tuple[float, float, float, float]:
        self.call_count += 1
        self.last_args = dict(Tc_in=Tc_in, dt=dt, P_turb=P_turb, manual_rod_cmd=manual_rod_cmd)

        # Manual rod (0..1)
        if manual_rod_cmd is not None:
            self.rod_pos = float(max(0.0, min(1.0, manual_rod_cmd)))

        # Target power with rod gain and small temperature feedback
        P_nom = float(getattr(self.cfg, "Q_CORE_NOMINAL_W", 3.4e9))
        rod_gain = (1.0 - 0.3 * (self.rod_pos - 0.55))
        Tavg = 0.5 * (self.Th + float(Tc_in))
        rho_T = self.alpha_T * (Tavg - self.Tref)     # dk/k
        temp_gain = max(0.0, 1.0 + rho_T)             # coarse effect

        target = P_nom * max(0.2, float(P_turb)) * rod_gain * temp_gain

        # Power first-order dynamics
        aP = max(0.0, min(1.0, dt / max(1e-6, self.tau_reactor)))
        self.P += (target - self.P) * aP

        # Outlet temperature with first-order lag toward energy-balance value
        cp = float(getattr(self.cfg, "CP_PRI_J_PER_KG_K", 5200.0))
        mdot = float(getattr(self.cfg, "M_DOT_PRI", 1.0e4))
        Th_eq = float(Tc_in + self.P / (mdot * cp))
        aT = max(0.0, min(1.0, dt / max(1e-6, self.tau_th)))
        self.Th += (Th_eq - self.Th) * aT

        return float(self.Th), float(self.P), float(self.rod_pos), float(rho_T)


class SGStub:
    """
    Minimal SG with:
      - Primary-side outlet T_cold from energy balance, filtered by tau_sg
      - Steam generation from core power: m_evap ≈ eta_evap * P_core / h_fg, limited by primary flow
      - First-order lag for steam generation (tau_evap)
    """
    def __init__(self, cfg: Config, tau_sg_s: float = None, tau_evap_s: float = None, h_fg_J_per_kg: float = None, eta_evap: float = None):
        self.cfg = cfg
        self.call_count = 0
        # diagnostic
        self.last_args = {}

        # State
        self.Tc = float(getattr(cfg, "T_COLD_INIT_K", 553.0))
        self.msteam = float(getattr(cfg, "M_DOT_SEC", 9.0e3))

        # Params
        self.tau_sg = float(tau_sg_s if tau_sg_s is not None else getattr(cfg, "TAU_SG_S", 6.0))
        self.tau_evap = float(tau_evap_s if tau_evap_s is not None else getattr(cfg, "TAU_SG_EVAP_S", 8.0))
        self.hfg = float(h_fg_J_per_kg if h_fg_J_per_kg is not None else getattr(cfg, "H_FG_SEC_J_PER_KG", 1.8e6))
        self.eta_evap = float(eta_evap if eta_evap is not None else getattr(cfg, "ETA_EVAP", 0.95))

    def step(self, T_hot_K: float, P_core_W: float, m_dot_primary_kg_s: float, cp_J_per_kgK: float, dt: float):
        self.call_count += 1
        self.last_args = dict(T_hot_K=T_hot_K, P_core_W=P_core_W,
                              m_dot_primary_kg_s=m_dot_primary_kg_s,
                              cp_J_per_kgK=cp_J_per_kgK, dt=dt)

        # Primary-side cold-leg temperature (first order toward energy balance)
        if m_dot_primary_kg_s <= 1e-12:
            T_cold_inst = float(T_hot_K)
        else:
            T_cold_inst = float(T_hot_K - P_core_W / (m_dot_primary_kg_s * cp_J_per_kgK))

        aTc = max(0.0, min(1.0, dt / max(1e-6, self.tau_sg)))
        self.Tc += (T_cold_inst - self.Tc) * aTc

        # Steam generation rate (limited and filtered)
        if self.hfg <= 0.0:
            m_evap_inst = 0.0
        else:
            m_evap_inst = self.eta_evap * float(P_core_W) / self.hfg
        m_evap_inst = max(0.0, min(float(m_dot_primary_kg_s), m_evap_inst))

        aMe = max(0.0, min(1.0, dt / max(1e-6, self.tau_evap)))
        self.msteam += (m_evap_inst - self.msteam) * aMe

        # Secondary pressure is left to the turbine/condenser stub (more natural)
        P_secondary = float(getattr(self.cfg, "P_SEC_INIT_PA", 6.0e6))

        return float(self.Tc), float(self.msteam), P_secondary


class PressurizerStub:
    """
    Pressurizer PI with:
      - Deadband on error
      - Anti-windup clamped integrator
      - First-order pressure plant: P -> P_target = P_meas + Kplant*(heater - spray)
    """
    def __init__(self, cfg: Config, kp: float = None, ki: float = None, deadband_Pa: float = None, tau_pzr_s: float = None, Kplant_Pa: float = None):
        self.cfg = cfg
        self.call_count = 0
        self.last_args = {}

        self.P = float(getattr(cfg, "P_PRI_INIT_PA", 15.5e6))
        self.I = 0.0

        # Params
        self.kp = float(kp if kp is not None else getattr(cfg, "PZR_KP", 1e-7))
        self.ki = float(ki if ki is not None else getattr(cfg, "PZR_KI", 1e-9))
        self.db = float(deadband_Pa if deadband_Pa is not None else getattr(cfg, "PZR_DB_PA", 2e5))
        self.tau = float(tau_pzr_s if tau_pzr_s is not None else getattr(cfg, "PZR_TAU_S", 5.0))
        self.Kplant = float(Kplant_Pa if Kplant_Pa is not None else getattr(cfg, "PZR_KPLANT_PA", 1.5e6))

        self.P_set = float(getattr(cfg, "P_PRI_SET", getattr(cfg, "P_PRI_INIT_PA", 15.5e6)))

    def step(self, dt: float, P_primary_Pa: float, T_hot_K: float, T_spray_K: float):
        self.call_count += 1
        self.last_args = dict(dt=dt, P_primary_Pa=P_primary_Pa, T_hot_K=T_hot_K, T_spray_K=T_spray_K)

        e = self.P_set - float(P_primary_Pa)
        if abs(e) < self.db:
            e = 0.0

        # PI control
        self.I += self.ki * e * dt
        self.I = max(-1.0, min(1.0, self.I))    # anti-windup
        u = self.kp * e + self.I

        heater = max(0.0, min(1.0, u))
        spray = max(0.0, min(1.0, -u))

        # Plant: target P around measured P_primary_Pa
        P_target = float(P_primary_Pa) + self.Kplant * (heater - spray)
        aP = max(0.0, min(1.0, dt / max(1e-6, self.tau)))
        self.P += (P_target - self.P) * aP

        level = 0.0
        return float(self.P), float(level), float(heater), float(spray)


class TurbineStub:
    """
    Turbine/condenser with:
      - Electrical power ~ load_cmd * eta_e * P_core_nom, first-order lag (tau_turb)
      - Secondary pressure dynamics driven by steam flow mismatch
        P_sec -> P_sec_nom + Ksec * (m_gen - m_demand), filtered by tau_sec
    """
    def __init__(self, cfg: Config, eta_e: float = None, tau_turb_s: float = None, tau_sec_s: float = None, Ksec_Pa_per_kgs: float = None):
        self.cfg = cfg
        self.call_count = 0
        self.last_args = {}

        self.eta = float(eta_e if eta_e is not None else getattr(cfg, "ETA_ELEC_PU", 0.33))
        self.tau_turb = float(tau_turb_s if tau_turb_s is not None else getattr(cfg, "TAU_TURB_S", 3.0))
        self.tau_sec = float(tau_sec_s if tau_sec_s is not None else getattr(cfg, "TAU_SEC_PRESS_S", 5.0))
        self.Ksec = float(Ksec_Pa_per_kgs if Ksec_Pa_per_kgs is not None else getattr(cfg, "KSEC_PA_PER_KG_S", 300.0))

        self.P_e = 0.0
        self.P_sec = float(getattr(cfg, "P_SEC_INIT_PA", 6.0e6))

    def step(self, m_dot_steam_kg_s: float, T_steam_K: float, P_inlet_Pa: float, P_back_Pa: float, load_cmd_pu: float, dt: float):
        self.call_count += 1
        self.last_args = dict(m_dot_steam_kg_s=m_dot_steam_kg_s, T_steam_K=T_steam_K,
                              P_inlet_Pa=P_inlet_Pa, P_back_Pa=P_back_Pa,
                              load_cmd_pu=load_cmd_pu, dt=dt)

        # Power with first-order lag to demand
        P_nom = float(getattr(self.cfg, "Q_CORE_NOMINAL_W", 3.4e9))
        P_dem = float(load_cmd_pu) * self.eta * P_nom
        a = max(0.0, min(1.0, dt / max(1e-6, self.tau_turb)))
        self.P_e += (P_dem - self.P_e) * a

        # Secondary pressure driven by steam flow mismatch
        m_demand = float(load_cmd_pu) * float(getattr(self.cfg, "M_DOT_SEC", 9.0e3))
        P_target = float(getattr(self.cfg, "P_SEC_INIT_PA", 6.0e6)) + self.Ksec * (float(m_dot_steam_kg_s) - m_demand)
        aP = max(0.0, min(1.0, dt / max(1e-6, self.tau_sec)))
        self.P_sec += (P_target - self.P_sec) * aP

        return float(self.P_e), float(self.P_sec)


# ---------------------------
# Test harness
# ---------------------------

def simulate(n_steps: int = 120, plot: bool = True, csv_path: Optional[str] = None, args=None):
    cfg = Config()

    # Build initial state from Config-backed defaults
    ps = PlantState.init_default()

    # Instantiate stubs (pass-through CLI overrides if provided)
    reactor = ReactorStub(cfg, tau_reactor_s=getattr(args, "tau_reactor", None),
                          tau_th_s=getattr(args, "tau_th", None),
                          alpha_T_dk_per_K=getattr(args, "alpha_T", None))
    sg = SGStub(cfg, tau_sg_s=getattr(args, "tau_sg", None),
                tau_evap_s=getattr(args, "tau_evap", None),
                h_fg_J_per_kg=getattr(args, "hfg", None),
                eta_evap=getattr(args, "eta_evap", None))
    pzr = PressurizerStub(cfg, kp=getattr(args, "pzr_kp", None),
                          ki=getattr(args, "pzr_ki", None),
                          deadband_Pa=getattr(args, "pzr_deadband", None),
                          tau_pzr_s=getattr(args, "pzr_tau", None),
                          Kplant_Pa=getattr(args, "pzr_gain", None))
    turb = TurbineStub(cfg, eta_e=getattr(args, "eta_e", None),
                       tau_turb_s=getattr(args, "tau_turb", None),
                       tau_sec_s=getattr(args, "tau_sec", None),
                       Ksec_Pa_per_kgs=getattr(args, "ksec", None))

    # Orchestrator (final simulator shape)
    ic = ICSystem(reactor=reactor, pressurizer=pzr, steamgen=sg, turbine=turb, cfg=cfg)

    dt = float(getattr(cfg, "dt", 0.1))
    t_final = float(getattr(cfg, "t_final", n_steps * dt))
    steps = int(t_final / dt)

    # Time series for plotting/logging
    ts, Ths, Tcs, Tavgs = [], [], [], []
    Ppris, Psecs, Ppzrs = [], [], []
    Pcores, Pturbs = [], []
    rods, rhos = [], []
    loads, rod_cmds = [], []

    # Simple command profiles
    def load_profile(t):
        return 1.0 if t < t_final * 0.5 else 0.90

    def rod_profile(t):
        return 0.60 if t < t_final * 0.5 else 0.80

    # Run loop
    for _ in range(steps):
        # advance time
        ps = ps.copy_advance_time(dt).clip_invariants()
        # commands
        ps.load_demand_pu = float(load_profile(ps.t_s))
        ps.rod_mode = "manual"
        ps.rod_cmd_manual_pu = float(rod_profile(ps.t_s))
        # step the orchestrator
        ps = ic.step(ps)

        ts.append(ps.t_s)
        Ths.append(ps.T_hot_K)
        Tcs.append(ps.T_cold_K)
        Tavgs.append(ps.Tavg_K)
        Ppris.append(ps.P_primary_Pa)
        Psecs.append(ps.P_secondary_Pa)
        Ppzrs.append(ps.pzr_pressure_Pa)
        Pcores.append(ps.P_core_W)
        Pturbs.append(ps.P_turbine_W)
        rods.append(ps.rod_pos_pu)
        rhos.append(ps.rho_reactivity_dk)
        loads.append(ps.load_demand_pu)
        rod_cmds.append(ps.rod_cmd_manual_pu if ps.rod_mode == "manual" else np.nan)

    # ---------------------------
    # Plotting
    # ---------------------------
    if plot and ts:
        t = np.array(ts)
        fig = plt.figure(figsize=(12, 9))

        ax = plt.subplot(3, 2, 1)
        ax.plot(t, Ths, label="T_hot [K]")
        ax.plot(t, Tcs, label="T_cold [K]")
        ax.plot(t, Tavgs, label="Tavg [K]")
        ax.set_title("Primary Temperatures")
        ax.legend(); ax.grid(True)

        ax = plt.subplot(3, 2, 2)
        ax.plot(t, np.array(Ppris)/1e6, label="P_primary [MPa]")
        ax.plot(t, np.array(Ppzrs)/1e6, label="P_pzr [MPa]")
        ax.plot(t, np.array(Psecs)/1e6, label="P_secondary [MPa]")
        ax.set_title("Pressures"); ax.legend(); ax.grid(True)

        ax = plt.subplot(3, 2, 3)
        ax.plot(t, np.array(Pcores)/1e6, label="P_core [MW]")
        ax.plot(t, np.array(Pturbs)/1e6, label="P_turb [MW]")
        ax.set_title("Powers"); ax.legend(); ax.grid(True)

        ax = plt.subplot(3, 2, 4)
        ax.plot(t, rods, label="rod_pos [pu]")
        ax.plot(t, np.array(rhos)*1e5, label="reactivity [pcm]")
        ax.set_title("Rod & Reactivity"); ax.legend(); ax.grid(True)

        ax = plt.subplot(3, 2, 5)
        ax.plot(t, loads, label="load [pu]")
        ax.plot(t, rod_cmds, label="rod_cmd [pu]")
        ax.set_title("Commands"); ax.legend(); ax.grid(True)

        plt.tight_layout()
        try:
            plt.show()
        except Exception as e:
            # Fallback for environments where interactive backends fail (e.g., PyCharm helper backend mismatch)
            out = os.path.abspath("run_plot.png")
            fig.savefig(out, dpi=150)
            print(f"[plot] Interactive show() failed ({e}). Saved figure to {out}")

    # Optional CSV logging
    if csv_path and ts:
        import csv
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t","T_hot_K","T_cold_K","Tavg_K","P_primary_Pa","P_secondary_Pa","P_pzr_Pa","P_core_W","P_turb_W","rod_pos_pu","rho_dk","load_cmd_pu","rod_cmd_pu"])
            for i in range(len(ts)):
                w.writerow([ts[i], Ths[i], Tcs[i], Tavgs[i], Ppris[i], Psecs[i], Ppzrs[i], Pcores[i], Pturbs[i], rods[i], rhos[i], loads[i], rod_cmds[i]])
        print(f"[csv] wrote {csv_path}")

    # ---------------------------
    # Assertions / summary
    # ---------------------------
    problems = []

    # 1) Each subsystem was called
    for name, stub in [("reactor", reactor), ("steamgen", sg), ("pressurizer", pzr), ("turbine", turb)]:
        if stub.call_count == 0:
            problems.append(f"{name}: never called")

    # 2) dt propagated correctly to subsystems (within a tolerance)
    for name, stub in [("reactor", reactor), ("steamgen", sg), ("pressurizer", pzr), ("turbine", turb)]:
        last_dt = stub.last_args.get("dt", None)
        if last_dt is None:
            problems.append(f"{name}: did not receive dt")
        else:
            if abs(float(last_dt) - dt) > 1e-9:
                problems.append(f"{name}: dt mismatch (got {last_dt}, expected {dt})")

    # 3) Manual rod command actually reached the reactor
    expected_cmd = 0.80  # final rod profile value
    last_cmd = reactor.last_args.get("manual_rod_cmd", None)
    if last_cmd is None or abs(float(last_cmd) - expected_cmd) > 1e-3:
        problems.append(f"reactor: manual_rod_cmd mismatch (got {last_cmd}, expected {expected_cmd})")

    # 4) Steam generator received cp and m_dot_primary
    if "cp_J_per_kgK" not in sg.last_args or "m_dot_primary_kg_s" not in sg.last_args:
        problems.append("steamgen: missing cp or m_dot_primary argument")

    # Report
    print("---- Test Summary ----")
    print(f"Steps run: {steps}, dt: {dt}")
    print(f"Calls  -> Reactor:{reactor.call_count}, SG:{sg.call_count}, PZR:{pzr.call_count}, TURB:{turb.call_count}")
    if problems:
        print("FAIL")
        for p in problems:
            print(" -", p)
    else:
        print("PASS  ✅  Timing, plots, and argument flow look good.")

    return not problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ICSystem wiring/timing harness with optional plots and CSV. Includes light dynamics for more physical behavior.")
    parser.add_argument("--no-plots", action="store_true", help="Disable matplotlib plots")
    parser.add_argument("--steps", type=int, default=120, help="Number of nominal steps if t_final not set")
    parser.add_argument("--csv", type=str, default=None, help="Optional CSV output path")

    # More-physical tuning knobs
    parser.add_argument("--tau-reactor", type=float, default=None, help="Reactor power time constant [s]")
    parser.add_argument("--tau-th", type=float, default=None, help="Reactor outlet temperature lag [s]")
    parser.add_argument("--alpha-T", type=float, default=None, help="Temp feedback coefficient [dk/k per K]")
    parser.add_argument("--tau-sg", type=float, default=None, help="SG cold-leg lag [s]")
    parser.add_argument("--tau-evap", type=float, default=None, help="SG evaporation lag [s]")
    parser.add_argument("--hfg", type=float, default=None, help="Secondary latent heat h_fg [J/kg]")
    parser.add_argument("--eta-evap", type=float, default=None, help="Evaporation efficiency [0..1]")
    parser.add_argument("--pzr-kp", type=float, default=None, help="Pressurizer PI Kp")
    parser.add_argument("--pzr-ki", type=float, default=None, help="Pressurizer PI Ki")
    parser.add_argument("--pzr-deadband", type=float, default=None, help="Pressurizer deadband [Pa]")
    parser.add_argument("--pzr-tau", type=float, default=None, help="Pressurizer pressure plant tau [s]")
    parser.add_argument("--pzr-gain", type=float, default=None, help="Pressurizer pressure plant gain [Pa per (heater-spray)]")
    parser.add_argument("--eta-e", type=float, default=None, help="Electrical efficiency")
    parser.add_argument("--tau-turb", type=float, default=None, help="Turbine power lag [s]")
    parser.add_argument("--tau-sec", type=float, default=None, help="Secondary pressure lag [s]")
    parser.add_argument("--ksec", type=float, default=None, help="Secondary pressure gain [Pa per kg/s steam mismatch]")

    args = parser.parse_args()

    ok = simulate(n_steps=args.steps, plot=not args.no_plots, csv_path=args.csv, args=args)
    import sys
    sys.exit(0 if ok else 1)
