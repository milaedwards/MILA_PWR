import numpy as np
import matplotlib.pyplot as plt

from config import Config  # require your project config
from plant_states import PlantState
from ic_system import ICSystem

def make_load_profile():
    def prof(t: float) -> float:
        return 1.0 if t < 120.0 else 0.90
    return prof

def make_rod_profile():
    def prof(t: float) -> float:
        return 0.55  # change as needed
    return prof

def run(reactor, pressurizer, steamgen, turbine, early_stop=True, csv_out=False, csv_name="run_log.csv", cfg=None):
    # Always use your Config unless an explicit cfg is passed in (e.g., tests)
    cfg = Config() if cfg is None else cfg
    ps = PlantState.init_from_config(cfg)
    ic = ICSystem(cfg, reactor=reactor, pressurizer=pressurizer, steamgen=steamgen, turbine=turbine)

    load = make_load_profile()
    rod = make_rod_profile()

    N = int(cfg.t_final / cfg.dt) + 1
    t = np.zeros(N)
    Th = np.zeros(N); Tc = np.zeros(N); Tavg = np.zeros(N)
    Ppri = np.zeros(N); Psec = np.zeros(N); Ppzr = np.zeros(N)
    Pcore = np.zeros(N); Pturb = np.zeros(N); rodpos = np.zeros(N); rho = np.zeros(N)
    dTavg = np.zeros(N); dPpri = np.zeros(N)

    for k in range(N):
        t[k] = ps.t_s
        Th[k] = ps.T_hot_K; Tc[k] = ps.T_cold_K; Tavg[k] = ps.Tavg_K
        Ppri[k] = ps.P_primary_Pa; Psec[k] = ps.P_secondary_Pa; Ppzr[k] = ps.pzr_pressure_Pa
        Pcore[k] = ps.P_core_W; Pturb[k] = ps.P_turbine_W
        rodpos[k] = ps.rod_pos_pu; rho[k] = ps.rho_reactivity_dk

        # Commands
        ps = ps.copy_advance_time(cfg.dt).clip_invariants()
        ps = ps.__class__(**{**ps.__dict__,
            "load_demand_pu": float(load(ps.t_s)),
            "rod_mode": "manual",
            "rod_cmd_manual_pu": float(rod(ps.t_s)),
        })

        # Step
        ps_next = ic.step(ps)

        # Derivatives for early stop
        dTavg[k] = (ps_next.Tavg_K - ps.Tavg_K) / cfg.dt
        dPpri[k] = (ps_next.P_primary_Pa - ps.P_primary_Pa) / cfg.dt

        ps = ps_next.clip_invariants()

        if early_stop and k > 20:
            window = max(1, int(10.0 / cfg.dt))
            if k > window:
                if (np.max(np.abs(dTavg[k-window:k])) < 1e-3) and (np.max(np.abs(dPpri[k-window:k])) < 50.0):
                    print(f"[early-stop] near steady-state at t={ps.t_s:.1f}s"); break
            if not (5e6 <= ps.P_primary_Pa <= 18e6):
                print(f"[early-stop] primary pressure out of bounds at t={ps.t_s:.1f}s"); break

    # Plots
    fig = plt.figure(figsize=(12, 9))
    ax = plt.subplot(3,2,1); ax.plot(t, Th, label="T_hot [K]"); ax.plot(t, Tc, label="T_cold [K]"); ax.plot(t, Tavg, label="Tavg [K]"); ax.set_title("Primary Temperatures"); ax.legend(); ax.grid(True)
    ax = plt.subplot(3,2,2); ax.plot(t, Ppri/1e6, label="P_primary [MPa]"); ax.plot(t, Ppzr/1e6, label="P_pzr [MPa]"); ax.plot(t, Psec/1e6, label="P_secondary [MPa]"); ax.set_title("Pressures"); ax.legend(); ax.grid(True)
    ax = plt.subplot(3,2,3); ax.plot(t, Pcore/1e6, label="P_core [MWt]"); ax.plot(t, Pturb/1e6, label="P_turb [MWe]"); ax.set_title("Powers"); ax.legend(); ax.grid(True)
    ax = plt.subplot(3,2,4); ax.plot(t, rodpos, label="rod_pos [pu]"); ax.plot(t, rho*1e5, label="reactivity [pcm]"); ax.set_title("Rod & Reactivity"); ax.legend(); ax.grid(True)
    ax = plt.subplot(3,2,5); ax.plot(t, [load(ti) for ti in t], label="load demand [pu]"); ax.set_title("Commands"); ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.show()

    if csv_out:
        data = np.column_stack([t, Th, Tc, Tavg, Ppri, Psec, Ppzr, Pcore, Pturb, rodpos, rho])
        header = "t,Th,Tc,Tavg,Ppri,Psec,Ppzr,Pcore,Pturb,rod,rho"
        np.savetxt(csv_name, data, delimiter=",", header=header, comments="")
        print(f"[csv] wrote {csv_name]")
