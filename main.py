import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

from config import Config
from plant_state import PlantState
from ic_system import ICSystem

def make_load_profile(load_demand_pu: float):
    """
    Build a turbine-load profile function.

    For now this is just a constant load, but you can later replace this
    with more complex time-dependent logic (steps, ramps, etc.).
    """
    def prof(t: float) -> float:
        return load_demand_pu

    return prof


def make_rod_profile(rod_cmd_manual_pu: float):
    """
    Build a manual rod-command profile function.

    For now this is constant in time; later you can make this time-dependent
    if you want scripted transients instead of a single user setting.
    """
    def prof(t: float) -> float:
        return rod_cmd_manual_pu

    return prof


def ask_user_inputs():
    """
    Prompt the user for turbine load and rod-control settings.
    """
    # Turbine / generator load demand (per-unit)
    try:
        load_str = input(
            "Enter turbine load demand (per-unit, e.g. 1.0 for 100%): "
        ).strip()
        load_demand = float(load_str) if load_str else 1.0
    except ValueError:
        load_demand = 1.0

    # Rod control mode
    rod_mode = input(
        "Rod control mode ('auto' or 'manual') [manual]: "
    ).strip().lower() or "manual"
    if rod_mode not in ("auto", "manual"):
        rod_mode = "manual"

    # Initial rod insertion (only if in manual mode)
    rod_insert_pu = None
    if rod_mode == "manual":
        try:
            cmd_str = input(
                "Enter initial rod insertion (0-100%, e.g. 50 for mid-stroke, 0 for fully withdrawn, 100 for fully inserted): "
            ).strip()
            if cmd_str:
                percent = float(cmd_str)
            else:
                percent = 50.0  # default mid-stroke if user just hits Enter
            rod_insert_pu = float(np.clip(percent / 100.0, 0.0, 1.0))
        except ValueError:
            rod_insert_pu = 0.5  # fallback to mid-stroke
    else:
        rod_insert_pu = None

    return load_demand, rod_mode, rod_insert_pu


def build_default_subsystems(cfg: Config):
    """
    Factory to build subsystem models.

    This now constructs the real subsystem objects so you can test the
    coupled plant model:
      - reactor_core.ReactorSimulator
      - steam_generator.SG
      - turbine_condenser.TurbineModel

    If you later re-introduce a pressurizer model, you can extend this
    function accordingly.
    """
    from subsystems.reactor_core import ReactorSimulator
    from subsystems.steam_generator import SG
    from subsystems.turbine_condenser import TurbineModel

    def maybe_with_cfg(cls):
        """
        Helper so this works whether your classes take (cfg) in __init__
        or no arguments at all.
        """
        try:
            return cls(cfg)
        except TypeError:
            return cls()

    reactor = maybe_with_cfg(ReactorSimulator)
    # pressurizer = maybe_with_cfg(Pressurizer)  # placeholder for future use
    steamgen = maybe_with_cfg(SG)
    turbine = maybe_with_cfg(TurbineModel)

    return reactor, steamgen, turbine


def run(
    reactor,
    steamgen,
    turbine,
    load_demand_pu: float = 1.0,
    rod_mode: str = "manual",
    rod_insert_pu: float = 0.5,
    early_stop: bool = True,
    csv_out: bool = False,
    csv_name: str = "run_log.csv",
    cfg=None,
):
    """
    Core time-marching driver.

    This is the same structure you had before, but now the load and rod
    profiles are parameterized by the user's requested settings, and this
    function can be called from a simple main() entrypoint.
    """
    cfg = Config() if cfg is None else cfg
    ps = PlantState.init_from_config(cfg)

    # Apply initial rod insertion if in manual mode; rods are then held fixed
    if rod_mode == "manual" and rod_insert_pu is not None:
        ps = replace(ps, rod_pos_pu=float(rod_insert_pu), rod_mode="manual", rod_cmd_manual_pu=0.0)
    else:
        # Ensure rod_cmd_manual_pu is defined even if not used in auto mode
        ps = replace(ps, rod_mode=rod_mode, rod_cmd_manual_pu=0.0)

    ic = ICSystem(
        reactor=reactor,
        steamgen=steamgen,
        turbine=turbine,
        cfg=cfg,
    )

    load = make_load_profile(load_demand_pu)

    N = int(cfg.t_final / cfg.dt) + 1
    t = np.zeros(N)
    Th = np.zeros(N)
    Tc = np.zeros(N)
    Tavg = np.zeros(N)
    Ppri = np.zeros(N)
    Psec = np.zeros(N)
    # Ppzr = np.zeros(N)  # reserved for future pressurizer model
    Pcore = np.zeros(N)
    Pturb = np.zeros(N)
    rodpos = np.zeros(N)
    rho = np.zeros(N)
    dTavg = np.zeros(N)
    dPpri = np.zeros(N)

    for k in range(N):
        # Log current state
        t[k] = ps.t_s
        Th[k] = ps.T_hot_K
        Tc[k] = ps.T_cold_K
        Tavg[k] = ps.Tavg_K
        Ppri[k] = ps.P_primary_Pa
        Psec[k] = ps.P_secondary_Pa
        # Ppzr[k] = ps.pzr_pressure_Pa  # requires pressurizer / pzr_pressure_Pa field
        Pcore[k] = ps.P_core_W
        Pturb[k] = ps.P_turbine_W
        rodpos[k] = ps.rod_pos_pu
        rho[k] = ps.rho_reactivity_dk

        # Advance time and apply user commands
        ps = ps.copy_advance_time(cfg.dt).clip_invariants()
        ps = replace(
            ps,
            load_demand_pu=float(load(ps.t_s)),
            rod_mode=rod_mode,
            # In manual mode we hold rods fixed at the initial insertion (manual speed = 0.0)
            rod_cmd_manual_pu=0.0,
        )

        # Call the coupled plant model
        ps_next = ic.step(ps)

        # Derivatives for early-stop criteria
        dTavg[k] = (ps_next.Tavg_K - ps.Tavg_K) / cfg.dt
        dPpri[k] = (ps_next.P_primary_Pa - ps.P_primary_Pa) / cfg.dt

        ps = ps_next.clip_invariants()

        # Early-stop logic (unchanged)
        if early_stop and k > 20:
            window = max(1, int(10.0 / cfg.dt))
            if k > window:
                if (np.max(np.abs(dTavg[k - window : k])) < 1e-3) and (
                    np.max(np.abs(dPpri[k - window : k])) < 50.0
                ):
                    print(f"[early-stop] near steady-state at t={ps.t_s:.1f}s")
                    break
            if not (5e6 <= ps.P_primary_Pa <= 18e6):
                print(f"[early-stop] primary pressure out of bounds at t={ps.t_s:.1f}s")
                break

    # === Prepare data for plotting ===
    # Use only the portion of the arrays actually filled (handles early-stop)
    n_steps = k + 1  # k is the last index visited in the loop above
    t_plot = t[:n_steps]
    Th_plot = Th[:n_steps]
    Tc_plot = Tc[:n_steps]
    Tavg_plot = Tavg[:n_steps]
    Ppri_plot = Ppri[:n_steps]
    Psec_plot = Psec[:n_steps]
    Pcore_plot = Pcore[:n_steps]
    Pturb_plot = Pturb[:n_steps]
    rodpos_plot = rodpos[:n_steps]
    rho_plot = rho[:n_steps]

    # Commands as functions of time
    load_plot = np.array([load(ti) for ti in t_plot])
    if rod_mode == "manual" and rod_insert_pu is not None:
        rod_cmd_plot = np.full_like(t_plot, rod_insert_pu, dtype=float)
    else:
        rod_cmd_plot = np.zeros_like(t_plot)

    # Normalizations
    Pcore_nom = getattr(cfg, "Q_CORE_NOMINAL_W", 1.0)
    if Pcore_nom <= 0.0:
        Pcore_nom = 1.0
    Pcore_norm = Pcore_plot / Pcore_nom

    Psec_nom = getattr(cfg, "P_SEC_INIT_PA", 1.0)
    if Psec_nom <= 0.0:
        Psec_nom = 1.0
    Psec_norm = Psec_plot / Psec_nom

    # For turbine, normalize by its initial value if available,
    # otherwise by its maximum nonzero value, otherwise fall back to 1.0.
    if n_steps > 0 and Pturb_plot[0] > 0.0:
        Pturb_nom = Pturb_plot[0]
    else:
        max_pturb = np.max(Pturb_plot)
        Pturb_nom = max_pturb if max_pturb > 0.0 else 1.0
    Pturb_norm = Pturb_plot / Pturb_nom

    # === Plots: 3x3 grid for the 9 requested views ===
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))

    # 1. Primary loop temperatures vs time
    ax = axes[0, 0]
    ax.plot(t_plot, Th_plot, label="T_hot [K]")
    ax.plot(t_plot, Tc_plot, label="T_cold [K]")
    ax.plot(t_plot, Tavg_plot, label="T_avg [K]")
    ax.set_title("Primary Temperatures")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.legend()
    ax.grid(True)

    # 2. Pressures vs time (absolute)
    ax = axes[0, 1]
    ax.plot(t_plot, Ppri_plot / 1e6, label="P_primary [MPa]")
    ax.plot(t_plot, Psec_plot / 1e6, label="P_secondary [MPa]")
    ax.set_title("Pressures")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pressure [MPa]")
    ax.legend()
    ax.grid(True)

    # 3. Powers vs time (absolute)
    ax = axes[0, 2]
    ax.plot(t_plot, Pcore_plot / 1e6, label="P_core [MWt]")
    ax.plot(t_plot, Pturb_plot / 1e6, label="P_turbine [MWe]")
    ax.set_title("Powers")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Power [MW]")
    ax.legend()
    ax.grid(True)

    # 4. Rod input / position vs time
    ax = axes[1, 0]
    ax.plot(t_plot, rodpos_plot, label="rod_pos [pu]")
    ax.plot(t_plot, rod_cmd_plot, label="rod_setpoint [pu]")
    ax.set_title("Rod Command and Position")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Per-unit")
    ax.legend()
    ax.grid(True)

    # 5. Normalized core power vs time
    ax = axes[1, 1]
    ax.plot(t_plot, Pcore_norm, label="P_core / P_core_nom")
    ax.set_title("Normalized Core Power")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Per-unit")
    ax.legend()
    ax.grid(True)

    # 6. Reactivity vs time
    ax = axes[1, 2]
    ax.plot(t_plot, rho_plot * 1e5, label="reactivity [pcm]")
    ax.set_title("Reactivity")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Reactivity [pcm]")
    ax.legend()
    ax.grid(True)

    # 7. Load demand vs time
    ax = axes[2, 0]
    ax.plot(t_plot, load_plot, label="load demand [pu]")
    ax.set_title("Load Demand")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Per-unit")
    ax.legend()
    ax.grid(True)

    # 8. Normalized secondary pressure vs time
    ax = axes[2, 1]
    ax.plot(t_plot, Psec_norm, label="P_secondary / P_sec_nom")
    ax.set_title("Normalized Secondary Pressure")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Per-unit")
    ax.legend()
    ax.grid(True)

    # 9. Normalized turbine power vs time
    ax = axes[2, 2]
    ax.plot(t_plot, Pturb_norm, label="P_turbine / P_turb_nom")
    ax.set_title("Normalized Turbine Power")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Per-unit")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    if csv_out:
        data = np.column_stack(
            [t, Th, Tc, Tavg, Ppri, Psec, Pcore, Pturb, rodpos, rho]
        )
        header = "t,Th,Tc,Tavg,Ppri,Psec,Pcore,Pturb,rod,rho"
        np.savetxt(csv_name, data, delimiter=",", header=header, comments="")
        print(f"[csv] wrote {csv_name}")


def main():
    """
    CLI entrypoint: prompt the user, build subsystems, run the sim, show plots.
    """
    cfg = Config()
    reactor, steamgen, turbine = build_default_subsystems(cfg)
    # pressurizer = None  # placeholder if a pressurizer model is added later

    load_demand_pu, rod_mode, rod_insert_pu = ask_user_inputs()

    run(
        reactor=reactor,
        steamgen=steamgen,
        turbine=turbine,
        load_demand_pu=load_demand_pu,
        rod_mode=rod_mode,
        rod_insert_pu=rod_insert_pu,
        early_stop=True,
        csv_out=False,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
