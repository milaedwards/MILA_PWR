import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

from config import Config
from plant_state import PlantState
from ic_system import ICSystem

def make_load_profile(final_load_pu: float, t_step: float, initial_load_pu: float):
    """Build a turbine-load profile function with a step at t_step.

    Parameters
    ----------
    final_load_pu : float
        The per-unit load after the step time.
    t_step : float
        Time (s) at which the load changes from the initial value to the
        final value.
    initial_load_pu : float
        The per-unit load before the step time. This will typically be the
        nominal/initial plant load from the PlantState.
    """
    def prof(t: float) -> float:
        if t < t_step:
            return initial_load_pu
        return final_load_pu

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

    # Rod insertion step (only if in manual mode), expressed in percentage points.
    # For example, entering 5 means "insert rods by 5% of full stroke", so if the
    # core starts at 57% insertion it will move toward ~62%.
    rod_step_pu = 0.0
    if rod_mode == "manual":
        try:
            cmd_str = input(
                "Enter rod insertion step (percentage points, e.g. 5 for +5% insertion, -5 for 5% withdrawal, 0 for no motion): "
            ).strip()
            if cmd_str:
                percent_step = float(cmd_str)
            else:
                percent_step = 0.0  # default: no motion
            rod_step_pu = percent_step / 100.0
        except ValueError:
            rod_step_pu = 0.0  # fallback: no motion
    else:
        rod_step_pu = 0.0

    return load_demand, rod_mode, rod_step_pu


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
    rod_step_pu: float = 0.0,
    early_stop: bool = True,
    csv_out: bool = True,
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

    print(
        f"[CFG] H_ws-cpTfw = {cfg.H_ws_minus_cpTfw_J_kg / 1e6:.3f} MJ/kg, "
        f"m_dot_steam_nom = {cfg.M_DOT_STEAM_NOM_KG_S:.1f} kg/s, "
        f"Q_core_nom = {cfg.Q_CORE_NOMINAL_W / 1e6:.1f} MW, "
        f"m*H = {cfg.M_DOT_STEAM_NOM_KG_S * cfg.H_ws_minus_cpTfw_J_kg / 1e6:.1f} MW"
    )

    ps = PlantState.init_from_config(cfg)

    # Set rod mode and initialize manual command to zero; the ICSystem will interpret
    # rod_cmd_manual_pu as a one-shot delta insertion, not a speed profile.
    ps = replace(ps, rod_mode=rod_mode, rod_cmd_manual_pu=0.0)

    # Time (s) at which the one-shot rod step command should be applied.
    rod_step_t_start = 10.0
    # Time (s) at which the turbine load demand should change.
    load_step_t_start = 10.0
    # Internal flag to ensure we send the rod step command only once.
    rod_step_sent = False

    ic = ICSystem(
        reactor=reactor,
        steamgen=steamgen,
        turbine=turbine,
        cfg=cfg,
    )

    # Build a load profile that holds the initial plant load until
    # load_step_t_start, then steps to the user-requested load_demand_pu.
    initial_load_pu = ps.load_demand_pu
    load = make_load_profile(
        final_load_pu=load_demand_pu,
        t_step=load_step_t_start,
        initial_load_pu=initial_load_pu,
    )

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
    m_steam = np.zeros(N)
    rodpos = np.zeros(N)
    rho = np.zeros(N)
    dTavg = np.zeros(N)
    dPpri = np.zeros(N)

    sg_Tp_in = np.full(N, np.nan)
    sg_Tp_out = np.full(N, np.nan)
    sg_T_metal = np.full(N, np.nan)

    for k in range(N):
        # Log current state
        t[k] = ps.t_s
        Th[k] = ps.T_hot_K
        Tc[k] = ps.T_cold_K
        Tavg[k] = ps.Tavg_K
        Ppri[k] = ps.P_primary_Pa
        Psec[k] = ps.P_secondary_Pa
        # Ppzr[k] = ps.pzr_pressure_Pa  # requires pressurizer / pzr_pressure_Pa field
        Pcore[k] = ps.P_core_MW
        Pturb[k] = ps.P_turbine_MW
        m_steam[k] = ps.m_dot_steam_kg_s
        rodpos[k] = ps.rod_pos_pu
        rho[k] = ps.rho_reactivity_dk

        val_in = getattr(ps, "T_sg_primary_in_K", None)
        if val_in is not None:
            sg_Tp_in[k] = val_in

        val_out = getattr(ps, "T_sg_primary_out_K", None)
        if val_out is not None:
            sg_Tp_out[k] = val_out

        val_m = getattr(ps, "T_sg_metal_K", None)
        if val_m is not None:
            sg_T_metal[k] = val_m

        # Advance time and apply user commands
        ps = ps.copy_advance_time(cfg.dt).clip_invariants()
        # Determine one-shot rod step command (delta insertion) if in manual mode.
        if rod_mode == "manual":
            if (not rod_step_sent) and abs(rod_step_pu) > 0.0 and ps.t_s >= rod_step_t_start:
                # Issue a single "insert/withdraw by X%" command in per-unit of stroke.
                manual_cmd = rod_step_pu
                rod_step_sent = True
            else:
                manual_cmd = 0.0
        else:
            manual_cmd = 0.0
        ps = replace(
            ps,
            load_demand_pu=float(load(ps.t_s)),
            rod_mode=rod_mode,
            rod_cmd_manual_pu=manual_cmd,
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
    m_steam_plot = m_steam[:n_steps]
    rodpos_plot = rodpos[:n_steps]
    rho_plot = rho[:n_steps]

    sg_Tp_in_plot = sg_Tp_in[:n_steps]
    sg_Tp_out_plot = sg_Tp_out[:n_steps]
    sg_T_metal_plot = sg_T_metal[:n_steps]

    # Commands as functions of time
    load_plot = np.array([load(ti) for ti in t_plot])
    # For rods, plot the requested delta-insertion command (per-unit of stroke).
    if rod_mode == "manual" and abs(rod_step_pu) > 0.0:
        # Show a step in the commanded insertion starting at rod_step_t_start.
        rod_cmd_plot = np.where(t_plot >= rod_step_t_start, rod_step_pu, 0.0)
    else:
        rod_cmd_plot = np.zeros_like(t_plot)

    # Normalizations
    Pcore_nom = getattr(cfg, "Q_CORE_NOMINAL_MW", 1.0)
    if Pcore_nom <= 0.0:
        Pcore_nom = 1.0
    Pcore_norm = Pcore_plot / Pcore_nom

    Psec_nom = getattr(cfg, "P_SEC_INIT_PA", 1.0)
    if Psec_nom <= 0.0:
        Psec_nom = 1.0
    Psec_norm = Psec_plot / Psec_nom

    # For turbine, normalize by a *fixed* nominal rating,
    # using the turbine model's own nameplate (P_nom_MWe) so it matches ICSystem.
    Pturb_nom = getattr(turbine, "P_nom_MWe", None)
    if Pturb_nom is None or Pturb_nom <= 0.0:
        # fallback to config if turbine doesn't define P_nom_MWe
        Pturb_nom = getattr(cfg, "P_TURBINE_NOMINAL_MW", None)
    if Pturb_nom is None or Pturb_nom <= 0.0:
        # final fallback: use max observed turbine power or 1.0
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
    ax.plot(t_plot, Pcore_plot, label="P_core [MWt]")
    ax.plot(t_plot, Pturb_plot, label="P_turbine [MWe]")
    ax.set_title("Powers")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Power [MW]")
    ax.legend()
    ax.grid(True)

    # 4. Rod input / position vs time
    ax = axes[1, 0]
    ax.plot(t_plot, rodpos_plot, label="rod_pos [pu]")
    ax.plot(t_plot, rod_cmd_plot, label="rod_step_cmd [pu]")
    ax.set_title("Rod Command and Position")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Per-unit (position / fraction of stroke)")
    ax.legend()
    ax.grid(True)

    # 5. Steam generator temperatures vs time (if available)
    ax = axes[1, 1]
    any_plotted = False
    if not np.all(np.isnan(sg_Tp_in_plot)):
        ax.plot(t_plot, sg_Tp_in_plot, label="T_sg_primary_in [K]")
        any_plotted = True
    if not np.all(np.isnan(sg_Tp_out_plot)):
        ax.plot(t_plot, sg_Tp_out_plot, label="T_sg_primary_out [K]")
        any_plotted = True
    if not np.all(np.isnan(sg_T_metal_plot)):
        ax.plot(t_plot, sg_T_metal_plot, label="T_sg_metal [K]")
        any_plotted = True

    if not any_plotted:
        ax.text(0.5, 0.5, "No SG temperature fields on PlantState", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Steam Generator Temperatures")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
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

    # 7. Load demand vs turbine power vs time (absolute MWe)
    P_load_plot = load_plot * cfg.P_RATED_MWe  # [MWe]

    ax = axes[2, 0]
    ax.plot(t_plot, P_load_plot, label="load demand [MWe]")
    ax.plot(t_plot, Pturb_plot, label="P_turbine [MWe]")
    ax.set_title("Load Demand vs Turbine Power")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Power [MWe]")
    # Disable scientific offset so the constant load line does not appear at 0
    ax.ticklabel_format(style="plain", useOffset=False, axis="y")
    ax.legend()
    ax.grid(True)

    # 8. Normalized secondary pressure vs time
    ax = axes[2, 1]
    ax.plot(t_plot, Psec_plot / 1e6, label="P_secondary [MPa]]")
    ax.set_title("Secondary Pressure")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pressure [MPa]")
    ax.legend()
    ax.grid(True)

    # 9. Steam mass flow vs time
    ax = axes[2, 2]
    ax.plot(t_plot, m_steam_plot, label="m_dot_steam [kg/s]")
    ax.set_title("Steam Mass Flow")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mass flow [kg/s]")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    if csv_out:
        data = np.column_stack(
            [
                t_plot,
                Th_plot,
                Tc_plot,
                Tavg_plot,
                Ppri_plot,
                Psec_plot,
                Pcore_plot,
                Pturb_plot,
                rodpos_plot,
                rho_plot * 1e5,
                m_steam_plot,
            ]
        )
        header = "t,Th,Tc,Tavg,Ppri,Psec,Pcore,Pturb,rod,rho,m_dot_steam"
        np.savetxt(csv_name, data, delimiter=",", header=header, comments="")
        print(f"[csv] wrote {csv_name}")

def main():
    """
    CLI entrypoint: prompt the user, build subsystems, run the sim, show plots.
    """
    cfg = Config()
    reactor, steamgen, turbine = build_default_subsystems(cfg)
    # pressurizer = None  # placeholder if a pressurizer model is added later

    load_demand_pu, rod_mode, rod_step_pu = ask_user_inputs()

    run(
        reactor=reactor,
        steamgen=steamgen,
        turbine=turbine,
        load_demand_pu=load_demand_pu,
        rod_mode=rod_mode,
        rod_step_pu=rod_step_pu,
        early_stop=False,
        csv_out=False,
        cfg=cfg,
    )

if __name__ == "__main__":
    main()

