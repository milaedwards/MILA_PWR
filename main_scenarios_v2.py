import math
import matplotlib.pyplot as plt
import numpy as np

from config import Config
from plant_state import PlantState
from reactor_core import ReactorSimulator
from steam_generator import SteamGenerator
from turbine_condenser import TurbineModel
from ic_system import ICSystem


def ask_float(prompt: str, default: float) -> float:
    """Read a float from input, with a default if the user hits Enter."""
    txt = input(prompt)
    if not txt.strip():
        return default
    try:
        return float(txt)
    except ValueError:
        print(f"Invalid input, using default {default}")
        return default


def ask_time_seconds(prompt: str, default_s: float) -> float:
    txt = input(prompt).strip().lower()
    if not txt:
        return default_s
    try:
        if txt.endswith("m"):
            return float(txt[:-1]) * 60.0
        if txt.endswith("h"):
            return float(txt[:-1]) * 3600.0
        return float(txt)
    except ValueError:
        print(f"Invalid input, using default {default_s}")
        return default_s


def ask_str(prompt: str, default: str) -> str:
    txt = input(prompt)
    txt = txt.strip().lower()
    return txt if txt else default.lower()


def run_sim():
    cfg = Config()

    print("=" * 70)
    print("AP1000 PLANT SIMULATOR - SCENARIO MODE")
    print("=" * 70)

    # ---- 1. Get control mode -------------------------------------------------
    rod_mode = ask_str(
        "\nRod control mode ('auto' or 'manual'): ",
        "auto",
    )
    if rod_mode not in ("auto", "manual"):
        print("Invalid mode, defaulting to 'auto'")
        rod_mode = "auto"

    # ---- 2. Scenario-specific inputs -----------------------------------------
    rod_step_percent = 0.0
    load_cut_mwe = 0.0
    event_time = 1000.0  # seconds
    rod_movement_duration = 10.0  # Duration over which rods move (seconds)

    if rod_mode == "manual":
        print("\n--- MANUAL ROD MOVEMENT SCENARIO ---")
        print("Rods will move at t=1000s")
        rod_step_percent = ask_float(
            "Enter rod movement (% of full stroke, + to INSERT, - to WITHDRAW): ",
            0.0,
        )

        # Ask for movement duration
        rod_movement_duration = ask_float(
            "Enter rod movement duration (seconds) [default 10s]: ",
            10.0,
        )

        # Convert percentage to per-unit change
        rod_step_pu = rod_step_percent / 100.0

        # Calculate rod speed to achieve this change over the duration
        rod_speed_pu_per_s = rod_step_pu / rod_movement_duration

        print(f"\nScenario configured:")
        print(f"  Rod movement: {rod_step_percent:+.1f}% ({'+INSERT' if rod_step_percent > 0 else 'WITHDRAW'})")
        print(f"  Movement duration: {rod_movement_duration:.1f} seconds")
        print(f"  Rod speed: {abs(rod_speed_pu_per_s) * 100:.2f}%/s")
        print(f"  Event time: {event_time:.0f} seconds")

    else:  # auto mode
        print("\n--- AUTO MODE WITH LOAD CUT SCENARIO ---")
        print("Turbine load cut will occur at t=1000s")
        load_cut_mwe = ask_float(
            "Enter load cut magnitude (MWe, positive = reduction): ",
            50.0,
        )
        rod_step_pu = 0.0  # Not used in auto mode
        rod_speed_pu_per_s = 0.0  # Not used in auto mode

        print(f"\nScenario configured:")
        print(f"  Initial load: {cfg.P_e_nom_MWe:.1f} MWe")
        print(f"  Load cut: {load_cut_mwe:.1f} MWe")
        print(f"  Final load: {cfg.P_e_nom_MWe - load_cut_mwe:.1f} MWe")
        print(f"  Event time: {event_time:.0f} seconds")

    # ---- 3. Time controls ----------------------------------------------------
    default_t_final = 2000.0  # Default 2000s for scenario runs
    t_final = ask_time_seconds(
        f"\nEnter total simulation time (e.g. 2000, 30m) [default {default_t_final}s]: ",
        default_t_final,
    )

    dt = ask_float(
        f"Enter time step dt in seconds [default {cfg.dt}]: ",
        cfg.dt,
    )

    # Basic safety
    if dt <= 0.0:
        print(f"Invalid dt, using default {cfg.dt}")
        dt = cfg.dt
    if t_final <= 0.0:
        print(f"Invalid t_final, using default {default_t_final}")
        t_final = default_t_final

    if t_final < event_time + 500:
        print(f"Warning: Simulation time ({t_final}s) should be at least {event_time + 500}s")
        print(f"         to see response after event at {event_time}s")

    # ---- 4. Initialize plant objects -----------------------------------------
    ps = PlantState()

    # Initial load demand (before any cut)
    initial_load_pu = 1.000  # Natural equilibrium from our debugging
    ps.load_demand_pu = initial_load_pu

    # Make turbine power consistent with load demand at t=0
    ps.P_turbine_MW = ps.load_demand_pu * cfg.P_e_nom_MWe

    # Optional: make initial steam command consistent too
    ps.m_dot_steam_cmd_kg_s = ps.load_demand_pu * cfg.m_dot_steam_nom_kg_s

    ps.rod_mode = rod_mode
    ps.rod_cmd_manual_pu = 0.0  # Start with no rod motion

    print(f"\n{'=' * 70}")
    print(f"INITIALIZATION:")
    print(f"{'=' * 70}")
    print(f"Initial load demand: {ps.load_demand_pu:.3f} pu ({ps.load_demand_pu * cfg.P_e_nom_MWe:.1f} MWe)")
    print(f"Rod control mode: {rod_mode}")
    print(f"Time step: {dt} s")
    print(f"Simulation duration: {t_final} s")
    print(f"{'=' * 70}\n")

    # Initialize reactor
    reactor = ReactorSimulator(
        Tc_init=ps.T_cold_K,
        P_turb_init=ps.load_demand_pu,
        control_mode=ps.rod_mode,
    )

    print("init dT_f:", reactor.Tf - reactor.fb.T_f0)
    print("init dT_m:", reactor.Tc1 - reactor.fb.Tc10)

    # ICSystem wires reactor + SG + turbine together
    steamgen = SteamGenerator(cfg)
    turbine = TurbineModel(cfg)
    ics = ICSystem(cfg=cfg, reactor=reactor, steamgen=steamgen, turbine=turbine)

    # ---------------- FIX 1: sync PlantState to subsystem equilibrium BEFORE logging ----------------

    # (Keep turbine power + steam command consistent with initial load)
    ps.P_turbine_MW = ps.load_demand_pu * cfg.P_e_nom_MWe
    ps.m_dot_steam_cmd_kg_s = ps.load_demand_pu * cfg.m_dot_steam_nom_kg_s

    # Reactor-consistent values
    ps.T_hot_K = reactor.T_hot_leg
    ps.P_core_MW = reactor.P_pu * cfg.P_core_nom_MWt
    ps.rod_pos_pu = reactor.x
    ps.rho_reactivity_dk = getattr(reactor, "rho_tot", 0.0)

    # SG-consistent values (optional but helps eliminate cold-leg mismatch)
    Tcold0, m0, Psec0, Tsec0, lim0, h0 = steamgen.step(ps.T_hot_K, ps.m_dot_steam_cmd_kg_s, dt=0.0)

    ps.T_cold_K = Tcold0
    ps.T_sg_in_K = ps.T_hot_K
    ps.T_sg_out_K = Tcold0
    ps.P_secondary_Pa = Psec0
    ps.T_sec_K = Tsec0
    ps.m_dot_steam_kg_s = m0
    ps.sg_power_limited = lim0
    ps.steam_h_J_kg = h0
    ps.T_metal_K = steamgen.T_metal_K  # step() sets this internally
    # -----------------------------------------------------------------------------------------------

    # ---- 5. Allocate histories -----------------------------------------------
    t_hist = []
    Thot_hist = []
    Tcold_hist = []
    mdot_steam_hist = []
    Ppri_hist = []
    Ppzr_hist = []
    Psec_hist = []
    Pcore_hist = []
    Pturb_hist = []
    rod_pos_hist = []
    rod_cmd_hist = []
    load_dem_hist = []
    T_sg_in_hist = []
    T_sg_out_hist = []
    T_sec_hist = []
    T_metal_hist = []
    reactivity_hist = []
    pzr_level_hist = []
    pzr_heater_hist = []
    pzr_spray_hist = []

    # ---- 6. Time integration loop --------------------------------------------
    n_steps = int(math.ceil(t_final / dt))

    print(f"Running simulation ({n_steps} steps)...")
    event_triggered = False
    rod_movement_start_time = 0.0
    target_rod_position = 0.0  # Will be set when event triggers

    for step in range(n_steps + 1):
        # Record current state
        t_hist.append(ps.t_s)
        Thot_hist.append(ps.T_hot_K)
        Tcold_hist.append(ps.T_cold_K)
        mdot_steam_hist.append(ps.m_dot_steam_kg_s)
        Ppri_hist.append(ps.P_primary_Pa)
        Ppzr_hist.append(ps.P_pzr_Pa)
        Psec_hist.append(ps.P_secondary_Pa)
        Pcore_hist.append(ps.P_core_MW)
        Pturb_hist.append(ps.P_turbine_MW)
        rod_pos_hist.append(ps.rod_pos_pu)
        rod_cmd_hist.append(ps.rod_cmd_manual_pu)
        load_dem_hist.append(ps.load_demand_pu)
        T_sg_in_hist.append(ps.T_sg_in_K)
        T_sg_out_hist.append(ps.T_sg_out_K)
        T_sec_hist.append(ps.T_sec_K)
        T_metal_hist.append(ps.T_metal_K)
        reactivity_hist.append(ps.rho_reactivity_dk)
        pzr_level_hist.append(ps.pzr_level_m)
        pzr_heater_hist.append(ps.pzr_heater_pu)
        pzr_spray_hist.append(ps.pzr_spray_pu)

        # ---- Apply scenario event at t=1000s ---------------------------------
        if ps.t_s >= event_time and not event_triggered:
            event_triggered = True
            rod_movement_start_time = ps.t_s

            if rod_mode == "manual":
                # Start manual rod movement
                ps.rod_cmd_manual_pu = rod_speed_pu_per_s
                target_rod_position = ps.rod_pos_pu + rod_step_pu

                print(f"\n[t={ps.t_s:.1f}s] MANUAL ROD MOVEMENT INITIATED")
                print(f"  Current position: {ps.rod_pos_pu:.3f} pu ({ps.rod_pos_pu * 100:.1f}%)")
                print(f"  Target position: {target_rod_position:.3f} pu ({target_rod_position * 100:.1f}%)")
                print(f"  Movement: {rod_step_percent:+.1f}%")
                print(f"  Duration: {rod_movement_duration:.1f} seconds")

            else:  # auto mode
                # Apply load cut
                new_load_pu = (cfg.P_e_nom_MWe - load_cut_mwe) / cfg.P_e_nom_MWe
                ps.load_demand_pu = new_load_pu
                print(f"\n[t={ps.t_s:.1f}s] TURBINE LOAD CUT INITIATED")
                print(f"  Load before: {initial_load_pu * cfg.P_e_nom_MWe:.1f} MWe")
                print(f"  Load after: {new_load_pu * cfg.P_e_nom_MWe:.1f} MWe")
                print(f"  Reduction: {load_cut_mwe:.1f} MWe")

        # ---- Check if manual rod movement should stop ------------------------
        if rod_mode == "manual" and event_triggered:
            elapsed_time = ps.t_s - rod_movement_start_time

            # Stop rod movement after duration OR if target reached
            if elapsed_time >= rod_movement_duration:
                if ps.rod_cmd_manual_pu != 0.0:  # Only print once
                    print(f"\n[t={ps.t_s:.1f}s] ROD MOVEMENT COMPLETE")
                    print(f"  Final position: {ps.rod_pos_pu:.3f} pu ({ps.rod_pos_pu * 100:.1f}%)")
                ps.rod_cmd_manual_pu = 0.0  # Stop moving

            # Also check if we've reached/exceeded target position
            if rod_step_pu > 0 and ps.rod_pos_pu >= target_rod_position:
                # Inserting and reached target
                if ps.rod_cmd_manual_pu != 0.0:
                    print(f"\n[t={ps.t_s:.1f}s] ROD MOVEMENT COMPLETE (target reached)")
                    print(f"  Final position: {ps.rod_pos_pu:.3f} pu ({ps.rod_pos_pu * 100:.1f}%)")
                ps.rod_cmd_manual_pu = 0.0
            elif rod_step_pu < 0 and ps.rod_pos_pu <= target_rod_position:
                # Withdrawing and reached target
                if ps.rod_cmd_manual_pu != 0.0:
                    print(f"\n[t={ps.t_s:.1f}s] ROD MOVEMENT COMPLETE (target reached)")
                    print(f"  Final position: {ps.rod_pos_pu:.3f} pu ({ps.rod_pos_pu * 100:.1f}%)")
                ps.rod_cmd_manual_pu = 0.0

        # Step the integrated system
        ps = ics.step(ps, dt)

        # Advance time
        ps = ps.copy_advance_time(dt)

        # Progress indicator every 10% completion
        if step % (n_steps // 10) == 0 and step > 0:
            progress = step / n_steps * 100
            print(f"  Progress: {progress:.0f}% (t={ps.t_s:.0f}s)")

        if ps.t_s >= t_final:
            break

    print(f"\nSimulation complete! (t={ps.t_s:.1f}s)")

    # ---- 7. Convert histories to numpy arrays --------------------------------
    t = np.array(t_hist)
    Thot = np.array(Thot_hist)
    Tcold = np.array(Tcold_hist)
    mdot_steam = np.array(mdot_steam_hist)
    Ppri = np.array(Ppri_hist)
    Ppzr = np.array(Ppzr_hist)
    Psec = np.array(Psec_hist)
    Pcore = np.array(Pcore_hist)
    Pturb = np.array(Pturb_hist)
    rod_pos = np.array(rod_pos_hist)
    rod_cmd = np.array(rod_cmd_hist)
    load_dem = np.array(load_dem_hist) * cfg.P_e_nom_MWe  # convert pu → MWe
    T_sg_in = np.array(T_sg_in_hist)
    T_sg_out = np.array(T_sg_out_hist)
    T_sec = np.array(T_sec_hist)
    T_metal = np.array(T_metal_hist)
    reactivity_pcm = np.array(reactivity_hist) * 1.0e5  # dk/k → pcm
    pzr_level = np.array(pzr_level_hist)
    pzr_heater = np.array(pzr_heater_hist)
    pzr_spray = np.array(pzr_spray_hist)

    # ---- 8. Plotting (3×3 grid) ----------------------------------------------
    print("\nGenerating plots...")
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    scenario_title = f"Manual Rod {'+INSERT' if rod_step_percent > 0 else 'WITHDRAW'} {abs(rod_step_percent):.0f}%" if rod_mode == "manual" else f"Load Cut {load_cut_mwe:.0f} MWe"
    fig.suptitle(f"AP1000 Simulator - {scenario_title} at t={event_time:.0f}s", fontsize=14, fontweight='bold')

    # Row 0, Col 0: Primary Temperatures
    ax = axes[0, 0]
    ax.plot(t, Thot, label="T_hot [K]")
    ax.plot(t, Tcold, label="T_cold [K]")
    ax.axvline(event_time, color='red', linestyle='--', alpha=0.5, label='Event')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Primary Temperatures")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.98, 0.05,
        f"Final T_hot: {Thot[-1]:.2f} K\nFinal T_cold: {Tcold[-1]:.2f} K",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Row 0, Col 1: Pressurizer Pressure
    ax = axes[0, 1]
    p_mpa = Ppzr / 1e6
    ax.plot(t, p_mpa, label="P_pressurizer [MPa]")

    pad = 0.05 * (p_mpa.max() - p_mpa.min())
    if pad == 0:
        pad = 0.1  # fallback if perfectly flat
    ax.set_ylim(p_mpa.min() - pad, p_mpa.max() + pad)

    ax.axvline(event_time, color='red', linestyle='--', alpha=0.5, label='Event')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pressure [MPa]")
    ax.set_title("Pressurizer Pressure")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Row 0, Col 2: Core & Turbine Power
    ax = axes[0, 2]
    ax.plot(t, Pcore, label="P_core [MWt]")
    ax.plot(t, Pturb, label="P_turbine [MWe]")
    ax.axvline(event_time, color='red', linestyle='--', alpha=0.5, label='Event')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Power")
    ax.set_title("Core & Turbine Power")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.05, f"Final P_core [MWt]: {Pcore[-1]:.2f} K\nFinal P_turbine [MWe]: {Pturb[-1]:.2f} K",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Row 1, Col 0: Rod position & command
    ax = axes[1, 0]
    ax.plot(t, rod_pos, label="rod_pos [pu]", linewidth=2)
    #ax.plot(t, rod_cmd, label="rod_cmd_manual [pu]", linestyle='--')
    ax.axvline(event_time, color='red', linestyle='--', alpha=0.5, label='Event')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Rod position [pu]")
    ax.set_title("Rod Position / Command")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(useOffset=False, style='plain', axis='y')

    # Row 1, Col 1: Steam generator & primary temperatures
    ax = axes[1, 1]
    ax.plot(t, Thot, label="T_hot [K]", linestyle="-")
    ax.plot(t, Tcold, label="T_cold [K]", linestyle="--")
    ax.plot(t, T_sg_in, label="T_sg_in [K]", linestyle="-.")
    ax.plot(t, T_sg_out, label="T_sg_out [K]", linestyle=":")
    ax.plot(t, T_sec, label="T_sec [K]")
    ax.plot(t, T_metal, label="T_metal [K]")
    ax.axvline(event_time, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Steam Generator & Primary Temperatures")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 1, Col 2: Reactivity
    ax = axes[1, 2]
    ax.plot(t, reactivity_pcm, label="reactivity [pcm]")
    ax.axvline(event_time, color='red', linestyle='--', alpha=0.5, label='Event')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Reactivity [pcm]")
    ax.set_title("Total Reactivity")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Row 2, Col 0: Load demand vs turbine power
    ax = axes[2, 0]
    ax.plot(t, load_dem, label="Load demand [MWe]")
    ax.plot(t, Pturb, label="P_turbine [MWe]")
    ax.axvline(event_time, color='red', linestyle='--', alpha=0.5, label='Event')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Power [MWe]")
    ax.set_title("Load Demand vs Turbine Power")
    ax.legend(loc="best")
    ax.ticklabel_format(style='plain', useOffset=False)
    ax.grid(True, alpha=0.3)

    # Row 2, Col 1: Secondary Pressure
    ax = axes[2, 1]
    ax.plot(t, Psec / 1e6, label="P_secondary [MPa]")
    ax.axvline(event_time, color='red', linestyle='--', alpha=0.5, label='Event')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pressure [MPa]")
    ax.set_title("Secondary Pressure")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Row 2, Col 2: Steam Mass Flow
    ax = axes[2, 2]
    ax.plot(t, mdot_steam, label="m_dot_steam [kg/s]")
    ax.axvline(event_time, color='red', linestyle='--', alpha=0.5, label='Event')
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mass flow [kg/s]")
    ax.set_title("Steam Mass Flow")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(useOffset=False, style='plain', axis='y')

    plt.tight_layout()
    plt.show()

    print("\nSimulation finished successfully!")


if __name__ == "__main__":
    run_sim()