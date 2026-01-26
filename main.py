import math
from dataclasses import replace
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


def ask_str(prompt: str, default: str) -> str:
    txt = input(prompt)
    txt = txt.strip().lower()
    return txt if txt else default.lower()


def run_sim():
    cfg = Config()

    # ---- 1. Get user inputs -------------------------------------------------
    load_dem_pu = ask_float(
        "Enter turbine load demand (per-unit, e.g. 1.0 for 100%): ",
        1.0,
    )

    rod_mode = ask_str(
        "Rod control mode ('auto' or 'manual') [manual]: ",
        "manual",
    )
    if rod_mode not in ("auto", "manual"):
        print("Invalid mode, defaulting to 'manual'")
        rod_mode = "manual"

    rod_step_pu = 0.0
    if rod_mode == "manual":
        rod_step_pu = ask_float(
            "Manual rod command (Δx per second, -1..1) [0.0]: ",
            0.0,
        )

    dt = cfg.dt
    t_final = cfg.t_final

    # ---- 2. Initialize plant objects ---------------------------------------
    ps = PlantState()
    ps.load_demand_pu = load_dem_pu
    ps.rod_mode = rod_mode
    ps.rod_cmd_manual_pu = rod_step_pu

    # Initialize reactor with consistent initial conditions
    reactor = ReactorSimulator(
        Tc_init=ps.T_cold_K,
        P_turb_init=ps.load_demand_pu,
        control_mode=ps.rod_mode,
    )

    # ICSystem wires reactor + SG + turbine together
    steamgen = SteamGenerator(cfg)
    turbine = TurbineModel(cfg)
    ics = ICSystem(cfg=cfg, reactor=reactor, steamgen=steamgen, turbine=turbine)

    # Sync PlantState to match component internals BEFORE first logging sample
    # (This removes the t=0 "vertical line" that happens when PlantState
    # starts at config-nominal values but SG/turbine internals have already
    # settled to something slightly different.)

    # Compute a consistent initial steam enthalpy for the current SG pressure
    try:
        from CoolProp.CoolProp import PropsSI
        _P0 = max(steamgen.P_sec_Pa, 1.0e5)
        _h0 = float(PropsSI("H", "P", _P0, "Q", 1, "Water"))
    except Exception:
        _P0 = max(steamgen.P_sec_Pa, 1.0e5)
        _h0 = float(cfg.h_steam_J_kg)

    # Compute a consistent initial turbine power for the SG's current state
    _Pt0 = float(
        turbine.turbine_power(
            inlet_p=_P0,
            inlet_h=_h0,
            outlet_p=turbine.outlet_p,
            m_dot_steam=steamgen.m_dot_steam_kg_s,
        )
    )

    ps = replace(
        ps,
        # secondary/SG states that often differ after SteamGenerator settles
        P_secondary_Pa=steamgen.P_sec_Pa,
        T_sec_K=steamgen.T_sec_K,
        T_metal_K=steamgen.T_metal_K,
        m_dot_steam_kg_s=steamgen.m_dot_steam_kg_s,
        m_dot_steam_cmd_kg_s=steamgen.m_dot_steam_kg_s,  # start cmd consistent with actual
        steam_h_J_kg=_h0,
        P_turbine_MW=_Pt0,
    )

    # Optional warmup lets the coupled system settle before recording data
    warmup_s = getattr(cfg, "startup_warm_time_s", 0.0)
    if warmup_s > 0.0:
        n_warm = max(1, int(math.ceil(warmup_s / dt)))
        for _ in range(n_warm):
            ps = ics.step(ps, dt)
            ps = ps.copy_advance_time(dt)
        ps = ps.clip_invariants()
        ps = replace(ps, t_s=0.0)

    # ---- 3. Allocate histories ----------------------------------------------
    t_hist = []
    Thot_hist = []
    Tcold_hist = []
    mdot_steam_hist = []
    Ppri_hist = []
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

    # ---- 4. Time integration loop -------------------------------------------
    n_steps = int(math.ceil(t_final / dt))

    for _ in range(n_steps + 1):
        # Record current state
        t_hist.append(ps.t_s)
        Thot_hist.append(ps.T_hot_K)
        Tcold_hist.append(ps.T_cold_K)
        mdot_steam_hist.append(ps.m_dot_steam_kg_s)
        Ppri_hist.append(ps.P_primary_Pa)
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

        # Step the integrated system
        ps = ics.step(ps, dt)

        # Advance time
        ps = ps.copy_advance_time(dt)

        if ps.t_s >= t_final:
            break

    # ---- 5. Convert histories to numpy arrays -------------------------------
    t = np.array(t_hist)
    Thot = np.array(Thot_hist)
    Tcold = np.array(Tcold_hist)
    mdot_steam = np.array(mdot_steam_hist)
    Ppri = np.array(Ppri_hist)
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

    caps = getattr(ics.steamgen, "_dbg_caps", None)
    if caps:
        print("\n--- SG caps (last call) ---")
        for k, v in caps.items():
            print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

    print("\n--- Steady-state (last sample) ---")
    print(f"P_secondary = {Psec[-1] / 1e6:.3f} MPa")
    print(f"m_dot_steam = {mdot_steam[-1]:.2f} kg/s")
    print(f"T_metal - T_sec = {(T_metal[-1] - T_sec[-1]):.2f} K")

    #Qsteam_W = mdot_steam[-1] * cfg.delta_h_steam_fw_J_kg

    try:
        import CoolProp.CoolProp as CP
        P = max(Psec[-1], 1e5)
        h_g = CP.PropsSI("H", "P", P, "Q", 1, "Water")
        h_fw = CP.PropsSI("H", "P", P, "T", cfg.T_fw_K, "Water")
        delta_h_ss = h_g - h_fw
    except Exception:
        delta_h_ss = cfg.delta_h_steam_fw_J_kg

    Qsteam_W = mdot_steam[-1] * delta_h_ss
    print(f"delta_h_ss = {delta_h_ss / 1e6:.3f} MJ/kg")

    print(f"P_core_ss = {Pcore[-1]:.1f} MWt")
    print(f"Q_steam_ss = {Qsteam_W / 1e6:.1f} MW")
    print(f"Implied eta = {Pturb[-1] * 1e6 / max(Qsteam_W, 1.0):.3f}")
    #print(f"m_dot_if_core_3400 = {3400e6 / cfg.delta_h_steam_fw_J_kg:.1f} kg/s")
    #print(f"m_dot_if_core_Pss = {Pcore[-1] * 1e6 / cfg.delta_h_steam_fw_J_kg:.1f} kg/s")
    print(f"m_dot_if_core_3400 = {3400e6 / delta_h_ss:.1f} kg/s")
    print(f"m_dot_if_core_Pss = {Pcore[-1] * 1e6 / delta_h_ss:.1f} kg/s")

    #Q_primary_ss_W = cfg.m_dot_primary_nom_kg_s * cfg.cp_primary_J_kgK * (Thot[-1] - Tcold[-1])
    Q_primary_core_W = cfg.m_dot_primary_nom_kg_s * cfg.cp_primary_J_kgK * (Thot[-1] - Tcold[-1])
    Q_primary_sg_W = cfg.m_dot_primary_nom_kg_s * cfg.cp_primary_J_kgK * (T_sg_in[-1] - T_sg_out[-1])

    print(f"Q_primary_core = {Q_primary_core_W / 1e6:.1f} MW")
    print(f"Q_primary_sg   = {Q_primary_sg_W / 1e6:.1f} MW")

    #print(f"Q_primary_ss = {Q_primary_ss_W / 1e6:.1f} MW")

    # ---- 6. Plotting (3×3 grid) --------------------------------------------
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    # Row 0, Col 0: Primary Temperatures
    ax = axes[0, 0]
    ax.plot(t, Thot, label="T_hot [K]")
    ax.plot(t, Tcold, label="T_cold [K]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Primary Temperatures")
    ax.legend(loc="best")

    # Row 0, Col 1: Pressures
    ax = axes[0, 1]
    ax.plot(t, Ppri / 1e6, label="P_primary [MPa]")
    ax.plot(t, Psec / 1e6, label="P_secondary [MPa]")
    ax.set_ylabel("Pressure [MPa]")
    ax.set_title("System Pressures")
    ax.legend(loc="best")

    # Row 0, Col 2: Powers
    ax = axes[0, 2]
    ax.plot(t, Pcore, label="P_core [MWt]")
    #ax.plot(t, Pturb, label="P_turbine [MWe]")
    ax.set_ylabel("Power")
    ax.set_title("Core & Turbine Power")
    ax.legend(loc="best")

    # Row 1, Col 0: Rod position & command
    ax = axes[1, 0]
    ax.plot(t, rod_pos, label="rod_pos [pu]")
    ax.plot(t, rod_cmd, label="rod_cmd_manual [pu]")
    ax.set_ylabel("Rod position [pu]")
    ax.set_title("Rod Position / Command")
    ax.legend(loc="best")

    # Row 1, Col 1: Steam generator & primary temperatures
    ax = axes[1, 1]
    ax.plot(t, Thot, label="T_hot [K]", linestyle="-")
    ax.plot(t, Tcold, label="T_cold [K]", linestyle="--")
    ax.plot(t, T_sg_in, label="T_sg_in [K]", linestyle="-.")
    ax.plot(t, T_sg_out, label="T_sg_out [K]", linestyle=":")
    ax.plot(t, T_sec, label="T_sec [K]")
    ax.plot(t, T_metal, label="T_metal [K]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Steam Generator & Primary Temperatures")
    ax.legend(loc="best", fontsize=8)

    # Row 1, Col 2: Reactivity
    ax = axes[1, 2]
    ax.plot(t, reactivity_pcm, label="reactivity [pcm]")
    ax.set_ylabel("Reactivity [pcm]")
    ax.set_title("Total Reactivity")
    ax.legend(loc="best")

    # Row 2, Col 0: Load demand vs turbine power
    ax = axes[2, 0]
    #ax.plot(t, load_dem, label="Load demand [MWe]")
    ax.plot(t, Pturb, label="P_turbine [MWe]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Power [MWe]")
    ax.set_title("Load Demand vs Turbine Power")
    ax.legend(loc="best")

    # Row 2, Col 1: Secondary pressure (zoomed)
    ax = axes[2, 1]
    ax.plot(t, Psec / 1e6, label="P_secondary [MPa]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pressure [MPa]")
    ax.set_title("Secondary Pressure")
    ax.legend(loc="best")

    # Row 2, Col 2: Steam mass flow
    ax = axes[2, 2]
    ax.plot(t, mdot_steam, label="m_dot_steam [kg/s]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mass flow [kg/s]")
    ax.set_title("Steam Mass Flow")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_sim()
