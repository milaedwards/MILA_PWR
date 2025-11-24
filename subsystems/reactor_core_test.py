"""
test_reactor_core_steady.py

Quick sanity check: is the standalone reactor core approximately steady
when held at nominal Tc_in and 100% power with no rod motion?

Run with:
    python test_reactor_core_steady.py
"""

from dataclasses import asdict

from config import Config
from reactor_core import ReactorSimulator


def main():
    # --- Load config and basic settings ---
    cfg = Config()

    # Use the same dt as the plant
    dt = cfg.dt

    # Number of steps for this short steady-state check
    # Falls back to 50 if N_DEBUG_STEPS isn't defined in Config.
    N_STEPS = getattr(cfg, "N_DEBUG_STEPS", 50)

    # Hold cold-leg temperature and turbine load constant
    Tc_in = float(cfg.T_COLD_INIT_K)
    P_turb = 1.0

    print("=" * 72)
    print("Reactor core steady-state sanity test")
    print(f"dt            = {dt:.4f} s")
    print(f"N_STEPS       = {N_STEPS}")
    print(f"T_cold_init   = {Tc_in:.3f} K")
    print("=" * 72)

    # --- Initialize reactor core in manual mode (no rod controller) ---
    reactor = ReactorSimulator(
        cfg=cfg,
        Tc_init=Tc_in,
        P_turb_init=P_turb,
        control_mode="manual",  # keep rods fixed
    )

    # Diagnostics at t = 0
    diag0 = reactor.get_diagnostics()
    T0 = {
        "T_f": diag0["T_f"],
        "Tc1": diag0["Tc1"],
        "Tc2": diag0["Tc2"],
        "T_core_inlet": diag0["T_core_inlet"],
        "T_hot_leg": diag0["T_hot_leg"],
        "T_avg": diag0["T_avg"],
    }
    P0 = diag0["P_MWt"]

    print("Initial state (t = 0):")
    print(f"  P_core   = {P0:8.3f} MWt")
    print(f"  T_f      = {T0['T_f']:8.3f} K")
    print(f"  Tc1      = {T0['Tc1']:8.3f} K")
    print(f"  Tc2      = {T0['Tc2']:8.3f} K")
    print(f"  Tin_core = {T0['T_core_inlet']:8.3f} K")
    print(f"  Thot     = {T0['T_hot_leg']:8.3f} K")
    print(f"  T_avg    = {T0['T_avg']:8.3f} K")
    print("-" * 72)
    print("  step    t[s]    P[MWt]   T_f[K]    Tc1[K]    Tc2[K]   Thot[K]   dT_avg[K]")
    print("-" * 72)

    # Track worst-case drifts
    max_drift = {
        "T_f": 0.0,
        "Tc1": 0.0,
        "Tc2": 0.0,
        "T_core_inlet": 0.0,
        "T_hot_leg": 0.0,
        "T_avg": 0.0,
        "P_MWt": 0.0,
    }

    # --- Time-stepping loop ---
    for k in range(1, N_STEPS + 1):
        # No rod motion in manual mode: manual_u = 0.0
        Th_out, P_out = reactor.step(
            Tc_in,
            dt,
            P_turb=P_turb,
            control_mode="manual",
            manual_u=0.0,
        )

        diag = reactor.get_diagnostics()
        t = diag["t"]

        # Compute drifts relative to initial values
        dT_f = diag["T_f"] - T0["T_f"]
        dTc1 = diag["Tc1"] - T0["Tc1"]
        dTc2 = diag["Tc2"] - T0["Tc2"]
        dT_core_inlet = diag["T_core_inlet"] - T0["T_core_inlet"]
        dT_hot_leg = diag["T_hot_leg"] - T0["T_hot_leg"]
        dT_avg = diag["T_avg"] - T0["T_avg"]
        dP = diag["P_MWt"] - P0

        # Update max absolute drifts
        max_drift["T_f"] = max(max_drift["T_f"], abs(dT_f))
        max_drift["Tc1"] = max(max_drift["Tc1"], abs(dTc1))
        max_drift["Tc2"] = max(max_drift["Tc2"], abs(dTc2))
        max_drift["T_core_inlet"] = max(max_drift["T_core_inlet"], abs(dT_core_inlet))
        max_drift["T_hot_leg"] = max(max_drift["T_hot_leg"], abs(dT_hot_leg))
        max_drift["T_avg"] = max(max_drift["T_avg"], abs(dT_avg))
        max_drift["P_MWt"] = max(max_drift["P_MWt"], abs(dP))

        # Print one line per step
        print(
            f"{k:6d}  {t:6.2f}  "
            f"{diag['P_MWt']:8.3f}  "
            f"{diag['T_f']:8.3f}  "
            f"{diag['Tc1']:8.3f}  "
            f"{diag['Tc2']:8.3f}  "
            f"{diag['T_hot_leg']:8.3f}  "
            f"{dT_avg:8.4f}"
        )

    # --- Summary ---
    print("-" * 72)
    print("Max absolute drifts over simulation:")
    print(f"  dP_core   = {max_drift['P_MWt']:8.4f} MWt")
    print(f"  dT_f      = {max_drift['T_f']:8.4f} K")
    print(f"  dTc1      = {max_drift['Tc1']:8.4f} K")
    print(f"  dTc2      = {max_drift['Tc2']:8.4f} K")
    print(f"  dTin_core = {max_drift['T_core_inlet']:8.4f} K")
    print(f"  dThot     = {max_drift['T_hot_leg']:8.4f} K")
    print(f"  dT_avg    = {max_drift['T_avg']:8.4f} K")
    print("=" * 72)
    print("If these drifts are large, the core alone is not in true steady state.\n")


if __name__ == "__main__":
    main()