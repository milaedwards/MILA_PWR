"""
steam_generator_test.py

Quick sanity check: is the standalone steam generator approximately steady
when fed with nominal core power per loop and nominal primary/secondary
conditions?

Run with:
    python steam_generator_test.py
"""

from config import Config
from steam_generator import SG


def main():
    cfg = Config()
    sg = SG(cfg)

    # Time step and number of steps
    dt = cfg.dt
    N_STEPS = getattr(cfg, "N_DEBUG_STEPS", 100)

    # Nominal primary temps
    T_hot_nom = getattr(cfg, "T_HOT_INIT_K", 596.483)   # [K] reactor outlet / hot leg
    T_cold_nom = getattr(cfg, "T_COLD_INIT_K", 553.817) # [K] cold leg / reactor inlet

    # Core power per loop [W]
    Q_core_total_W = getattr(cfg, "Q_CORE_NOMINAL_W", 3.4e9)
    N_loops = float(getattr(cfg, "N_LOOPS", 2.0))
    Q_core_loop_W = Q_core_total_W / max(N_loops, 1.0)

    # Secondary nominal conditions
    p_s_nom = getattr(cfg, "P_SEC_CONST_PA", sg.p_s_nom_Pa)   # [Pa]
    mdot_fw = getattr(cfg, "mdot_fw_kg_s", getattr(cfg, "M_DOT_STEAM_NOM_KG_S", 1000.0))

    print("=" * 72)
    print("Steam generator steady-state sanity test")
    print(f"dt              = {dt:.4f} s")
    print(f"N_STEPS         = {N_STEPS}")
    print(f"T_hot_nom       = {T_hot_nom:.3f} K")
    print(f"T_cold_nom      = {T_cold_nom:.3f} K")
    print(f"Q_core_loop_nom = {Q_core_loop_W / 1e6:.3f} MW")
    print(f"p_s_nom         = {p_s_nom / 1e6:.3f} MPa")
    print(f"mdot_fw_nom     = {mdot_fw:.3f} kg/s")
    print("=" * 72)

    # -------- Initial SG state (near nominal) --------
    # We let SG._get_state fill in T_s from p_s if needed.
    state = {
        "T_rxu": T_hot_nom,      # reactor outlet (into hot leg)
        "T_hot": T_hot_nom,      # hot leg
        "T_sgi": T_hot_nom,      # SG inlet plenum
        "T_p1": T_hot_nom,       # primary lump 1
        "T_p2": T_cold_nom,      # primary lump 2
        "T_m1": T_hot_nom,       # metal node 1
        "T_m2": T_cold_nom,      # metal node 2
        "T_sgu": T_cold_nom,     # SG outlet plenum
        "T_cold": T_cold_nom,    # cold leg
        "T_rxi": T_cold_nom,     # reactor inlet
        "p_s": p_s_nom,          # steam pressure
        "M_dot_stm": mdot_fw,    # steam/feeedwater flow
        "Q_core_W": Q_core_loop_W,  # core power per loop
        "dt": dt,
    }

    # Let SG compute T_s from p_s
    s0 = sg._get_state(state)
    T0 = {
        "T_hot": s0["T_hot"],
        "T_p1": s0["T_p1"],
        "T_p2": s0["T_p2"],
        "T_m1": s0["T_m1"],
        "T_m2": s0["T_m2"],
        "T_cold": s0["T_cold"],
        "T_s": s0["T_s"],
    }
    p0 = s0["p_s"]
    mdot0 = s0["M_dot_stm"]

    print("Initial SG state (t = 0):")
    print(f"  T_hot   = {T0['T_hot']:8.3f} K")
    print(f"  T_p1    = {T0['T_p1']:8.3f} K")
    print(f"  T_p2    = {T0['T_p2']:8.3f} K")
    print(f"  T_m1    = {T0['T_m1']:8.3f} K")
    print(f"  T_m2    = {T0['T_m2']:8.3f} K")
    print(f"  T_cold  = {T0['T_cold']:8.3f} K")
    print(f"  T_s     = {T0['T_s']:8.3f} K")
    print(f"  p_s     = {p0/1e6:8.3f} MPa")
    print(f"  M_dot   = {mdot0:8.3f} kg/s")
    print("-" * 72)
    print("  step    t[s]   T_hot[K]  T_p1[K]  T_p2[K]  T_m1[K]  T_m2[K]  "
          "T_cold[K]   p_s[MPa]   T_s[K]   Mdot[kg/s]   dT_hot[K]")
    print("-" * 72)

    # Track max drifts
    max_drift = {
        "T_hot": 0.0,
        "T_p1": 0.0,
        "T_p2": 0.0,
        "T_m1": 0.0,
        "T_m2": 0.0,
        "T_cold": 0.0,
        "T_s": 0.0,
        "p_s": 0.0,
        "M_dot": 0.0,
    }

    t = 0.0

    for k in range(1, N_STEPS + 1):
        # Always feed the same reactor outlet temperature and core power
        state["T_rxu"] = T_hot_nom
        state["Q_core_W"] = Q_core_loop_W
        state["dt"] = dt

        out = sg.step(state)

        # Build next state from outputs
        state = {
            "T_rxu": T_hot_nom,
            "T_hot": out["T_hot_new"],
            "T_sgi": out["T_sgi_new"],
            "T_p1": out["T_p1_new"],
            "T_p2": out["T_p2_new"],
            "T_m1": out["T_m1_new"],
            "T_m2": out["T_m2_new"],
            "T_sgu": out["T_sgu_new"],
            "T_cold": out["T_cold_new"],
            "T_rxi": out["T_rxi_new"],
            "p_s": out["p_s_new"],
            "T_s": out["T_s_new"],
            "M_dot_stm": out["M_dot_stm_new"],
            "Q_core_W": Q_core_loop_W,
            "dt": dt,
        }

        t += dt

        # Current values
        Th = state["T_hot"]
        Tp1 = state["T_p1"]
        Tp2 = state["T_p2"]
        Tm1 = state["T_m1"]
        Tm2 = state["T_m2"]
        Tc = state["T_cold"]
        Ts = state["T_s"]
        ps = state["p_s"]
        mdot = state["M_dot_stm"]

        # Drifts from initial
        dTh = Th - T0["T_hot"]
        dTp1 = Tp1 - T0["T_p1"]
        dTp2 = Tp2 - T0["T_p2"]
        dTm1 = Tm1 - T0["T_m1"]
        dTm2 = Tm2 - T0["T_m2"]
        dTc = Tc - T0["T_cold"]
        dTs = Ts - T0["T_s"]
        dp = ps - p0
        dM = mdot - mdot0

        max_drift["T_hot"] = max(max_drift["T_hot"], abs(dTh))
        max_drift["T_p1"] = max(max_drift["T_p1"], abs(dTp1))
        max_drift["T_p2"] = max(max_drift["T_p2"], abs(dTp2))
        max_drift["T_m1"] = max(max_drift["T_m1"], abs(dTm1))
        max_drift["T_m2"] = max(max_drift["T_m2"], abs(dTm2))
        max_drift["T_cold"] = max(max_drift["T_cold"], abs(dTc))
        max_drift["T_s"] = max(max_drift["T_s"], abs(dTs))
        max_drift["p_s"] = max(max_drift["p_s"], abs(dp))
        max_drift["M_dot"] = max(max_drift["M_dot"], abs(dM))

        print(
            f"{k:6d}  {t:6.2f}  "
            f"{Th:9.3f}  "
            f"{Tp1:8.3f}  "
            f"{Tp2:8.3f}  "
            f"{Tm1:8.3f}  "
            f"{Tm2:8.3f}  "
            f"{Tc:9.3f}  "
            f"{ps/1e6:9.3f}  "
            f"{Ts:8.3f}  "
            f"{mdot:11.3f}  "
            f"{dTh:9.4f}"
        )

    print("-" * 72)
    print("Max absolute drifts over simulation:")
    print(f"  dT_hot   = {max_drift['T_hot']:8.4f} K")
    print(f"  dT_p1    = {max_drift['T_p1']:8.4f} K")
    print(f"  dT_p2    = {max_drift['T_p2']:8.4f} K")
    print(f"  dT_m1    = {max_drift['T_m1']:8.4f} K")
    print(f"  dT_m2    = {max_drift['T_m2']:8.4f} K")
    print(f"  dT_cold  = {max_drift['T_cold']:8.4f} K")
    print(f"  dT_s     = {max_drift['T_s']:8.4f} K")
    print(f"  dp_s     = {max_drift['p_s']/1e3:8.4f} kPa")
    print(f"  dM_dot   = {max_drift['M_dot']:8.4f} kg/s")
    print("=" * 72)
    print("If these drifts are large, the SG alone is not in true steady state.\n")


if __name__ == "__main__":
    main()