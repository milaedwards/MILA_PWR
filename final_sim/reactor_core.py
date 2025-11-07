# AP1000 Point-Kinetics Simulator (BOC–HFP, no xenon)
# - Full 6-group (Keepin U-235 dominant) point-kinetics scaled to beta-eff from AP1000 DCD with Radau solver (for stiff ODEs)
# - AP1000 rod worth from DCD Figure "All rods inserting less most reactive stuck rod"
#   implemented as a normalized LUT + PCHIP spline, scaled to TOTAL WORTH = 10,490 pcm
#   (Table 4.3-3 item 3b: "All assemblies but one (highest worth inserted)").
# - BOC–HFP temperature feedbacks (fuel Doppler + moderator), xenon ignored
# - Multi-lumps = 3 parallel fuel/coolant lumps 
# - Control Rod Drive Mechanics (CRDM) kinematics: 45 in/min over 166.755 in
# - Sliding Tavg program: T_ref(P_turb) = T_m0 + S_Tavg * (P_turb − 1)
# - Recieves 4 inputs from main.py:
#    (1) cold leg temperature (Tc_in); (2) time step (dt); (3) turbine power demand in per-unit (P_turb); and (4) rod control mode (auto/manual)
#    and outputs hot leg temperature (Th_out) and reactor power (P_out)
# - Internal simulation: Tc rises (simulated at t=10 seconds), then Th_eq_v and T_avg rise
#   PI controller sees T_avg > T_ref (since T_ref is constant with turbine at 1.0)
#   rods insert to reduce power, mimicking a load cut driven by a hotter cold leg

import numpy as np
import os, sys, matplotlib
# --- plotting backend policy (explicit & IDE/CI friendly) ---
if os.environ.get("CI") or os.environ.get("PYCHARM_HOSTED"):
    matplotlib.use("Agg")
else:
    try:
        matplotlib.use("QtAgg")
    except Exception:
        try:
            matplotlib.use("MacOSX" if sys.platform == "darwin" else "TkAgg")
        except Exception:
            matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Export the following variables for running on main.py
__all__ = [
    'ReactorSimulator',  # Main interface for external use
    'run',               # For standalone testing
    'PKParams',          # If external programs need to modify physics
    'FBParams',          # If external programs need to modify parameters
    'RodWorthAP1000LUT', # If external programs need rod worth data
    'P_RATED_MWT',       # Useful constants
    'P_RATED_MWE'
]

# Global constants
P_RATED_MWT = 3400.0  # ~AP1000 nominal thermal
P_RATED_MWE = 1117.0  # ~AP1000 nominal electric
P_RATED_W = P_RATED_MWT * 1e6
PCM_TO_DK = 1e-5  # conversion factor pcm to delta_k/k
STROKE_IN = 166.755  # rod dimensions from AP1000 DCD Chapter 3, section 3.9.4.2.1
VMAX_IN_PER_MIN = 45.0  # maximum travel in inches per minute from AP1000 DCD Chapter 3, section 3.9.4.2.1
X_OP = 0.55  # % rod insertion at BOC-HFP critical operations
U_FC = 954.2  # W/m²·K, fuel-to-coolant heat transfer coefficient
A_FC = 5267.6  # m², fuel-to-coolant heat transfer surface area for entire core
UA = U_FC * A_FC  # W/K

# core multi-lump settings
N_LUMPS = 3 # number of fuel-coolant lumps
W_LUMP = np.ones(N_LUMPS, float) / N_LUMPS # heat split weights w_k (sum=1)
UA_k = UA * W_LUMP # per-lump UA (keeps ΔT_fc identical at HFP)

STEP_T_S       = 10.0 # seconds (time of the load cut)
P_TURB_STEP_PU = 1.0 - 50.0/P_RATED_MWE  # ≈ 0.9552 pu (50 MWe load reduction)

def pcm_to_dk(pcm: float) -> float:
    return pcm * PCM_TO_DK


def x_rate_max() -> float:
    return (
                VMAX_IN_PER_MIN / 60.0) / STROKE_IN  # conversion of maximum stroke rate from per minute to per second [s^-1]


# Rod worth Lookup Table (LUT) data points and Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)
#  generated with ChatGPT on 9/14/2025 with Figure 4.3-30 from AP1000 DCD
@dataclass
class RodWorthAP1000LUT:
    rho_tot_pcm: float = 10490.0  # Table 4.3-3 item 3b → 10,490 pcm (all-rods minus MOST worth stuck)
    x_pts: np.ndarray = field(default_factory=lambda: np.array(
        [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00], dtype=float))
    f_pts: np.ndarray = field(default_factory=lambda: np.array(
        [0.0000, 0.0001, 0.0003, 0.0020, 0.0055, 0.0200, 0.0600, 0.1200, 0.2400, 0.4600,
         0.7000, 0.9000, 0.9700, 1.0000], dtype=float))

    def __post_init__(self):
        x = np.clip(self.x_pts, 0.0, 1.0).astype(float)
        f = np.clip(self.f_pts, 0.0, 1.0).astype(float)
        f[0], f[-1] = 0.0, 1.0
        self._s = PchipInterpolator(x, f, extrapolate=False)
        self._ds = self._s.derivative()

    # insertion ⇒ NEGATIVE reactivity
    def rho_pcm_abs(self, x: float) -> float:
        x = np.clip(x, 0.0, 1.0)  # normalizes the rod insertion inches from 0 to 1.0
        return -float(self._s(x)) * self.rho_tot_pcm

    def rho_pcm_rel(self, x: float, x_ref: float = X_OP) -> float:
        return self.rho_pcm_abs(x) - self.rho_pcm_abs(
            x_ref)  # defining relative reactivity given the rod starting position X_0


# Point Kinetics Parameters from AP1000 DCD Table 4.3-2
@dataclass
class PKParams:
    Lambda: float = 19.8e-6  # Prompt neutron generation time [s]
    # Keepin shape for U-235 (scaled to beta_eff below) due to U-235 dominating BOC
    beta_shape: np.ndarray = field(default_factory=lambda: np.array(
        [0.000247, 0.001642, 0.00147, 0.002963, 0.000863, 0.000315], dtype=float))
    lambda_i: np.ndarray = field(default_factory=lambda: np.array(
        [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01], dtype=float))
    beta_eff: float = 0.0075  # from AP1000 DCD Table 4.3-2

    # normalizing beta fractions and computing beta_eff
    def __post_init__(self):
        self.beta_i = self.beta_shape * (self.beta_eff / float(self.beta_shape.sum()))
        self.beta = float(self.beta_i.sum())


# Temperature Feedback Parameters from AP1000 DCD Table 4.3-2
@dataclass
class FBParams:
    # DCD coefficients in pcm/°F (converted to pcm/K in use)
    alpha_D_input: float = -1.4  # Doppler, pcm/°F
    alpha_M_input: float = -2.0  # Moderator, pcm/°F
    coeff_units: str = "F"

    # Reference temps (Kelvin) at HFP
    T_f0: float = 1253.0  # average fuel temperature at BOC-HFP (980 degrees Celsius)
    T_m0: float = 574.04 # from Table 4.4-1 of AP1000 DCD

    # Lags
    tau_f_s: float = 0.8 # fuel time constant used to smooth out response (Holbert: 0.8 seconds)
    tau_m_s: float = 7.0 # moderator time constant used to smooth out response
    tau_hot_s: float = 3.0  # hot-leg (core outlet) response time constant [s] (Holbert: 2.5 or 3 seconds )
    tau_cold_s: float = 3.0  # cold-leg lag (core/loop mixing) [s]

    W_kgps: float = 14516.1   # kg/s  mass flow rate of coolant calculated from equation 12.5 in Kerlin/Upadhyaya textbook
    cp_JpkgK: float = 5461 # J/(kg*K)  specific heat capcity of coolant at 2248 psig and 300 deg C from https://enghandbook.com/thermodynamic-calculators/water/?temperature=300&pressure=155

# -------------------- Sliding T_avg program ----------------
# Load step: −50 MWe at t = 10 s  →  P_turb goes from 1.0 to ~0.955
T_STEP_TIME = 10.0  # seconds
DELTA_MWE = -50.0  # MWe load cut
DELTA_P_TURB = DELTA_MWE / P_RATED_MWE  # ≈ −0.045 normalized power step
P_TURB_INIT = 1.0  # normalized turbine power at t=0

# Programmed T_avg reference vs turbine load (linear “sliding” schedule)
# Modest slope (order 10–20 K per p.u.) so Tref changes a few kelvin for a few-% load step.
S_TAVG_K_PER_PU = 15.0 #


# Turbine demand at time t
def P_turb_dem(t: float) -> float:
    return 1.0 if t < STEP_T_S else P_TURB_STEP_PU

def turbine_load_pu(t: float) -> float:
    # step to 1 - 50/1117 at any chosen time (e.g., 20 s)
    return 1.0 if t < 20.0 else (1.0 - 50.0 / P_RATED_MWE)  # 0.95525 pu


# Reference moderator temperature setpoint calculation based on turbine load
def Tref(P_turb: float, fb: FBParams) -> float:
    return fb.T_m0 + S_TAVG_K_PER_PU * (P_turb - 1.0)


# -------------------- Rod Proportional-Integral (PI) Controller based on Tavg error ----------------
# Controller output u in [-U_MAX,+U_MAX]; +u = insertion (negative ρ)
Kp = +0.03  # proportional gain if Tavg > Tref (too hot), then +u (insert rods) → Kp>0
Ki = +0.0012  # integral gain
K_load = 0.0  # K per unit (P - P_turb)
U_MAX = 0.12  # maximum controller output to prevent rods from being inserted or withdrawn too aggressively
K_AW = 2.0  # anti-windup back-calculation gain to avoid overshoot
LEAK = 0.05  # integral leak [1/s] (small) to prevent integral term from growing indefinitely and becoming unstable
LEAK_OUTER = 0.01  # Outer loop integral leak [1/s] (slower than inner)
Kpow_i = 0.75   # outer power-error integrator (start conservative)
Z_BIAS_MAX = 5.0  # clamp for the bias (kelvin)
DB_K = 0.10  # temperature deadband [K] around Tref prevents control rods from constantly moving to correct small fluctuations
K_feedforward = 0.7  # Gain: how much to relax T_avg target per K of T_cold rise

# introduce deadband to suppress small errors
def _deadband(e: float, db: float) -> float:
    if e > db: return e - db  # if error is greater than the positive deadband value, return the error minus the deadband.
    if e < -db: return e + db  # if error is less than the negative deadband value, return the error plus the deadband
    return 0.0  # if error is within the range of the deadband, return 0.0


# calculate rod speed command variable u and integrator update variable dz using PI controller
def rod_speed_pi_aw(Tavg: float, Tref_val: float, z: float):
    # calculate raw error (Tavg - Tref_val) where Tavg is current average temperature and Tref_val is desired reference temperature
    e_db = _deadband(Tavg - Tref_val, DB_K)  # pass error to deadband function defined above

    # calculate unsaturated PI output before any limits are applied
    u0 = Kp * e_db + Ki * z  # sum of proportional and integral terms

    # limit controller output to physical speed limit of control rods
    u = max(-U_MAX, min(U_MAX, u0))

    # Conditional integration:
    # If saturated *and* pushing further into saturation, pause integrating the error
    pushing_sat = (abs(u) >= U_MAX) and (np.sign(u) == np.sign(u0))
    if pushing_sat:
        dz = (u - u0) / K_AW - LEAK * z  # unwind if saturated
    else:
        dz = e_db + (u - u0) / K_AW - LEAK * z  # integrate error + back-calc + leak

    return u, dz  # returns the rod speed controller output u and rate of change dz


_rhs_did_print_once = False

# calculate right-hand side (rhs) of ODE system using Radau solver
def rhs(t, y, pk: PKParams, fb: FBParams, worth: RodWorthAP1000LUT, Tcold_fn, P_turb_fn, control_mode, manual_u):
    global _rhs_did_print_once
    # indices
    N_DG  = pk.lambda_i.size
    idx_P  = 0 # reactor power
    idx_C  = slice(1, 1 + N_DG) # precursor concentrations
    start_T = 1 + N_DG
    idx_Tf  = slice(start_T, start_T + N_LUMPS) # 3 fuel lumps
    idx_Th  = slice(start_T + N_LUMPS, start_T + 2*N_LUMPS) # 3 resulting hot leg temps
    idx_x   = start_T + 2*N_LUMPS # rod position
    idx_z   = idx_x + 1 # integral term
    idx_Z   = idx_x + 2 # outer loop bias

    # unpack
    P     = max(0.0, float(y[idx_P]))
    C     = y[idx_C]
    T_fv  = y[idx_Tf]
    T_hv  = y[idx_Th]
    Tc_in = float(Tcold_fn(t))                  # EXTERNAL disturbance
    x     = np.clip(float(y[idx_x]), 0.0, 1.0)
    z     = float(y[idx_z])
    z_pow = float(y[idx_Z])

    # coefficients (pcm/K)
    alpha_D_K = fb.alpha_D_input * (9.0/5.0) if fb.coeff_units == "F" else fb.alpha_D_input
    alpha_M_K = fb.alpha_M_input * (9.0/5.0) if fb.coeff_units == "F" else fb.alpha_M_input

    # core-average moderator temperature
    T_avg = float(np.dot(W_LUMP, 0.5*(T_hv + Tc_in)))

    # setpoint (turbine demand fixed at 1.0 here)
    P_turb = P_turb_dem(t)
    T_ref  = Tref(P_turb_fn(t), fb)

    # Outer power-tracking loop: bias T_ref to drive power toward turbine demand
    z_pow_clamped = np.clip(z_pow, -Z_BIAS_MAX, Z_BIAS_MAX)

    # Feedforward compensation for cold-leg temperature disturbance
    # When T_cold is hotter than nominal, accept proportionally higher T_avg
    Tcold0_nominal = 552.59  # Kelvin, nominal cold leg temp at HFP from Table 4.1-1
    dT_cold_disturbance = Tc_in - Tcold0_nominal  # Deviation from nominal
    T_cold_feedforward = K_feedforward * dT_cold_disturbance

    # Apply both outer loop bias AND feedforward compensation
    T_m_eq_avg = T_ref + z_pow_clamped + T_cold_feedforward

    # rod PI
    if control_mode == 'manual':
        u = manual_u  # Use external command
        dz = -LEAK * z  # Let integral wind down
    else:
        u, dz = rod_speed_pi_aw(T_avg, T_m_eq_avg, z)  # Auto control

    dx = x_rate_max() * u

    # reactivity (pcm → Δk/k)
    rho_rod_pcm       = worth.rho_pcm_rel(x, X_OP)
    rho_doppler_pcm   = alpha_D_K * float(np.dot(W_LUMP, (T_fv - fb.T_f0)))
    rho_moderator_pcm = alpha_M_K * (T_avg - fb.T_m0)
    rho_total         = pcm_to_dk(rho_rod_pcm + rho_doppler_pcm + rho_moderator_pcm)

    # kinetics
    dP = ((rho_total - pk.beta) / pk.Lambda) * P + np.sum(pk.lambda_i * C)
    dC = (pk.beta_i / pk.Lambda) * P - pk.lambda_i * C

    # thermal
    Q_core_k = P * P_RATED_W * W_LUMP
    Wcp_k    = fb.W_kgps * fb.cp_JpkgK * W_LUMP
    Th_eq_v  = Tc_in + Q_core_k / Wcp_k
    dT_h_v   = (Th_eq_v - T_hv) / fb.tau_hot_s

    T_f_eq_v = T_avg + Q_core_k / UA_k
    dT_f_v   = (T_f_eq_v - T_fv) / fb.tau_f_s

    # Outer power-tracking integrator
    P_error = P_turb - P  # Positive when turbine demands more than actual
    P_DEADBAND = 0.02  # Don't react to <2% power errors
    if abs(P_error) > P_DEADBAND:
        dz_pow = Kpow_i * P_error
    else:
        dz_pow = -LEAK_OUTER * z_pow  # Slowly decay bias if no error

    # Anti-windup for outer loop
    z_pow_clamped = np.clip(z_pow, -Z_BIAS_MAX, Z_BIAS_MAX)
    if abs(z_pow_clamped) >= Z_BIAS_MAX and np.sign(z_pow) == np.sign(P_error):
        dz_pow = 0.0  # Stop integrating if saturated

    # Print only once, right after the step occurs (Radau calls RHS many times around the event)
    if (not _rhs_did_print_once) and (t >= STEP_T_S):
        print(f"t={t:.2f}s: P={P:.4f}, P_turb={P_turb:.4f}, P_error={P_error:.4f}, dz_pow={dz_pow:.6f}")
        _rhs_did_print_once = True

    # pack derivatives
    dy = np.zeros(1 + N_DG + 2*N_LUMPS + 3)
    dy[idx_P]  = dP
    dy[idx_C]  = dC
    dy[idx_Tf] = dT_f_v
    dy[idx_Th] = dT_h_v
    dy[idx_x]  = dx
    dy[idx_z]  = dz
    dy[idx_Z]  = dz_pow
    return dy

# Initialize starting values (MULTI-LUMP + Tc_in state)
def initial_state(pk: PKParams, fb: FBParams, Tcold0: float, Th_eq0_vec: np.ndarray):
    """
    State order: [P, C1..CN, T_f[0..N-1], T_h[0..N-1], x, z, z_pow]
    """
    P0 = 1.0
    C0 = (pk.beta_i / (pk.Lambda * pk.lambda_i)) * P0
    T_f_init = np.full(N_LUMPS, fb.T_f0, float)
    T_h_init = np.array(Th_eq0_vec, dtype=float)         # <-- equilibrium hot-leg
    x0, z0 = X_OP, 0.0
    z_pow0 = 0.0  # Start with no bias (outer loop will build up as needed)

    y0 = np.concatenate(([P0], C0, T_f_init, T_h_init, [x0, z0, z_pow0]))
    expected = 1 + pk.lambda_i.size + 2*N_LUMPS + 3
    assert y0.size == expected, f"y0 size {y0.size} != {expected}"
    return y0

# Run simulation
def run(Tcold_fn=None):
    pk = PKParams()
    fb = FBParams()
    worth = RodWorthAP1000LUT()

    # HFP references
    dT_fc_hfp = P_RATED_W / UA
    fb.T_m0   = 574.04 # Kelvin, average between 535 deg F inlet and 612.2 deg F outlet
    fb.T_f0   = fb.T_m0 + dT_fc_hfp
    try: fb.T_f_ref = fb.T_f0
    except AttributeError: pass

    # loop ΔT and initial legs
    dT_loop_hfp = P_RATED_W / (fb.W_kgps * fb.cp_JpkgK)
    Tavg0 = fb.T_m0
    Tcold0 = Tavg0 - 0.5*dT_loop_hfp
    Th0    = Tavg0 + 0.5*dT_loop_hfp

    if Tcold_fn is None:
    # Model SG cooling reduction: lower turbine power → less steam flow → hotter T_cold
    # Estimate: ~4K rise for ~4.5% power reduction
        def Tcold_fn(tt):
            P_t = P_turb_dem(tt)
            # Linear approximation: ΔT_cold ≈ k * ΔP_turb
            # For 4.5% load drop → +4K, so k ≈ 89 K per unit power change
            dT_cold = 89.0 * (1.0 - P_t)  # Positive when P_t < 1.0
            return Tcold0 + dT_cold

    print(f"UA = {UA:.3e} W/K,  ΔT_fc@HFP = {dT_fc_hfp:.1f} K (expect ~35–45)")
    print(f"ΔT_loop@HFP = {dT_loop_hfp:.1f} K (expect ~35–45)")

    # ---- equilibrium hot-leg at t=0 (per-lump)
    Tc0        = float(Tcold_fn(0.0))
    Q_core_k0  = 1.0 * P_RATED_W * W_LUMP
    Wcp_k      = fb.W_kgps * fb.cp_JpkgK * W_LUMP
    Th_eq0_vec = Tc0 + Q_core_k0 / Wcp_k

    # build y0 (no Tc_in state)
    y0 = initial_state(pk, fb, Tc0, Th_eq0_vec)

    print(f"State vector size: {y0.size}")
    print(f"Expected: {1 + pk.lambda_i.size + 2*N_LUMPS + 3} = {1 + 6 + 2*3 + 3} = 16")

    # indices (matching rhs)
    N_DG    = pk.lambda_i.size
    start_T = 1 + N_DG
    idx_x   = start_T + 2*N_LUMPS
    idx_z   = idx_x + 1
    idx_Z   = idx_x + 2

    # ----- t=0 sanity print -----
    np.set_printoptions(precision=6, suppress=True)
    T_f0v  = y0[start_T : start_T + N_LUMPS]
    T_h0v  = y0[start_T + N_LUMPS : start_T + 2*N_LUMPS]
    x0, z0, zpow0 = y0[idx_x], y0[idx_z], y0[idx_Z]
    Tavg0_chk = float(np.dot(W_LUMP, 0.5*(T_h0v + Tc0)))
    print("y0 size:", y0.size, "N_DG:", N_DG, "N_LUMPS:", N_LUMPS)
    print("----- t=0 diagnostics (after init) -----")
    print(f"P0=1.000000, x0={x0:.6f}, z0={z0:.6f}, zpow0={zpow0:.6f}")
    print(f"Tc0={Tc0:.6f} K, Tavg0={Tavg0_chk:.6f} K, T_m0={fb.T_m0:.6f} K, T_f0={fb.T_f0:.6f} K")
    print("----------------------------------------")
    assert abs(Tavg0_chk - Tref(1.0, fb)) <= DB_K*0.5, "Non-zero PI error at t=0"

    print(f"At t=0: P_turb={P_turb_dem(0.0):.4f}, T_ref={Tref(1.0, fb):.2f} K")
    print(f"At t=50: P_turb={P_turb_dem(50.0):.4f}, T_ref={Tref(P_turb_dem(50.0), fb):.2f} K")
    print(f"Expected T_ref drop: {15.0 * (P_turb_dem(50.0) - 1.0):.2f} K")

    # integrate
    t0, tf = 0.0, 400.0
    t_eval = np.linspace(t0, tf, 2001)
    print(f"Integrating with Radau — Load reduction to {P_TURB_STEP_PU:.4f} pu at t={STEP_T_S} s")
    print(f"  → T_ref drops {15.0*(P_TURB_STEP_PU-1.0):.2f} K, T_cold rises ~4 K")
    P_turb_fn = lambda t: P_turb_dem(t)
    global _rhs_did_print_once
    _rhs_did_print_once = False
    sol = solve_ivp(
        fun=lambda tt, yy: rhs(tt, yy, pk, fb, worth, Tcold_fn, P_turb_fn, 'auto', 0.0),
        t_span=(t0, tf), y0=y0, method="Radau",
        t_eval=t_eval, atol=1e-9, rtol=1e-7, max_step=0.05
    )

    t = sol.t
    P = sol.y[0]
    T_f_mat = sol.y[start_T : start_T + N_LUMPS]
    T_h_mat = sol.y[start_T + N_LUMPS : start_T + 2*N_LUMPS]
    x       = np.clip(sol.y[idx_x], 0.0, 1.0)

    # averages & tracks
    T_f_avg  = np.einsum('k,kt->t', W_LUMP, T_f_mat)
    T_h_avg  = np.einsum('k,kt->t', W_LUMP, T_h_mat)
    Tc_track = np.array([float(Tcold_fn(tt)) for tt in t]) # tracks cold leg temperature over each time step
    T_avg    = 0.5*(T_h_avg + Tc_track)

    # reactivity values (pcm)
    alpha_D_K = fb.alpha_D_input * (9.0/5.0) if fb.coeff_units == "F" else fb.alpha_D_input
    alpha_M_K = fb.alpha_M_input * (9.0/5.0) if fb.coeff_units == "F" else fb.alpha_M_input
    worth_abs = np.vectorize(worth.rho_pcm_abs)
    rho_rod_pcm = worth_abs(x) - worth_abs(X_OP)
    rho_d_pcm   = alpha_D_K * (T_f_avg - fb.T_f0)
    rho_m_pcm   = alpha_M_K * (T_avg   - fb.T_m0)
    rho_total_pcm = rho_rod_pcm + rho_d_pcm + rho_m_pcm

    # reference track (P_turb=1.0)
    T_ref_track = np.array([Tref(1.0, fb) for _ in t]) # tracks the temperature controller setpoint over time
    P_turb_track = np.array([P_turb_dem(tt) for tt in t]) # tracks the turbine demand over time
    z_pow_track = sol.y[idx_Z] # tracks the outer loop bias over time

    # Print final values
    print(f"\n----- Final State (t={t[-1]:.1f}s) -----")
    print(f"Power: {P[-1]*P_RATED_MWT:.1f} MWt ({P[-1]*100:.1f}%)")
    print(f"Turbine demand: {P_turb_dem(t[-1])*P_RATED_MWT:.1f} MWt ({P_turb_dem(t[-1])*100:.1f}%)")
    print(f"T_avg: {T_avg[-1]:.2f} K, T_ref: {Tref(P_turb_dem(t[-1]), fb):.2f} K")
    print(f"z_pow bias: {z_pow_track[-1]:.3f} K")
    print(f"Rod position: {x[-1]*100:.2f}%")

    powers_MWt = P * P_RATED_MWT
    powers_MWe = powers_MWt * (P_RATED_MWE/P_RATED_MWT)

    # Plots
    fig, axes = plt.subplots(6, 1, figsize=(12, 12))
    axes[0].plot(t, powers_MWt, linewidth=2, label='Thermal Power (MWt)'); axes[0].set_ylabel('Thermal Power [MWt]'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'AP1000 Load Reduction ({DELTA_MWE:.0f} MWe) — Sliding Tavg Control (Radau)', fontweight='bold')
    axes[1].plot(t, P_turb_track * P_RATED_MWE, '--', linewidth=2, alpha=0.7, label='Turbine Demand')
    axes[1].plot(t, powers_MWe, linewidth=2, label='Electric Power (MWe)'); axes[1].set_ylabel('Electric Power [MWe]'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[2].plot(t, T_avg, linewidth=2, label='Core Avg T (=(Th+Tc)/2)')
    axes[2].plot(t, T_ref_track, '--', linewidth=2, label='T_ref(P_turb)'); axes[2].set_ylabel('Temperature [K]'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    axes[3].plot(t, T_h_avg, linewidth=1.8, label='Hot Leg (avg)')
    axes[3].plot(t, Tc_track, linewidth=1.2, linestyle='--', label='Cold Leg (input)'); axes[3].set_ylabel('Primary Temps [K]'); axes[3].legend(); axes[3].grid(True, alpha=0.3)
    axes[4].plot(t, x*100.0, linewidth=2); axes[4].set_ylabel('Rod Insertion [%]'); axes[4].grid(True, alpha=0.3)
    axes[5].plot(t, rho_rod_pcm, linewidth=2, label='Rod')
    axes[5].plot(t, rho_d_pcm, '--', linewidth=2, label='Doppler')
    axes[5].plot(t, rho_m_pcm, '--', linewidth=2, label='Moderator')
    axes[5].plot(t, rho_total_pcm, linewidth=2.5, alpha=0.8, label='Total')
    axes[5].axhline(0.0, color='gray', linestyle=':'); axes[5].set_ylabel('Reactivity [pcm]'); axes[5].set_xlabel('Time [s]')
    axes[5].legend(loc='center left', bbox_to_anchor=(1, 0.5)); axes[5].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reactor_core_plots.png", dpi=150)
    # Show only when using an interactive backend; always save for artifacts
    if matplotlib.get_backend().lower() not in ("agg",):
        plt.show()
    plt.close("all")
    print("Saved plots to reactor_core_plots.png")

    return t, T_h_avg, Tc_track, T_avg, P

def make_Tcold_fn(t_points, Tc_points, fill_value='extrapolate'):
    return interp1d(np.asarray(t_points, float),
                    np.asarray(Tc_points, float),
                    kind='linear', bounds_error=False, fill_value=fill_value)

# Time-stepping interface for external coupling with main.py
class ReactorSimulator:
    """
    Main.py will call step() repeatedly with externally-provided Tc_in (from SG) and dt.
    """
    
    def __init__(self, Tc_init=None, P_turb_init=1.0, control_mode='auto'):
        """
        Initialize reactor at equilibrium conditions.
        
        Args:
            Tc_init: Initial cold leg temperature [K], if None uses default HFP
            P_turb_init: Initial turbine power [p.u.], default 1.0
        """
        # Create initial reactor states
        self.pk = PKParams()
        self.fb = FBParams()
        self.worth = RodWorthAP1000LUT()
        
        # Calculate HFP references
        dT_fc_hfp = P_RATED_W / UA
        self.fb.T_m0 = 576.54
        self.fb.T_f0 = self.fb.T_m0 + dT_fc_hfp
        
        # Initial cold leg temp
        if Tc_init is None:
            dT_loop_hfp = P_RATED_W / (self.fb.W_kgps * self.fb.cp_JpkgK)
            Tavg0 = self.fb.T_m0
            Tc_init = Tavg0 - 0.5 * dT_loop_hfp
        
        self.Tc_current = Tc_init
        self.P_turb_current = P_turb_init
        
        # Calculate equilibrium hot leg temps per lump
        Q_core_k0 = 1.0 * P_RATED_W * W_LUMP
        Wcp_k = self.fb.W_kgps * self.fb.cp_JpkgK * W_LUMP
        Th_eq0_vec = Tc_init + Q_core_k0 / Wcp_k
        
        # Initialize state vector
        self.y = initial_state(self.pk, self.fb, Tc_init, Th_eq0_vec)
        
        # Store indices for extracting values
        N_DG = self.pk.lambda_i.size
        start_T = 1 + N_DG
        self.idx_P = 0
        self.idx_T_f_start = start_T
        self.idx_T_h_start = start_T + N_LUMPS
        self.idx_x = start_T + 2 * N_LUMPS
        self.idx_z = self.idx_x + 1
        self.idx_Z = self.idx_x + 2
        
        self.control_mode = control_mode  # 'auto' or 'manual'
        self.manual_rod_speed = 0.0  # For manual mode
        
        print(f"ReactorSimulator initialized: Tc={Tc_init:.2f} K, P_turb={P_turb_init:.3f} pu, Control Mode: {self.control_mode}")
    
    def step(self, Tc_in, dt, P_turb=None, manual_rod_cmd=None):
        """
        Advance reactor by dt seconds.
        
        Args:
            Tc_in: Cold leg temperature [K] for this timestep
            dt: Time step size [s] provided by main.py
            P_turb: Turbine power demand [p.u.], if None uses last value
            manual_rod_cmd: Manual rod speed command [-1.0, +1.0]
            
        Returns:
            Th_out: Hot leg temperature [K] (weighted average)
            P_out: Reactor thermal power [MWt]
        """
        # Update current values
        self.Tc_current = Tc_in
        if P_turb is not None:
            self.P_turb_current = P_turb
        
        if manual_rod_cmd is not None:
            self.control_mode = 'manual'
            self.manual_rod_speed = np.clip(manual_rod_cmd, -1.0, 1.0)
            
        # Create constant functions for the timestep
        Tcold_fn = lambda t: self.Tc_current
        P_turb_fn = lambda t: self.P_turb_current
        
        # Integrate forward by dt
        sol = solve_ivp(
            fun=lambda t, y: rhs(t, y, self.pk, self.fb, self.worth, 
                                Tcold_fn, P_turb_fn, self.control_mode, self.manual_rod_speed),
            t_span=(0.0, dt),
            y0=self.y,
            method="Radau",
            max_step=0.05,
            atol=1e-9,
            rtol=1e-7
        )
        
        # Update state
        self.y = sol.y[:, -1]
        
        # Extract outputs
        T_h_vec = self.y[self.idx_T_h_start : self.idx_T_h_start + N_LUMPS]
        Th_out = float(np.dot(W_LUMP, T_h_vec))  # Weighted average
        P_out = float(self.y[self.idx_P]) * P_RATED_MWT
        
        return Th_out, P_out
    
    def get_diagnostics(self):
        """
        Return detailed reactor state for logging/plotting.
        
        Returns:
            dict with current reactor state variables
        """
        T_f_vec = self.y[self.idx_T_f_start : self.idx_T_f_start + N_LUMPS]
        T_h_vec = self.y[self.idx_T_h_start : self.idx_T_h_start + N_LUMPS]
        
        return {
            'P_pu': float(self.y[self.idx_P]),
            'P_MWt': float(self.y[self.idx_P]) * P_RATED_MWT,
            'x_rods': float(self.y[self.idx_x]),
            'z_integral': float(self.y[self.idx_z]),
            'z_pow_bias': float(self.y[self.idx_Z]),
            'T_f_avg': float(np.dot(W_LUMP, T_f_vec)),
            'T_h_avg': float(np.dot(W_LUMP, T_h_vec)),
            'Tc_in': self.Tc_current,
            'P_turb': self.P_turb_current
        }
    
if __name__ == "__main__":
    RUN_TEST_1 = True   # Set to False to skip
    RUN_TEST_2 = True  # Set to False to ski[p
    
    if RUN_TEST_1:
        # Internal coee test 1: runs load cut (increased cold leg) simulation for defiend duration (tf defined above) with all 6 plots
        print("=" * 60)
        print("Test 1: Full batch simulation with run()")
        print("=" * 60)
        t, T_h_avg, Tc_track, T_avg, P = run() # output these values
    
    if RUN_TEST_2:
        # Intrnal code test 2: simulates time-step (0.1 seconds) for 10 seconds at steady-state and prints output variables every second 
        print("\n" + "=" * 60)
        print("Test 2: Time-stepping interface with ReactorSimulator")
        print("=" * 60)
        
        reactor = ReactorSimulator()
    
        # Simulate 10 seconds with constant inputs
        dt = 0.1
        for i in range(100):  # 100 steps × 0.1s = 10 seconds 
            t_sim = i * dt
            Tc_in = 555.78 # Kelvin, constant cold leg temp
            Th_out, P_out = reactor.step(Tc_in, dt)
            
            if i % 10 == 0:  # Print every second
                diag = reactor.get_diagnostics()
                print(f"t={t_sim:.1f}s: P={P_out:.1f} MWt, Th={Th_out:.1f} K, x={diag['x_rods']*100:.2f}%")
        
        print("\nBoth tests complete!")

