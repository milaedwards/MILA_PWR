# AP1000 Reactor Simulator Rev8 FIXED v3 (BOC–HFP, no xenon, no boron)
# - Full 6-group (Keepin U-235 dominant) point-kinetics scaled to beta-eff from AP1000 DCD Table 4.3-2 with Radau solver
# - AP1000 rod worth from DCD Figure "All rods inserting less most reactive stuck rod"
#   implemented as a normalized LUT + PCHIP spline, scaled to TOTAL WORTH = 10,490 pcm
#   (Table 4.3-3 item 3b: "All assemblies but one (highest worth inserted)").
# - BOC–HFP temperature feedbacks (fuel Doppler + moderator), xenon and boron ignored
# - Mann's lumped model (2 coolant lumps and 1 fuel lump)
# - Control Rod Drive Mechanics (CRDM) kinematics: 45 in/min over 166.755 in
# - Sliding Tavg program: T_ref(P_turb) = T_m0 + S_Tavg * (P_turb − 1)
# - NEW IN REV8: Transport delay ODEs for coolant transit times
#   * tau_rxi = 2.145s: Vessel inlet → Core inlet (Could not find accurate information to calculate.  Using Vajpayee's values)
#   * tau_rxu = 2.517s: Core exitt → Vessel exit
#   * Bypass flow (5.9%) at cold leg temp mixes in upper plenum, creating ~2.75K temp drop
# - Receives 4 inputs from main.py:
#    (1) cold leg temperature (Tc_in); (2) time step (dt); (3) turbine power demand in per-unit (P_turb); and (4) rod control mode (auto/manual)
#    and outputs hot leg temperature (Th_out) and reactor power (P_out)
# - Internal simulation #1: Tc rises (simulated at t=10 seconds), then Th and T_avg rise.
#       PI controller sees T_avg > T_ref (since T_ref is constant with turbine at 1.0).
#       Rods insert to reduce power, mimicking a load cut
# - Internal simulation #2: runs for 100 seconds at steady-state and prints output variables every second
# - Internal simulation #3: add 1 cent of reactivity (no rods)
# - Internal simulation #4: subtract 1 cent of reactivity (no rods)
# - Internal simulation #5: add 1 degree Fahrenheit to cold leg input temperature (no rods)
# - Internal simulation #6: subtract 1 degree Fahrenheit from cold leg input temperature (no rods)

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Global constants
P_RATED_MWT = 3400.0  # MWt, AP1000 reactor core heat output from AP1000 DCD Chapter 4, Table 4.4-1
P_RATED_MWE = 1117.0  # MWe, AP1000 nominal electric power at 33% eficiency
P_RATED_W = P_RATED_MWT * 1e6
PCM_TO_DK = 1e-5  # conversion factor pcm to delta_k/k
STROKE_IN = 166.755  # rod dimensions from AP1000 DCD Chapter 3, section 3.9.4.2.1
VMAX_IN_PER_MIN = 45.0  # maximum travel in inches per minute from AP1000 DCD Chapter 3, section 3.9.4.2.1
X_OP = 0.57  # 57% = initial rod insertion at BOC-HFP critical operations
U = 1143  # W/m²·K, fuel-to-coolant heat transfer coefficient, calculated using average from Kerlin (H.B. Robinson) and Masoud/Appendix J values for delta_T_f-c
A = 5267.6  # m², active heat transfer surface area from AP1000 DCD Chapter 4, Table 4.4-1
UA = U * A  # W/K
m_f = 95974.7  # kg UO2 from AP1000 DCD Chapter 4, Table 4.1-1
cp_f = 313  # J/kg-K specific heat of fuel from Table 4.3 of Oak Ridge National Labs paper: https://info.ornl.gov/sites/publications/Files/Pub57523.pdf
m_c = 11923.53  # kg coolant in core (calculated on 11/9/2025) m_c = (W*H)/v_c
cp_JpkgK = 5442  # J/kg-K, specific heat capcity of coolant was 5545 from enghandbook.  NEW calculated value from
W_kgps = 13456.6  # kg/s, effective mass flow rate of coolant in core (106.8 x 10^6 lbm/hr) from AP1000 DCD Chapter 4, Table 4.1-1
f = 0.974  # % heat generated in fuel from AP1000 DCD Chapter 4, Table 4.1-1
tau_f = (m_f * cp_f) / (UA)  # fuel-to-coolant heat transfer time constant
tau_c = (m_c * cp_JpkgK) / (UA)  # time constant  for fuel-to-coolant heat transfer
tau_r = m_c / W_kgps  # residence time of coolant in the core
tau_rxi = 2.145  # s  Tcold_in -> T_core_inlet (downcomer + lower plenum lag)  (using Vajpayee)
tau_rxu = 2.517  # s  Tc2 -> T_hot_leg (upper plenum hot-leg lag)  (using Vajpayee)
f_bypass = 0.059  # Upper plenum bypass flow fraction (5.9%) - cooler flow bypasses core and mixes in upper plenum
H_f = f * P_RATED_W / (m_f * cp_f)  # K/s per pu
H_c = (1.0 - f) * P_RATED_W / (m_c * cp_JpkgK)  # K/s per pu
RHO_STEP_PCM = 0.0  # for adding or subtracting cents of reactivity
_REACTIVITY_DEBUG_PRINTED = False


# Mann two-coolant-lump and one fuel lump core model
# Inputs: Tf, Tc1, Tc2, Tin (all in Kelvin);  P_pu (per-unit reactor power)
# calculate derivatives
# NOTE: Tin is now the DELAYED core inlet temperature (T_core_inlet), not the externally provided cold leg temperature
def thermal_derivs(Tf, Tc1, Tc2, Tin, P_pu):
    # Holbert's Eq. 2
    dTf = H_f * P_pu - (Tf - Tc1) / tau_f

    # Holbert's Eq. 5
    dTc1 = (H_c * P_pu + (Tf - Tc1) / tau_c
            - (2.0 / tau_r) * (Tc1 - Tin))

    # Holbert's Eq. 6
    dTc2 = (H_c * P_pu + (Tf - Tc1) / tau_c
            - (2.0 / tau_r) * (Tc2 - Tc1))

    return dTf, dTc1, dTc2


def step_thermal(dt, state, Tin, P_pu):
    Tf, Tc1, Tc2 = state
    # calculate derivatives at beginning of step
    k1 = thermal_derivs(Tf, Tc1, Tc2, Tin, P_pu)

    # advance half-step Euler midpoint estimate for 2nd order Runge-Kutta midpoint method
    Tf2 = Tf + 0.5 * dt * k1[0]
    Tc12 = Tc1 + 0.5 * dt * k1[1]
    Tc22 = Tc2 + 0.5 * dt * k1[2]

    # calculate derivatives at half-step
    k2 = thermal_derivs(Tf2, Tc12, Tc22, Tin, P_pu)

    # advance state using midpoint slope, k2
    Tf += dt * k2[0]
    Tc1 += dt * k2[1]
    Tc2 += dt * k2[2]
    return (Tf, Tc1, Tc2), Tc2  # Tc2 is the core outlet temperature (before upper plenum delay)


DELTA_MWE = -50.0  # MWe, for load cut scenario
STEP_T_S = 10.0  # seconds (time of the load cut or rod insertion) for internal simulation
P_TURB_STEP_PU = 1.0 + DELTA_MWE / P_RATED_MWE  # ≈ 0.9552 pu (50 MWe load reduction)


def set_rho_step(pcm: float):  # create variable for simulated reactivity increase/decrease
    global RHO_STEP_PCM
    RHO_STEP_PCM = float(pcm)  # fixed reactivity addition/subtraction at t=0


def x_rate_max() -> float:
    return (
            VMAX_IN_PER_MIN / 60.0) / STROKE_IN  # conversion of maximum stroke rate from per minute to per second [s^-1]


# Rod worth Lookup Table (LUT) data points generated with ChatGPT on 9/14/2025 with Figure 4.3-30 from AP1000 DCD
@dataclass
class RodWorthAP1000LUT:
    rho_tot_pcm: float = 10490.0  # Table 4.3-3 item 3b → 10,490 pcm (all-rods minus MOST worth stuck)
    x_pts: np.ndarray = field(default_factory=lambda: np.array(
        [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00], dtype=float))
    f_pts: np.ndarray = field(default_factory=lambda: np.array(
        [0.0000, 0.0001, 0.0003, 0.0020, 0.0055, 0.0200, 0.0600, 0.1200, 0.2400, 0.4600,
         0.7000, 0.9000, 0.9700, 1.0000], dtype=float))

    def __post_init__(
            self):  # use Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) function to create a spline from LUT values for insertion (x) and normalized worth (f)
        x = np.clip(self.x_pts, 0.0, 1.0).astype(float)
        f = np.clip(self.f_pts, 0.0, 1.0).astype(float)
        f[0], f[-1] = 0.0, 1.0
        self._s = PchipInterpolator(x, f,
                                    extrapolate=False)  # create spline, False prevents extrapolation beyond values in table
        self._ds = self._s.derivative()  # derivative of spline representing slope, not being used

    # insertion ⇒ NEGATIVE reactivity
    def rho_pcm_abs(self, x: float) -> float:
        x = np.clip(x, 0.0, 1.0)  # normalizes the rod insertion inches from 0 to 1.0
        return -float(self._s(x)) * self.rho_tot_pcm  #

    def rho_pcm_rel(self, x: float, x_ref: float = X_OP) -> float:
        return self.rho_pcm_abs(x) - self.rho_pcm_abs(
            x_ref)  # defining relative reactivity given the rod starting position X_0


# Point Kinetics Parameters from AP1000 DCD Table 4.3-2
@dataclass
class PKParams:
    Lambda: float = 19.8e-6  # seconds, Prompt neutron generation time from AP1000 DCD Chapter 4, Table 4.3-2
    # Keepin shape for U-235 due to U-235 dominating at BOC
    # U-235 delayed fractions and decay constants were provided in Table 3.1 from Kerlin-Upadhyaya textbook
    beta_shape: np.ndarray = field(default_factory=lambda: np.array(
        [0.000221, 0.001467, 0.001313, 0.002647, 0.000771, 0.000281], dtype=float))
    lambda_i: np.ndarray = field(default_factory=lambda: np.array(
        [0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01], dtype=float))
    beta_eff: float = 0.0075  # from AP1000 DCD Chapter 4, Table 4.3-2

    # Scale beta values to beta_eff from Ap1000
    def __post_init__(self):
        self.beta_i = self.beta_shape * (self.beta_eff / float(self.beta_shape.sum()))
        self.beta = float(self.beta_i.sum())


# Temperature Feedback Parameters from AP1000 DCD Table 4.3-2
@dataclass
class FBParams:
    # feedback coefficients in pcm/°F
    # moderator coefficient chosen to match Kerlin value from H.B. Robinson paper
    alpha_D_input: float = -1.4  # pcm/°F, Doppler
    alpha_M1_input: float = -14  # pcm/°F Moderator coefficient for inlet/avg coolant (70% weight of overall moderator temperature coefficient of -2e-4 delta_k/k/°F)
    alpha_M2_input: float = -6  # pcm/°F Moderator coefficient foroutlet coolant (smaller effect ~ 30% weight of overall moderator temperature coefficient of -2e-4 delta_k/k/°F)
    coeff_units: str = "F"

    # Reference temps (Kelvin) at HFP with bypass flow
    T_f0: float = 1128.0  # Fuel temperature at steady state with bypass flow
    T_m0: float = 576.54  # Kelvin, average coolant temperature (measured with bypass flow effect)
    Tc10: float = 577.91  # Kelvin, first coolant lump temperature at steady state
    Tc20: float = 601.12  # Kelvin, second coolant lump temperature at steady state


# Sliding T_avg program ----------------
P_TURB_INIT = 1.0  # normalized turbine power at t=0

# Programmed T_avg reference vs turbine load (linear "sliding" schedule)
# Modest slope (order 10–20 K per p.u.) so Tref changes a few kelvin for a few-% load step.
S_TAVG_K_PER_PU = 19.0  # specific slope


# Turbine demand at time t
def P_turb_dem(t: float) -> float:
    return 1.0 if t < STEP_T_S else P_TURB_STEP_PU


# Reference moderator temperature setpoint calculation based on turbine load
def Tref(P_turb: float, fb: FBParams) -> float:
    return fb.T_m0 + S_TAVG_K_PER_PU * (P_turb - 1.0)


# Temperature deadband function (from Rev 7)
def _deadband(e: float, db: float) -> float:
    """
    Apply deadband to suppress errors smaller than threshold.
    Prevents rod hunting due to noise and small fluctuations.
    """
    if e > db:
        return e - db  # Error above positive deadband
    elif e < -db:
        return e + db  # Error below negative deadband
    else:
        return 0.0  # Within deadband, return zero


@dataclass
class ControlParams:
    # PI Controller Gains
    Kp: float = 0.003  # proportional gain
    Ki: float = 0.0  # integral gain

    # Power Tracking Controller
    Kpow_i: float = 0.001  # power gain

    # Feedforward Compensation
    Kff: float = 0.0  # feedforward gain

    # Stability Features
    K_AW: float = 2.0  # Anti-windup back-calculation gain
    LEAK: float = 0.10  # Inner loop integral leak [1/s] - high for stability
    LEAK_OUTER: float = 0.10  # Outer loop integral leak [1/s] - high for stability
    DB_K: float = 0.3  # Temperature deadband [K] - large to prevent hunting
    P_DEADBAND: float = 0.04  # Power deadband [pu] (4%)

    # Rate and Position Limits
    U_MAX: float = 0.04  # Maximum controller output [1/s] - very restricted
    x_rate_max: float = (VMAX_IN_PER_MIN / 60.0) / STROKE_IN  # max rod speed [1/s]
    x_min: float = 0.0  # fully withdrawn
    x_max: float = 1.0  # fully inserted
    Z_BIAS_MAX: float = 0.5  # Maximum power bias [K] - very tight


class ReactorSimulator:
    """
    Reactor simulator with transport delays:
    - Tc_in (cold leg) → T_core_inlet [tau_rxi] → core thermal model
    - Tc2 (core outlet) → T_hot_leg [tau_rxu] → Th_out (hot leg)
    """

    def __init__(self, Tc_init: float = None, P_turb_init: float = 1.0, control_mode: str = "auto"):
        """
        Initialize the reactor simulator.

        Args:
            Tc_init: Initial cold leg temperature [K] (if None, calculated from equilibrium)
            P_turb_init: Initial turbine power demand [pu]
            control_mode: "auto" or "manual"
        """
        self.pk = PKParams()
        self.fb = FBParams()
        self.ctrl = ControlParams()
        self.rod_worth = RodWorthAP1000LUT()

        # Control mode
        self.control_mode = control_mode

        # Manual rod command (rod speed in fraction of stroke per second).
        # Used only in manual mode.
        self.manual_u = 0.0

        # Calculate equilibrium temperatures if not provided
        if Tc_init is None:
            T_avg_ref = self.fb.T_m0
            dT_core_HFP = P_RATED_W / (W_kgps * cp_JpkgK)
            # Account for bypass flow reducing measured temperature rise
            Tc_init = T_avg_ref - 0.5 * dT_core_HFP * (1 - f_bypass)

        self.Tc_external = Tc_init  # Store for reference

        # Initialize state variables
        # There are 5 temperature states instead of 3
        # Tf, Tc1, Tc2 (core thermal model) + T_core_inlet, T_hot_leg (transport delays)
        self.Tf = self.fb.T_f0
        self.Tc1 = self.fb.Tc10
        self.Tc2 = self.fb.Tc20
        self.T_core_inlet = Tc_init  # Initialize to cold leg temperature (no initial delay)
        # Initialize T_hot_leg with bypass flow mixing
        self.T_hot_leg = (1 - f_bypass) * self.fb.Tc20 + f_bypass * Tc_init

        # Point kinetics state
        self.n = 1.0  # normalized neutron population
        self.C = self.pk.beta_i / (self.pk.Lambda * self.pk.lambda_i)  # precursor concentrations

        # Control state
        self.x = X_OP  # rod position
        self.z_integral = 0.0  # integral error accumulator
        self.z_pow_bias = 0.0  # power bias integral
        self.P_turb = P_turb_init
        self._P_turb_prev = P_turb_init  # For feedforward calculation

        # Manual rod command (rod speed in 1/s, used only in manual mode)
        self.manual_u = 0.0

        # For diagnostics
        self.t = 0.0
        self.P_pu = 1.0
        # Total reactivity (delta k / k) for diagnostics
        self.rho_dk = 0.0

    def _compute_derivs(self, t, y, Tc_in_external, P_turb):
        """
        Compute derivatives for the full reactor system including transport delays.

        State vector y:
        [0:6]   : C_i (precursor concentrations)
        [6]     : n (normalized neutron population)
        [7]     : Tf (fuel temperature)
        [8]     : Tc1 (first coolant lump)
        [9]     : Tc2 (second coolant lump)
        [10]    : T_core_inlet (delayed inlet temperature)
        [11]    : T_hot_leg (delayed hot leg temperature)
        [12]    : x (rod position)
        [13]    : z_integral (integral error)
        [14]    : z_pow_bias (power bias integral)
        """
        # Extract state
        C = y[0:6]
        n = y[6]
        Tf = y[7]
        Tc1 = y[8]
        Tc2 = y[9]
        T_core_inlet = y[10]  # This is what enters the core
        T_hot_leg = y[11]  # This is what exits to hot leg
        x = y[12]
        z_integral = y[13]
        z_pow_bias = y[14]

        # Power
        P_pu = n

        # Calculate reactivity feedbacks using core coolant temperatures
        dT_f = Tf - self.fb.T_f0
        dT_m1 = Tc1 - self.fb.Tc10  # First coolant lump feedback
        dT_m2 = Tc2 - self.fb.Tc20  # Second coolant lump feedback

        # Convert to Fahrenheit for feedback coefficients
        dT_f_F = dT_f * 9.0 / 5.0
        dT_m1_F = dT_m1 * 9.0 / 5.0
        dT_m2_F = dT_m2 * 9.0 / 5.0

        # Reactivity components
        rho_fb_pcm = (self.fb.alpha_D_input * dT_f_F +
                      self.fb.alpha_M1_input * dT_m1_F +
                      self.fb.alpha_M2_input * dT_m2_F)

        rho_rod_pcm = self.rod_worth.rho_pcm_rel(x, X_OP)
        rho_ext_pcm = RHO_STEP_PCM
        rho_tot_pcm = rho_fb_pcm + rho_rod_pcm + rho_ext_pcm
        rho_tot = rho_tot_pcm * PCM_TO_DK
        # Store total reactivity for diagnostics / ICSystem coupling
        self.rho_dk = rho_tot

        # Point kinetics
        rho_minus_beta = rho_tot - self.pk.beta
        dn_dt = (rho_minus_beta / self.pk.Lambda) * n + np.sum(self.pk.lambda_i * C)
        dC_dt = (self.pk.beta_i / self.pk.Lambda) * n - self.pk.lambda_i * C

        # Thermal model using DELAYED core inlet temperature
        dTf, dTc1, dTc2 = thermal_derivs(Tf, Tc1, Tc2, T_core_inlet, P_pu)

        # Transport delay ODEs
        # Inlet delay: cold leg → core inlet
        dT_core_inlet_dt = (Tc_in_external - T_core_inlet) / tau_rxi

        # Upper plenum delay with bypass flow mixing:
        # Core outlet (Tc2) mixes with bypass flow at cold leg temperature
        # This models the ~6.6°F temperature drop observed in AP1000 DCD
        T_upper_mixed = (1 - f_bypass) * Tc2 + f_bypass * Tc_in_external
        dT_hot_leg_dt = (T_upper_mixed - T_hot_leg) / tau_rxu

        # CONTROL SYSTEM
        if self.control_mode == "auto":
            # Calculate average coolant temperature using measured temperatures (cold leg input and hot leg output, as would be measured by RTDs)
            T_avg = 0.5 * (T_core_inlet + T_hot_leg)
            T_ref = Tref(P_turb, self.fb)

            # Temperature Error with Deadband
            e_temp_raw = T_avg - T_ref
            e_temp = _deadband(e_temp_raw, self.ctrl.DB_K)

            # Power Error with Deadband
            P_ref = P_turb  # Power should follow turbine load
            e_pow_raw = P_ref - P_pu
            # Only react to power errors larger than deadband
            e_pow = e_pow_raw if abs(e_pow_raw) > self.ctrl.P_DEADBAND else 0.0

            # Feedforward Compensation for load changes
            if not hasattr(self, '_P_turb_prev'):
                self._P_turb_prev = P_turb
            dP_turb = P_turb - self._P_turb_prev
            self._P_turb_prev = P_turb
            u_ff = self.ctrl.Kff * dP_turb

            # Outer Loop: Power Tracking Bias
            # Clamp power bias to prevent excessive temperature adjustments
            z_pow_clamped = 0.0  # np.clip(z_pow_bias, -self.ctrl.Z_BIAS_MAX, self.ctrl.Z_BIAS_MAX)

            # Apply power bias to temperature setpoint (slowly adjusts Tref based on power error)
            T_ref_adjusted = T_ref + z_pow_clamped

            # Recalculate temperature error with adjusted setpoint
            e_temp_with_bias = _deadband(T_avg - T_ref_adjusted, self.ctrl.DB_K)

            # Inner Loop: PI Control with Anti-Windup
            # Calculate unsaturated PI output
            u_pi_unsaturated = self.ctrl.Kp * e_temp_with_bias + self.ctrl.Ki * z_integral

            # Apply Rate Limiting to max rod travel
            u_total_unsaturated = u_pi_unsaturated + u_ff
            u_total = np.clip(u_total_unsaturated, -self.ctrl.U_MAX, self.ctrl.U_MAX)
            dx_dt = u_total

            # Conditional Integration with Anti-Windup if pushing into saturation
            pushing_saturation = (abs(u_total) >= self.ctrl.U_MAX and
                                  np.sign(u_total) == np.sign(u_total_unsaturated))

            if pushing_saturation:
                # Anti-windup: unwind integral when saturated
                dz_integral_dt = ((u_total - u_total_unsaturated) / self.ctrl.K_AW -
                                  self.ctrl.LEAK * z_integral)
            else:
                # Normal integration: error + back-calculation + leak
                dz_integral_dt = (e_temp_with_bias +
                                  (u_total - u_total_unsaturated) / self.ctrl.K_AW -
                                  self.ctrl.LEAK * z_integral)

            # Outer Loop Power Bias Integral
            # Check if power bias is saturated
            if abs(z_pow_clamped) >= self.ctrl.Z_BIAS_MAX and np.sign(z_pow_bias) == np.sign(e_pow):
                # Stop integrating if saturated and pushing further
                dz_pow_bias_dt = 0.0
            else:
                # Normal integration with leak
                dz_pow_bias_dt = 0.0  # self.ctrl.Kpow_i * e_pow - self.ctrl.LEAK_OUTER * z_pow_bias

        else:
            # Manual Rod Mode ignores controllers and uses externally commanded rod speed, but within physical rod limits
            u_manual = np.clip(self.manual_u,
                               -self.ctrl.x_rate_max,
                               self.ctrl.x_rate_max)
            dx_dt = u_manual

            # Let integrals wind down toward zero while in manual
            dz_integral_dt = -self.ctrl.LEAK * z_integral
            dz_pow_bias_dt = 0.0  # -self.ctrl.LEAK_OUTER * z_pow_bias

        # Assemble derivative vector
        dydt = np.zeros(15)
        dydt[0:6] = dC_dt
        dydt[6] = dn_dt
        dydt[7] = dTf
        dydt[8] = dTc1
        dydt[9] = dTc2
        dydt[10] = dT_core_inlet_dt
        dydt[11] = dT_hot_leg_dt
        dydt[12] = dx_dt
        dydt[13] = dz_integral_dt
        dydt[14] = dz_pow_bias_dt

        return dydt

    def step(
        self,
        Tc_in: float,
        dt: float,
        P_turb: float = 1.0,
        control_mode: str | None = None,
        manual_u: float = 0.0,
    ) -> tuple[float, float]:
        """
        Advance reactor simulation by one time step.

        Args:
            Tc_in: Cold leg temperature input [K] (external, before inlet delay)
            dt: Time step [s]
            P_turb: Turbine power demand [pu]
            control_mode: Override control mode for this step (optional)
            manual_u: manual rod speed command
        Returns:
            Th_out: Hot leg temperature output [K] (after upper plenum delay)
            P_out: Reactor power output [MWt]
        """
        if control_mode is not None:
            self.control_mode = control_mode

        # Store current manual rod speed command (used only in manual mode)
        self.manual_u = manual_u

        # Store for diagnostics
        self.Tc_in = Tc_in

        # Store turbine power for feedforward calculation
        self.P_turb = P_turb

        # Assemble state vector
        y0 = np.zeros(15)
        y0[0:6] = self.C
        y0[6] = self.n
        y0[7] = self.Tf
        y0[8] = self.Tc1
        y0[9] = self.Tc2
        y0[10] = self.T_core_inlet
        y0[11] = self.T_hot_leg
        y0[12] = self.x
        y0[13] = self.z_integral
        y0[14] = self.z_pow_bias

        # Integrate
        t_span = [self.t, self.t + dt]
        sol = solve_ivp(
            lambda t, y: self._compute_derivs(t, y, Tc_in, P_turb),
            t_span,
            y0,
            method='Radau',
            rtol=1e-6,
            atol=1e-8
        )

        # Extract final state
        y_final = sol.y[:, -1]
        self.C = y_final[0:6]
        self.n = y_final[6]
        self.Tf = y_final[7]
        self.Tc1 = y_final[8]
        self.Tc2 = y_final[9]
        self.T_core_inlet = y_final[10]
        self.T_hot_leg = y_final[11]
        self.x = np.clip(y_final[12], self.ctrl.x_min, self.ctrl.x_max)
        self.z_integral = y_final[13]
        self.z_pow_bias = y_final[14]

        # Update time
        self.t += dt

        # Calculate outputs
        self.P_pu = self.n
        P_out_MWt = self.P_pu * P_RATED_MWT
        Th_out = self.T_hot_leg  # Return DELAYED hot leg temperature

        return Th_out, P_out_MWt

    def get_diagnostics(self):
        """Get diagnostic information about current reactor state."""
        # T_avg as measured by plant instrumentation (cold leg + hot leg "RTDs")
        T_avg = 0.5 * (self.T_core_inlet + self.T_hot_leg) if hasattr(self, 'Tc_in') else 0.5 * (
                    self.T_core_inlet + self.Tc2)

        return {
            't': self.t,
            'P_MWt': self.P_pu * P_RATED_MWT,
            'P_pu': self.P_pu,
            'n': self.n,
            'x_rods': self.x,
            'T_f': self.Tf,
            'Tf': self.Tf,
            'Tc1': self.Tc1,
            'Tc2': self.Tc2,
            'T_c2': self.Tc2,
            'T_core_inlet': self.T_core_inlet,
            'T_hot_leg': self.T_hot_leg,
            'T_avg': T_avg,
            'z_integral': self.z_integral,
            'z_pow_bias': self.z_pow_bias
        }


# def run_four_scenarios(): # not used by main
# """
# Run four reactivity/temperature perturbation scenarios.
# All scenarios start from steady-state with no control rods active.
# """

# Calculate correct steady-state temperatures with bypass flow
# T_avg_ref = 576.54  # K
# dT_core_HFP = P_RATED_W / (W_kgps * cp_JpkgK)
# Tc_inlet = T_avg_ref - 0.5 * dT_core_HFP * (1 - f_bypass)

# scenarios = {
# 'plus_1_cent': {
# 'title': '+1 cent reactivity',
# 'rho_step_pcm': 7.5,  # +1 cent = +0.01 * 750 pcm
# 'dTc_F': 0.0
# },
# 'minus_1_cent': {
# 'title': '-1 cent reactivity',
# 'rho_step_pcm': -7.5,  # -1 cent
# 'dTc_F': 0.0
# },
# 'plus_1F': {
# 'title': '+1°F cold leg temperature',
# 'rho_step_pcm': 0.0,
# 'dTc_F': 1.0
# },
# 'minus_1F': {
# 'title': '-1°F cold leg temperature',
# 'rho_step_pcm': 0.0,
# 'dTc_F': -1.0
# }
# }

# results = {}

# for name, config in scenarios.items():
# print(f"\n{'='*60}")
# print(f"Running scenario: {config['title']}")
# print(f"{'='*60}")

# Set reactivity step
# set_rho_step(config['rho_step_pcm'])

# Calculate input temperature
# dTc_K = config['dTc_F'] * 5.0/9.0
# Tc_in = Tc_inlet + dTc_K

# print(f"Reactivity step: {config['rho_step_pcm']:.2f} pcm")
# print(f"Cold leg temp: {Tc_in:.2f} K (ΔT = {config['dTc_F']:.1f}°F)")

# Initialize reactor in manual mode (no control)
# reactor = ReactorSimulator(Tc_init=Tc_inlet, control_mode="manual")

# Simulation parameters
# t_end = 60.0
# dt = 0.05
# n_steps = int(t_end / dt)

# Storage
# times = []
# P_MWt = []
# T_avg = []
# T_f = []
# T_c2 = []
# T_core_inlet_log = []
# T_hot_leg_log = []

# Run simulation
# t = 0.0
# for i in range(n_steps):
# Th_out, P_out = reactor.step(Tc_in, dt, P_turb=1.0, control_mode="manual", manual_u=0.0)
# diag = reactor.get_diagnostics()

# times.append(t)
# P_MWt.append(diag['P_MWt'])
# T_avg.append(diag['T_avg'])
# T_f.append(diag['Tf'])
# T_c2.append(diag['Tc2'])
# T_core_inlet_log.append(diag['T_core_inlet'])
# T_hot_leg_log.append(diag['T_hot_leg'])

# t += dt

# Store results
# results[name] = {
# 't': np.array(times),
# 'P_MWt': np.array(P_MWt),
# 'T_avg': np.array(T_avg),
# 'T_f': np.array(T_f),
# 'T_c2': np.array(T_c2),
# 'T_core_inlet': np.array(T_core_inlet_log),
# 'T_hot_leg': np.array(T_hot_leg_log),
# 'config': config
# }

# print(f"Final power: {P_MWt[-1]:.1f} MWt")
# print(f"Final T_avg: {T_avg[-1]:.2f} K")

# return results

# def plot_four_scenarios(results): # not used by main
# """Plot the four reactivity/temperature scenarios."""

# results is a dict: {scenario_name: scenario_result_dict}
# for name, res in results.items():
# t      = res["t"]
# Pmw    = res["P_MWt"]
# Tf     = res["T_f"]
# Tavg   = res["T_avg"]

# Tf_plot   = Tf
# Tavg_plot = Tavg
# tlabel = "Temperature (K)"

# Get a nice title from config; fall back to the dict key if needed
# title = res.get("config", {}).get("title", name)

# fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
# fig.suptitle(title, fontsize=12, fontweight="bold")

# Power
# ax = axes[0]
# ax.plot(t, Pmw)
# ax.set_title("Reactor Power")
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Power (MWt)")
# ax.grid(True, alpha=0.3)

# Fuel temperature
# ax = axes[1]
# ax.plot(t, Tf_plot)
# ax.set_title("Fuel Temperature (K)")
# ax.set_xlabel("Time (s)")
# ax.set_ylabel(tlabel)
# ax.grid(True, alpha=0.3)

# Coolant temperature
# ax = axes[2]
# ax.plot(t, Tavg_plot)
# ax.set_title("Average Coolant Temperature (K)")
# ax.set_xlabel("Time (s)")
# ax.set_ylabel(tlabel)
# ax.grid(True, alpha=0.3)

# plt.show()


def run_load_cut_simulation():  # shouldn't be run my main during actual simualtion with other systems
    """
    Run the load cut simulation where turbine demand drops at t=10s
    """
    print("Running load cut simulation...")
    print(f"Load change: {DELTA_MWE} MWe at t={STEP_T_S}s")
    print(f"New turbine power: {P_TURB_STEP_PU:.4f} pu")

    # Calculate correct inlet temperature for control equilibrium with bypass flow
    T_avg_ref = 576.54  # K, from FBParams T_m0
    dT_core_HFP = (P_RATED_W) / (W_kgps * cp_JpkgK)  # Temperature rise through core

    # With bypass flow, the measured loop temperature rise is reduced:
    # T_avg = Tc_in + 0.5*dT_core*(1 - f_bypass)
    # Solving for Tc_in:
    Tc_inlet = T_avg_ref - 0.5 * dT_core_HFP * (1 - f_bypass)

    print(f"Calculated Tc_inlet: {Tc_inlet:.2f} K")
    print(f"Core dT at HFP: {dT_core_HFP:.2f} K")
    print(f"Measured loop dT (with bypass): {dT_core_HFP * (1 - f_bypass):.2f} K")

    # Initialize reactor
    reactor = ReactorSimulator(Tc_init=Tc_inlet, P_turb_init=1.0, control_mode="auto")

    # Simulation parameters
    t_end = 200.0  # seconds
    dt = 0.05
    n_steps = int(t_end / dt)

    # Storage for results
    times = []
    P_MWt = []
    x_rods = []
    T_avg = []
    T_f = []
    T_c2 = []
    T_core_inlet_log = []
    T_hot_leg_log = []
    P_turb_log = []
    z_integral = []
    z_pow_bias = []

    # Time-stepping loop
    t = 0.0
    for i in range(n_steps):
        # Determine turbine power demand based on time
        P_turb = 1.0 if t < STEP_T_S else P_TURB_STEP_PU

        # Constant cold leg temperature
        Tc_in = Tc_inlet
        # Step the reactor
        Th_out, P_out = reactor.step(Tc_in, dt, P_turb=P_turb, manual_u=0.0)

        # Get diagnostics
        diag = reactor.get_diagnostics()

        # Store results
        times.append(t)
        P_MWt.append(diag['P_MWt'])
        x_rods.append(diag['x_rods'])
        T_avg.append(diag['T_avg'])
        T_f.append(diag['T_f'])
        T_c2.append(diag['Tc2'])
        T_core_inlet_log.append(diag['T_core_inlet'])
        T_hot_leg_log.append(diag['T_hot_leg'])
        P_turb_log.append(P_turb)
        z_integral.append(diag['z_integral'])
        z_pow_bias.append(diag['z_pow_bias'])

        # Print progress every 10 seconds
        if i % int(10.0 / dt) == 0:
            print(f"t={t:.1f}s: P={P_out:.1f} MWt, T_avg={diag['T_avg']:.2f} K, "
                  f"x={diag['x_rods'] * 100:.2f}%, P_turb={P_turb:.4f} pu")

        t += dt

    # Convert to numpy arrays
    results = {
        't': np.array(times),
        'P_MWt': np.array(P_MWt),
        'x_rods': np.array(x_rods),
        'T_avg': np.array(T_avg),
        'T_f': np.array(T_f),
        'T_c2': np.array(T_c2),
        'T_core_inlet': np.array(T_core_inlet_log),
        'T_hot_leg': np.array(T_hot_leg_log),
        'P_turb': np.array(P_turb_log),
        'z_integral': np.array(z_integral),
        'z_pow_bias': np.array(z_pow_bias)
    }

    # Plot results
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle(f'Rev 8: {DELTA_MWE} MWe reduction at t={STEP_T_S}s',
                 fontsize=14, fontweight='bold')

    # Power
    axes[0, 0].plot(results['t'], results['P_MWt'])
    axes[0, 0].set_ylabel('Power (MWt)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_title('Reactor Power')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=STEP_T_S, color='r', linestyle='--', alpha=0.5, label='Load cut')
    axes[0, 0].legend()

    # Rod position
    axes[0, 1].plot(results['t'], results['x_rods'] * 100)
    axes[0, 1].set_ylabel('Rod Position (%)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_title('Control Rod Position')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=STEP_T_S, color='r', linestyle='--', alpha=0.5)

    # Average coolant temperature
    axes[1, 0].plot(results['t'], results['T_avg'])
    axes[1, 0].set_ylabel('Temperature (K)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_title('Average Coolant Temperature')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=STEP_T_S, color='r', linestyle='--', alpha=0.5)

    # Fuel temperature
    axes[1, 1].plot(results['t'], results['T_f'])
    axes[1, 1].set_ylabel('Temperature (K)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_title('Fuel Temperature')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=STEP_T_S, color='r', linestyle='--', alpha=0.5)

    # Core outlet vs Hot leg (showing transport delay)
    axes[2, 0].plot(results['t'], results['T_c2'], label='Core Outlet (Tc2)', linewidth=2)
    axes[2, 0].plot(results['t'], results['T_hot_leg'], label='Hot Leg (delayed)', linewidth=2, linestyle='--')
    axes[2, 0].set_ylabel('Temperature (K)')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_title('Hot Leg Transport Delay (tau_rxu = 2.2s)')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].axvline(x=STEP_T_S, color='r', linestyle='--', alpha=0.5)
    axes[2, 0].legend()

    # Core inlet (showing transport delay)
    axes[2, 1].plot(results['t'], results['T_core_inlet'], label='Core Inlet (delayed)')
    axes[2, 1].set_ylabel('Temperature (K)')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_title('Core Inlet Transport Delay (tau_rxi = 2.4s)')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axvline(x=STEP_T_S, color='r', linestyle='--', alpha=0.5)
    axes[2, 1].legend()

    # Turbine demand
    axes[3, 0].plot(results['t'], results['P_turb'])
    axes[3, 0].set_ylabel('Turbine Demand (pu)')
    axes[3, 0].set_xlabel('Time (s)')
    axes[3, 0].set_title('Turbine Power Demand')
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].axvline(x=STEP_T_S, color='r', linestyle='--', alpha=0.5)
    axes[3, 0].set_ylim([0.94, 1.02])

    # Control integrals
    axes[3, 1].plot(results['t'], results['z_integral'], label='Temperature integral')
    axes[3, 1].plot(results['t'], results['z_pow_bias'], label='Power bias')
    axes[3, 1].set_ylabel('Integral State')
    axes[3, 1].set_xlabel('Time (s)')
    axes[3, 1].set_title('Control Integrals')
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].axvline(x=STEP_T_S, color='r', linestyle='--', alpha=0.5)
    axes[3, 1].legend()

    plt.tight_layout()
    plt.show()

    print("\nLoad cut simulation complete!")
    return results


if __name__ == "__main__":
    RUN_TEST_1 = False  # Load cut simulation
    RUN_TEST_2 = False  # Time-stepping test

    if RUN_TEST_1:
        print("=" * 60)
        print("Test 1: Load Cut Simulation with Transport Delays")
        print("=" * 60)
        results = run_load_cut_simulation()

    if RUN_TEST_2:
        print("\n" + "=" * 60)
        print("Test 2: Time-stepping interface with ReactorSimulator")
        print("=" * 60)

        reactor = ReactorSimulator()

        dt = 0.1
        for i in range(1000):
            t_sim = i * dt
            T_avg_ref = 576.54
            dT_core_HFP = P_RATED_W / (W_kgps * cp_JpkgK)
            Tc_in = T_avg_ref - 0.5 * dT_core_HFP * (1 - f_bypass)
            Th_out, P_out = reactor.step(Tc_in, dt)

            if i % 10 == 0:
                diag = reactor.get_diagnostics()
                print(f"t={t_sim:.1f}s: P={P_out:.1f} MWt, Th={Th_out:.1f} K, x={diag['x_rods'] * 100:.2f}%")

        print("\nBoth tests complete!")

    # Commenting out 4 scenarios internal test lines below
    # results = run_four_scenarios()
    # plot_four_scenarios(results)

