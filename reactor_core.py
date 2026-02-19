import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy.interpolate import PchipInterpolator
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from config import Config

cfg = Config()

P_RATED_MWT = cfg.P_core_nom_MWt  # MWt, AP1000 reactor core heat output from AP1000 DCD Chapter 4, Table 4.4-1
P_RATED_MWE = cfg.P_e_nom_MWe  # MWe, AP1000 nominal electric power at 33% eficiency
P_RATED_W = P_RATED_MWT * 1e6
PCM_TO_DK = 1e-5  # conversion factor pcm to delta_k/k

STROKE_IN = cfg.stroke_in  # rod dimensions from AP1000 DCD Chapter 3, section 3.9.4.2.1
VMAX_IN_PER_MIN = cfg.vmax_rod_in_per_min  # maximum travel in inches per minute from AP1000 DCD Chapter 3, section 3.9.4.2.1
X_OP = cfg.rod_insert_nom  # 57% = initial rod insertion at BOC-HFP critical operations
U = cfg.U_fc_W_m2K  # W/m²·K, fuel-to-coolant heat transfer coefficient, calculated using average from Kerlin (H.B. Robinson) and Masoud/Appendix J values for delta_T_f-c
A = cfg.A_fc_m2  # m², active heat transfer surface area from AP1000 DCD Chapter 4, Table 4.4-1
UA = cfg.U_fc_W_m2K * cfg.A_fc_m2  # W/K
m_f = cfg.m_fuel_core_kg  # kg UO2 from AP1000 DCD Chapter 4, Table 4.1-1
cp_f = cfg.cp_fuel_core_J_kgK  # J/kg-K specific heat of fuel from Table 4.3 of Oak Ridge National Labs paper: https://info.ornl.gov/sites/publications/Files/Pub57523.pdf
m_c = cfg.m_coolant_core_kg  # kg coolant in core (calculated on 11/9/2025) m_c = (W*H)/v_c
cp_JpkgK = cfg.cp_coolant_core_J_kgK  # J/kg-K, specific heat capcity of coolant was 5545 from enghandbook.  NEW calculated value from
W_kgps = cfg.m_dot_core_kg_s  # kg/s, effective mass flow rate of coolant in core (113.3 x 10^6 lbm/hr) from AP1000 DCD Chapter 5, Table 5.1-3
f = cfg.f_fuel_fraction  # % heat generated in fuel from AP1000 DCD Chapter 4, Table 4.1-1
tau_f = cfg.tau_f  # fuel-to-coolant heat transfer time constant
tau_c = cfg.tau_c  # time constant  for fuel-to-coolant heat transfer
tau_r = cfg.tau_r  # residence time of coolant in the core
tau_rxi = cfg.tau_rxi_s  # inlet → core delay [s]
tau_rxu = cfg.tau_rxu_s  # core → outlet delay [s]

# Bypass fraction
f_bypass = cfg.bypass_fraction  # upper-plenum bypass fraction [-]
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


def thermal_steady_state(Tin: float, P_pu: float):
    """
    Solve dTf=dTc1=dTc2=0 for the Holbert 3-state core thermal model.
    Returns (Tf, Tc1, Tc2).
    """
    # From dTf=0: Tf - Tc1 = H_f * P_pu * tau_f
    a = H_f * P_pu * tau_f

    # The term (H_c*P + (Tf-Tc1)/tau_c) appears in both Tc eqns
    qterm = H_c * P_pu + a / tau_c

    # From dTc1=0: (2/tau_r)(Tc1 - Tin) = qterm
    b = (tau_r / 2.0) * qterm

    Tc1 = Tin + b
    Tc2 = Tin + 2.0 * b
    Tf = Tc1 + a
    return Tf, Tc1, Tc2


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


DELTA_MWE = 0.0  # MWe, for load cut scenario
STEP_T_S = 0.0  # seconds (time of the load cut or rod insertion) for internal simulation
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
    Lambda: float = cfg.Lambda_prompt_s
    beta_shape: np.ndarray = field(
        default_factory=lambda: np.array(cfg.beta_shape_tuple, dtype=float)
    )
    lambda_i: np.ndarray = field(
        default_factory=lambda: np.array(cfg.lambda_i_tuple, dtype=float)
    )
    beta_eff: float = cfg.beta_eff

    def __post_init__(self):
        # Scale beta_i to sum to beta_eff
        self.beta_i = self.beta_shape * (self.beta_eff / float(self.beta_shape.sum()))
        self.beta = float(self.beta_i.sum())


# Temperature Feedback Parameters from AP1000 DCD Table 4.3-2
@dataclass
class FBParams:
    # Feedback coefficients in pcm/°F, from Config
    alpha_D_input: float = cfg.alpha_D_pcm_per_F
    alpha_M_input: float = cfg.alpha_M_pcm_per_F
    coeff_units: str = "F"

    # Reference temps (Kelvin) at HFP with bypass flow
    # These are specific to this core model, so it's fine to leave them here.
    T_f0: float = cfg.T_f0
    T_m0: float = cfg.T_m0
    Tc10: float = cfg.Tc10
    Tc20: float = cfg.Tc20


# Sliding T_avg program ----------------
P_TURB_INIT = 1.0  # normalized turbine power at t=0

# Programmed T_avg reference vs turbine load (linear "sliding" schedule)
# Modest slope (order 10–20 K per p.u.) so Tref changes a few kelvin for a few-% load step.
S_TAVG_K_PER_PU = 16.7  # specific slope


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
    Kp: float = cfg.Kp_rc  # proportional gain
    Ki: float = cfg.Ki_rc  # integral gain

    # Power Tracking Controller
    Kpow_i: float = cfg.Kpow_i_rc  # power gain

    # Feedforward Compensation
    Kff: float = cfg.Kff_rc  # feedforward gain

    # Stability Features
    K_AW: float = cfg.K_AW_rc  # Anti-windup back-calculation gain
    LEAK: float = cfg.LEAK_rc  # Inner loop integral leak [1/s] - high for stability
    LEAK_OUTER: float = cfg.LEAK_OUTER_rc  # Outer loop integral leak [1/s] - high for stability
    DB_K: float = cfg.DB_rc_K  # Temperature deadband [K] - large to prevent hunting
    P_DEADBAND: float = cfg.P_DEADBAND_rc  # Power deadband [pu] (4%)

    # Rate and Position Limits
    U_MAX: float = cfg.U_MAX_rc  # Maximum controller output [1/s] based on physical limits
    x_rate_max: float = (cfg.vmax_rod_in_per_min / 60.0) / cfg.stroke_in  # max rod speed [1/s]
    x_min: float = 0.0  # fully withdrawn
    x_max: float = 1.0  # fully inserted
    Z_BIAS_MAX: float = cfg.Z_BIAS_MAX_rc  # Maximum power bias [K]


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
            Tc_init = 545.89  # K

        self.Tc_external = Tc_init  # Store for reference

        # Initialize state variables
        # There are 5 temperature states instead of 3
        # Tf, Tc1, Tc2 (core thermal model) + T_core_inlet, T_hot_leg (transport delays)

        # Start power consistent with initial turbine demand (pu)
        self.P_pu = float(P_turb_init)

        # Steady-state core temps for current inlet Tin and power
        Tf_ss, Tc1_ss, Tc2_ss = thermal_steady_state(Tc_init, self.P_pu)
        self.Tf = Tf_ss
        self.Tc1 = Tc1_ss
        self.Tc2 = Tc2_ss

        # --- make rho_fb start at ~0 pcm ---
        self.fb.T_f0 = float(self.Tf)
        self.fb.Tc10 = float(self.Tc1)
        self.fb.Tc20 = float(self.Tc2)

        # keep the programmed Tref schedule consistent too
        self.fb.T_m0 = float(self.Tc1)

        # Transport-delay states consistent with steady-state
        self.T_core_inlet = Tc_init
        f_bypass = 0.059
        self.T_hot_leg = (1 - f_bypass) * Tc2_ss + f_bypass * Tc_init

        # Precursor concentrations at equilibrium for current power
        self.C = (self.pk.beta_i / (self.pk.Lambda * self.pk.lambda_i)) * self.P_pu

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

        # Reactivity diagnostics (pcm and delta-k/k)
        self.rho_fb_pcm = 0.0
        self.rho_rod_pcm = 0.0
        self.rho_ext_pcm = 0.0
        self.rho_tot_pcm = 0.0
        self.rho_tot = 0.0

    def _compute_derivs(self, t, y, Tc_in_external, P_turb):
        """
        Compute derivatives for the full reactor system including transport delays.

        State vector y:
        [0:6]   : C_i (precursor concentrations)
        [6]     : P_pu (normalized power)
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
        P_pu = y[6]
        Tf = y[7]
        Tc1 = y[8]
        Tc2 = y[9]
        T_core_inlet = y[10]  # This is what enters the core
        T_hot_leg = y[11]  # This is what exits to hot leg
        x = y[12]
        z_integral = y[13]
        z_pow_bias = y[14]

        # Calculate reactivity feedbacks using core coolant temperatures
        dT_f = Tf - self.fb.T_f0
        dT_m = Tc1 - self.fb.Tc10  # First coolant lump feedback

        # Convert to Fahrenheit for feedback coefficients
        dT_f_F = dT_f * 9.0 / 5.0
        dT_m_F = dT_m * 9.0 / 5.0

        # Reactivity components
        rho_fb_pcm = (self.fb.alpha_D_input * dT_f_F +
                      self.fb.alpha_M_input * dT_m_F)

        rho_rod_pcm = self.rod_worth.rho_pcm_rel(x, X_OP)
        rho_ext_pcm = RHO_STEP_PCM
        rho_tot_pcm = rho_fb_pcm + rho_rod_pcm + rho_ext_pcm
        rho_tot = rho_tot_pcm * PCM_TO_DK

        # Store reactivity components for diagnostics
        self.rho_fb_pcm = rho_fb_pcm
        self.rho_rod_pcm = rho_rod_pcm
        self.rho_ext_pcm = rho_ext_pcm
        self.rho_tot_pcm = rho_tot_pcm
        self.rho_tot = rho_tot

        # Point kinetics
        rho_minus_beta = rho_tot - self.pk.beta
        dP_pu_dt = (rho_minus_beta / self.pk.Lambda) * P_pu + np.sum(self.pk.lambda_i * C)
        dC_dt = (self.pk.beta_i / self.pk.Lambda) * P_pu - self.pk.lambda_i * C

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
            # -----------------------------------------------------
            # HOLD RODS & INTEGRATORS BEFORE THE LOAD CUT
            # -----------------------------------------------------
            if t < STEP_T_S:
                dx_dt = 0.0
                dz_integral_dt = 0.0
                dz_pow_bias_dt = 0.0

            else:
                # -------------------------------------------------
                # NORMAL AUTO CONTROL AFTER t >= STEP_T_S
                # -------------------------------------------------
                # Use Mann/Holbert first coolant lump (Tc1) as core-average coolant temperature (θ1)
                # for control. Instrumentation-based Tavg is still available for diagnostics/plots.
                T_avg = Tc1
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
                u_ff = -self.ctrl.Kff * dP_turb

                # Outer Loop: Power Tracking Bias
                # Clamp power bias to prevent excessive temperature adjustments
                z_pow_clamped = np.clip(
                    z_pow_bias,
                    -self.ctrl.Z_BIAS_MAX,
                    self.ctrl.Z_BIAS_MAX
                )

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
                pushing_saturation = (
                        abs(u_total) >= self.ctrl.U_MAX and
                        np.sign(u_total) == np.sign(u_total_unsaturated)
                )

                if pushing_saturation:
                    # Anti-windup: unwind integral when saturated
                    dz_integral_dt = (
                            (u_total - u_total_unsaturated) / self.ctrl.K_AW -
                            self.ctrl.LEAK * z_integral
                    )
                else:
                    # Normal integration: error + back-calculation + leak
                    dz_integral_dt = (
                            e_temp_with_bias +
                            (u_total - u_total_unsaturated) / self.ctrl.K_AW -
                            self.ctrl.LEAK * z_integral
                    )

                # Outer Loop Power Bias Integral
                # Check if power bias is saturated
                if abs(z_pow_clamped) >= self.ctrl.Z_BIAS_MAX and np.sign(z_pow_bias) == np.sign(e_pow):
                    # Stop integrating if saturated and pushing further
                    dz_pow_bias_dt = 0.0
                else:
                    # Normal integration with leak: bias Tref based on power error
                    dz_pow_bias_dt = (
                            self.ctrl.Kpow_i * e_pow -
                            self.ctrl.LEAK_OUTER * z_pow_bias
                    )

        else:
            # Manual Rod Mode ignores controllers and uses externally commanded rod speed, but within physical rod limits
            # Guard against None so numpy.clip never sees a NoneType.
            u_raw = 0.0 if self.manual_u is None else self.manual_u
            u_manual = np.clip(u_raw,
                               -self.ctrl.x_rate_max,
                               self.ctrl.x_rate_max)
            dx_dt = u_manual

            # Let integrals wind down toward zero while in manual
            dz_integral_dt = -self.ctrl.LEAK * z_integral
            dz_pow_bias_dt = 0.0  # -self.ctrl.LEAK_OUTER * z_pow_bias

        # Assemble derivative vector
        dydt = np.zeros(15)
        dydt[0:6] = dC_dt
        dydt[6] = dP_pu_dt
        dydt[7] = dTf
        dydt[8] = dTc1
        dydt[9] = dTc2
        dydt[10] = dT_core_inlet_dt
        dydt[11] = dT_hot_leg_dt
        dydt[12] = dx_dt
        dydt[13] = dz_integral_dt
        dydt[14] = dz_pow_bias_dt

        return dydt

    def step(self, Tc_in: float, dt: float, P_turb: float = 1.0, control_mode: str = None, manual_u: float = 0.0):
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
        # Guard against None so numpy clip never sees a NoneType.
        if manual_u is None:
            manual_u = 0.0
        self.manual_u = manual_u

        # Store for diagnostics
        self.Tc_in = Tc_in

        # Store turbine power for feedforward calculation
        self.P_turb = P_turb

        # Assemble state vector
        y0 = np.zeros(15)
        y0[0:6] = self.C
        y0[6] = self.P_pu
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
        self.P_pu = y_final[6]
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
            'z_pow_bias': self.z_pow_bias,
            'rho_fb_pcm': self.rho_fb_pcm,
            'rho_rod_pcm': self.rho_rod_pcm,
            'rho_ext_pcm': self.rho_ext_pcm,
            'rho_tot_pcm': self.rho_tot_pcm,
            'rho_tot': self.rho_tot,
        }


