"""
AP1000 PWR Simulator – NiceGUI Dashboard
=========================================
A real-time, interactive GUI that mirrors the plant schematic,
with live readouts linked to the simulator's PlantState variables.

Run:
    pip install nicegui
    python gui_app.py

The browser will open at http://localhost:8080
"""

from nicegui import ui
from collections import deque
import copy
import time

# ── Import your simulator modules (they must be on PYTHONPATH) ──────────
from config import Config
from plant_state import PlantState
from reactor_core import ReactorSimulator
from steam_generator import SteamGenerator
from turbine_condenser import TurbineModel
from ic_system import ICSystem
from pathlib import Path

# NiceGUI compatibility helper
def html(content: str):
    return ui.html(content, sanitize=False)

PWR_SVG_PATH = Path('pwr_diagram.svg')  # <-- rename to your actual exported file
PWR_SVG_TEXT = PWR_SVG_PATH.read_text(encoding='utf-8')

# ══════════════════════════════════════════════════════════════════════════
# TRENDING / PLOT SYSTEM
# ══════════════════════════════════════════════════════════════════════════
TREND_MAX_PTS = 300  # ring buffer depth per parameter

# Map display-key → (human label, unit, color, value_getter_from_ps)
PLOTTABLE_PARAMS = {
    "core_mw":    ("Core Power",         "MW",     "#FF6D00",  lambda ps: ps.P_core_MW),
    "rod_pct":    ("Rod Insertion",       "%",      "#FFD600",  lambda ps: ps.rod_pos_pu * 100),
    "tavg_f":     ("Avg Coolant Temp",   "°F",     "#00E676",  lambda ps: (ps.Tavg_K - 273.15) * 9/5 + 32),
    "rho_dk":     ("Reactivity Δk/k",    "Δk/k",   "#E040FB",  lambda ps: ps.rho_reactivity_dk),
    "pzr_level":  ("PZR Level",          "%",      "#00B0FF",  lambda ps: ps.pzr_level_m / 16.27 * 100),
    "pzr_psig":   ("PZR Pressure",       "psig",   "#FFD600",  lambda ps: ps.P_pzr_Pa / 6894.76 - 14.696),
    "thot_f":     ("Hot Leg Temp",       "°F",     "#FF1744",  lambda ps: (ps.T_hot_K - 273.15) * 9/5 + 32),
    "tcold_f":    ("Cold Leg Temp",      "°F",     "#2196F3",  lambda ps: (ps.T_cold_K - 273.15) * 9/5 + 32),
    "steam_flow": ("Steam Flow",         "Mlb/hr", "#00E676",  lambda ps: ps.m_dot_steam_kg_s * 7936.64 / 1e6),
    "psec_psia":  ("Secondary Press",    "psig",   "#FF9100",  lambda ps: ps.P_secondary_Pa / 6894.76 - 14.696),
    "tsteam_f":   ("Steam Temp",         "°F",     "#CE93D8",  lambda ps: (ps.T_sec_K - 273.15) * 9/5 + 32),
    "turb_mw":    ("Turbine Power",      "MW",     "#76FF03",  lambda ps: ps.P_turbine_MW),
    "pzr_htr_kw": ("Heater Power",       "kW",     "#FF6D00",  lambda ps: ps.pzr_heater_kW),
}

# Ring buffers: key → deque of (t_s, value)
_trend_data: dict[str, deque] = {k: deque(maxlen=TREND_MAX_PTS) for k in PLOTTABLE_PARAMS}

# Active plot slots: list of 4 entries, each None or a display-key string
_plot_slots: list[str | None] = [None, None, None, None]
# EChart widget references (set during page build)
_plot_charts: list = [None, None, None, None]
# Label refs for slot headers
_plot_headers: list = [None, None, None, None]


# ── Simulator bootstrap ─────────────────────────────────────────────────
cfg = Config()
dt = 0.1  # integration time-step [s]

ps = PlantState()
ps.load_demand_pu = 1.0
ps.P_turbine_MW = ps.load_demand_pu * cfg.P_e_nom_MWe
ps.m_dot_steam_cmd_kg_s = ps.load_demand_pu * cfg.m_dot_steam_nom_kg_s
ps.rod_mode = "auto"
ps.rod_cmd_manual_pu = 0.0

reactor = ReactorSimulator(
    Tc_init=ps.T_cold_K,
    P_turb_init=ps.load_demand_pu,
    control_mode=ps.rod_mode,
)

steamgen = SteamGenerator(cfg)
turbine = TurbineModel(cfg)
ics = ICSystem(cfg=cfg, reactor=reactor, steamgen=steamgen, turbine=turbine)

# Sync initial state
ps.T_hot_K = reactor.T_hot_leg
ps.P_core_MW = reactor.P_pu * cfg.P_core_nom_MWt
ps.rod_pos_pu = reactor.x
Tcold0, m0, Psec0, Tsec0, lim0, h0 = steamgen.step(ps.T_hot_K, ps.m_dot_steam_cmd_kg_s, dt=0.0)
ps.T_cold_K = Tcold0
ps.P_secondary_Pa = Psec0
ps.T_sec_K = Tsec0
ps.m_dot_steam_kg_s = m0
ps.steam_h_J_kg = h0
ps.T_metal_K = steamgen.T_metal_K

# ── Simulation state ────────────────────────────────────────────────────

sim_running = False
sim_speed = 1  # how many dt steps per GUI tick (1 => time advances by dt = 0.1s per tick)

# --- UI refresh throttling (helps on low-RAM / slower machines) ---
# We can advance the simulation quickly, but only redraw the GUI at a steady rate
# so the browser/event-loop doesn't get overwhelmed.
_UI_REFRESH_HZ = 15.0                 # target GUI redraw rate
_UI_REFRESH_DT = 1.0 / _UI_REFRESH_HZ
_last_ui_update_wall_s = 0.0          # last redraw time (wall clock)


# --- Simulator hard reset function ---
def _reset_simulator_state():
    """Hard reset the simulator to initial conditions."""
    global ps, reactor, steamgen, turbine, ics, _plot_slots

    # Fresh state
    ps = PlantState()
    ps.load_demand_pu = 1.0
    ps.P_turbine_MW = ps.load_demand_pu * cfg.P_e_nom_MWe
    ps.m_dot_steam_cmd_kg_s = ps.load_demand_pu * cfg.m_dot_steam_nom_kg_s
    ps.rod_mode = "auto"
    ps.rod_cmd_manual_pu = 0.0

    reactor = ReactorSimulator(
        Tc_init=ps.T_cold_K,
        P_turb_init=ps.load_demand_pu,
        control_mode=ps.rod_mode,
    )

    steamgen = SteamGenerator(cfg)
    turbine = TurbineModel(cfg)
    ics = ICSystem(cfg=cfg, reactor=reactor, steamgen=steamgen, turbine=turbine)

    # Sync initial state
    ps.T_hot_K = reactor.T_hot_leg
    ps.P_core_MW = reactor.P_pu * cfg.P_core_nom_MWt
    ps.rod_pos_pu = reactor.x
    Tcold0, m0, Psec0, Tsec0, lim0, h0 = steamgen.step(ps.T_hot_K, ps.m_dot_steam_cmd_kg_s, dt=0.0)
    ps.T_cold_K = Tcold0
    ps.P_secondary_Pa = Psec0
    ps.T_sec_K = Tsec0
    ps.m_dot_steam_kg_s = m0
    ps.steam_h_J_kg = h0
    ps.T_metal_K = steamgen.T_metal_K

    # Clear trend data and plots
    for k in _trend_data:
        _trend_data[k].clear()
    for i in range(4):
        old_key = _plot_slots[i]
        if old_key:
            _update_plot_highlight(old_key, False)
        _plot_slots[i] = None
        _clear_plot(i)


# ── Conversions ─────────────────────────────────────────────────────────────
def K_to_F(K: float) -> float:
    return (K - 273.15) * 9.0 / 5.0 + 32.0


def Pa_to_psig(Pa: float) -> float:
    return Pa / 6894.76 - 14.696


def kg_s_to_lbm_hr(kg_s: float) -> float:
    return kg_s * 7936.64


# ── Display widget factory ──────────────────────────────────────────────
DISPLAY_UNITS = {}

def _toggle_plot(key: str):
    """Toggle a parameter into/out of the plot grid (up to 4 slots)."""
    global _plot_slots
    # If already plotted, remove it
    for i, slot_key in enumerate(_plot_slots):
        if slot_key == key:
            _plot_slots[i] = None
            _clear_plot(i)
            _update_plot_highlight(key, False)
            return
    # Find an empty slot
    for i, slot_key in enumerate(_plot_slots):
        if slot_key is None:
            _plot_slots[i] = key
            _update_plot_highlight(key, True)
            return
    # All 4 slots full — replace the oldest (slot 0), shift others down
    old_key = _plot_slots[0]
    if old_key:
        _update_plot_highlight(old_key, False)
    _plot_slots[0] = _plot_slots[1]
    _plot_slots[1] = _plot_slots[2]
    _plot_slots[2] = _plot_slots[3]
    _plot_slots[3] = key
    _update_plot_highlight(key, True)
    # Rebuild all charts on shift
    for i in range(4):
        if _plot_slots[i] is None:
            _clear_plot(i)

def _update_plot_highlight(key: str, active: bool):
    """Add/remove visual highlight on a display-box to show it's being plotted."""
    if key not in _all_display_labels:
        return
    lbl = _all_display_labels[key]
    if active:
        lbl.style("box-shadow: 0 0 8px 2px rgba(0,224,255,0.6); border-color: #00E0FF;")
    else:
        lbl.style("box-shadow: none; border-color: var(--accent-yellow);")

def _clear_plot(slot_idx: int):
    """Clear a plot slot to the empty state."""
    chart = _plot_charts[slot_idx]
    header = _plot_headers[slot_idx]
    if header:
        header.text = "Click a value to plot"
        header.style("color: #555;")
    if chart:
        chart.options["series"] = [{"type": "line", "data": [], "showSymbol": False}]
        chart.options["xAxis"]["data"] = []
        chart.update()

# Store all display-box label refs for highlight toggling
_all_display_labels: dict = {}

def _display(label_text: str, key: str, unit: str, labels: dict):
    """Create a labeled display readout. Clicking toggles trending plot."""
    DISPLAY_UNITS[key] = unit or ""
    with ui.column().classes("items-center gap-0"):
        ui.label(label_text).classes("display-label")
        lbl = ui.label("---").classes("display-box")
        # Make it clickable if it's a plottable parameter
        if key in PLOTTABLE_PARAMS:
            lbl.style("cursor: pointer;")
            lbl.on("click", lambda k=key: _toggle_plot(k))
        labels[key] = lbl
        _all_display_labels[key] = lbl


# ── Rod step movement system ────────────────────────────────────────────
_rod_step_size_pct = 5.0        # default step size in %
_rod_step_target = None         # target rod position (pu) or None if idle
_rod_step_direction = 0.0       # +1 insert, -1 withdraw
_ROD_SPEED_PU_PER_S = 0.02      # faster visible rod motion for manual steps
# ── NiceGUI slider event compatibility helper ───────────────────────────
def _event_value(e):
    """Compatibility: NiceGUI slider events can be raw numbers, dicts, or EventArguments with args/value."""
    # raw value
    if isinstance(e, (int, float)):
        return float(e)
    if isinstance(e, str):
        try:
            return float(e)
        except Exception:
            return None

    # dict payload (some versions)
    if isinstance(e, dict):
        if 'value' in e:
            try:
                return float(e['value'])
            except Exception:
                return None
        # sometimes nested
        if 'args' in e:
            a = e['args']
            if isinstance(a, (list, tuple)) and len(a) > 0:
                try:
                    return float(a[0])
                except Exception:
                    return None

    # EventArguments style
    if hasattr(e, 'value'):
        try:
            return float(getattr(e, 'value'))
        except Exception:
            pass

    if hasattr(e, 'args'):
        a = getattr(e, 'args')
        # common: args is a dict with 'value'
        if isinstance(a, dict) and 'value' in a:
            try:
                return float(a['value'])
            except Exception:
                return None
        # common: args is a list like [new_value]
        if isinstance(a, (list, tuple)) and len(a) > 0:
            try:
                return float(a[0])
            except Exception:
                return None
        # sometimes args is the raw value
        if isinstance(a, (int, float)):
            return float(a)

    return None

# Store labels ref for status updates from callbacks
_rod_labels_ref = None
_rod_mode_label = None  # set during page build


def _set_rod_step_size(val):
    global _rod_step_size_pct
    _rod_step_size_pct = float(val) if val is not None else 1.0
    print(f"[GUI] rod step size set to {_rod_step_size_pct:.2f}%")
    if _rod_labels_ref and "step_size_disp" in _rod_labels_ref:
        _rod_labels_ref["step_size_disp"].text = f"{_rod_step_size_pct:.1f}%"


def _set_rod_mode(mode: str):
    global ps, _rod_mode_label
    ps.rod_mode = mode
    if _rod_mode_label is not None:
        _rod_mode_label.text = f"Mode: {mode.upper()}"


def _rod_step_insert():
    global ps, _rod_step_target, _rod_step_direction
    if ps.rod_mode != "manual":
        _set_rod_mode("manual")
    delta = _rod_step_size_pct / 100.0
    _rod_step_target = min(1.0, ps.rod_pos_pu + delta)
    _rod_step_direction = 1.0
    ps.rod_cmd_manual_pu = _ROD_SPEED_PU_PER_S
    if _rod_labels_ref and "rod_move_status" in _rod_labels_ref:
        _rod_labels_ref["rod_move_status"].text = f"Inserting to {_rod_step_target*100:.1f}%"


def _rod_step_withdraw():
    global ps, _rod_step_target, _rod_step_direction
    if ps.rod_mode != "manual":
        _set_rod_mode("manual")
    delta = _rod_step_size_pct / 100.0
    _rod_step_target = max(0.0, ps.rod_pos_pu - delta)
    _rod_step_direction = -1.0
    ps.rod_cmd_manual_pu = -_ROD_SPEED_PU_PER_S
    if _rod_labels_ref and "rod_move_status" in _rod_labels_ref:
        _rod_labels_ref["rod_move_status"].text = f"Withdrawing to {_rod_step_target*100:.1f}%"


def _check_rod_step_complete():
    """Called each tick to stop rod motion when target is reached."""
    global ps, _rod_step_target, _rod_step_direction
    if _rod_step_target is None:
        return
    reached = (
        (_rod_step_direction > 0 and ps.rod_pos_pu >= _rod_step_target) or
        (_rod_step_direction < 0 and ps.rod_pos_pu <= _rod_step_target)
    )
    if reached:
        ps.rod_cmd_manual_pu = 0.0
        _rod_step_target = None
        _rod_step_direction = 0.0
        if _rod_labels_ref and "rod_move_status" in _rod_labels_ref:
            _rod_labels_ref["rod_move_status"].text = "Complete"


# ── Load demand step system ─────────────────────────────────────────────
_load_step_size_pct = 1.0
_load_labels_ref = None


def _set_load_step_size(val):
    global _load_step_size_pct
    _load_step_size_pct = float(val) if val is not None else 1.0
    if _load_labels_ref and "load_step_disp" in _load_labels_ref:
        _load_labels_ref["load_step_disp"].text = f"{_load_step_size_pct:.1f}%"


def _load_raise():
    global ps
    new_load = min(1.20, ps.load_demand_pu + _load_step_size_pct / 100.0)
    ps.load_demand_pu = new_load
    if _load_labels_ref:
        _load_labels_ref["load_disp"].text = f"{new_load * 100:.1f}%"
        _load_labels_ref["load_status"].text = f"Raised to {new_load * 100:.1f}%"


def _load_lower():
    global ps
    new_load = max(0.20, ps.load_demand_pu - _load_step_size_pct / 100.0)
    ps.load_demand_pu = new_load
    if _load_labels_ref:
        _load_labels_ref["load_disp"].text = f"{new_load * 100:.1f}%"
        _load_labels_ref["load_status"].text = f"Lowered to {new_load * 100:.1f}%"


def _set_speed(val):
    global sim_speed
    try:
        sim_speed = int(val)
    except Exception:
        sim_speed = 10


# ── Simulation tick ─────────────────────────────────────────────────────
def _tick(labels: dict):
    global ps, _last_ui_update_wall_s

    if not sim_running:
        return

    # 1) Advance physics as fast as requested (sim_speed steps per GUI tick)
    for _ in range(sim_speed):
        _check_rod_step_complete()
        ps = ics.step(ps, dt)
        ps = ps.copy_advance_time(dt)

    # 2) Record trend data for actively plotted parameters (cheap)
    t_now = ps.t_s
    for key in _plot_slots:
        if key is not None and key in PLOTTABLE_PARAMS:
            try:
                _trend_data[key].append((t_now, PLOTTABLE_PARAMS[key][3](ps)))
            except Exception:
                pass

    # 3) Throttle GUI redraw to a steady rate so the UI doesn't stutter/skips
    now_wall = time.monotonic()
    if (now_wall - _last_ui_update_wall_s) < _UI_REFRESH_DT:
        return
    _last_ui_update_wall_s = now_wall

    _update_displays(labels)


def _update_displays(labels: dict):
    labels["sim_time"].text = f"T = {ps.t_s:.1f} s"

    def set_box(key: str, value_text: str):
        unit = DISPLAY_UNITS.get(key, "")
        labels[key].text = f"{value_text} {unit}".strip()

    # Reactor core
    set_box("core_mw", f"{ps.P_core_MW:.0f}")
    # Rod insertion percentage (original convention)
    set_box("rod_pct", f"{ps.rod_pos_pu * 100:.0f}")
    set_box("tavg_f", f"{K_to_F(ps.Tavg_K):.0f}")

    # Pressurizer
    pzr_level_pct = getattr(ps, "pzr_level_m", 0.0) / 16.27 * 100
    set_box("pzr_level", f"{pzr_level_pct:.0f}")
    set_box("pzr_psig", f"{Pa_to_psig(ps.P_pzr_Pa):.0f}")

    # Hot / Cold legs
    set_box("thot_f", f"{K_to_F(ps.T_hot_K):.0f}")
    set_box("tcold_f", f"{K_to_F(ps.T_cold_K):.0f}")

    # Steam generator
    steam_flow_mlb_hr = kg_s_to_lbm_hr(ps.m_dot_steam_kg_s) / 1_000_000.0
    set_box("steam_flow", f"{steam_flow_mlb_hr:.2f}")
    set_box("psec_psia", f"{Pa_to_psig(ps.P_secondary_Pa):.0f}")
    set_box("tsteam_f", f"{K_to_F(ps.T_sec_K):.0f}")

    # Turbine
    set_box("turb_mw", f"{ps.P_turbine_MW:.0f}")

    # Load
    labels["load_disp"].text = f"{ps.load_demand_pu * 100:.1f}%"

    # Diagnostics
    set_box("rho_dk", f"{ps.rho_reactivity_dk:.5f}")
    set_box("sg_lim", "YES" if ps.sg_power_limited else "NO")

    # --- Pressurizer enhanced displays ---

    # Color-coded pressure: green=deadband, yellow=heaters, cyan=spray, red=near PORV
    pzr_p_mpa = ps.P_pzr_Pa / 1.0e6
    if pzr_p_mpa > 17.0:       # near PORV
        p_color = "#FF1744"
    elif pzr_p_mpa > 15.58:    # spray band
        p_color = "#00E5FF"
    elif pzr_p_mpa < 15.30:    # below heater band
        p_color = "#FF6D00"
    elif pzr_p_mpa < 15.51:    # heater band
        p_color = "#FFD600"
    else:                       # deadband (15.51–15.58)
        p_color = "#00E676"
    labels["pzr_psig"].style(f"color: {p_color}; border-color: {p_color};")

    # Heater bar gauge + kW readout
    heater_pct = ps.pzr_heater_frac * 100.0
    heater_kw = ps.pzr_heater_kW
    ui.run_javascript(
        f'document.getElementById("pzr_heater_bar") && '
        f'(document.getElementById("pzr_heater_bar").style.width = "{heater_pct:.1f}%");'
    )
    labels["pzr_htr_kw"].text = f"{heater_kw:.0f} kW"
    # Dim the kW readout when heater is off
    if heater_kw < 1.0:
        labels["pzr_htr_kw"].style("color: #555;")
    else:
        labels["pzr_htr_kw"].style("color: #FF6D00;")

    # Spray active indicator (icon color + label)
    spray_active = ps.pzr_spray_pu > 0.5
    if spray_active:
        ui.run_javascript(
            'var el = document.querySelector("#pzr_spray_icon svg"); if(el){el.setAttribute("stroke","#00E5FF");}'
            'var lb = document.getElementById("pzr_spray_label"); if(lb){lb.style.color="#00E5FF"; lb.textContent="ACTIVE";}'
        )
    else:
        ui.run_javascript(
            'var el = document.querySelector("#pzr_spray_icon svg"); if(el){el.setAttribute("stroke","#555");}'
            'var lb = document.getElementById("pzr_spray_label"); if(lb){lb.style.color="#555"; lb.textContent="OFF";}'
        )

    # Surge direction indicator
    surge_dir = ps.pzr_surge_direction
    if surge_dir == "IN-SURGE":
        labels["pzr_surge"].text = "▲ IN-SURGE"
        labels["pzr_surge"].style("color: #FF6D00; font-size: 0.7rem; font-family: Orbitron; font-weight: 700;")
    elif surge_dir == "OUT-SURGE":
        labels["pzr_surge"].text = "▼ OUT-SURGE"
        labels["pzr_surge"].style("color: #00B0FF; font-size: 0.7rem; font-family: Orbitron; font-weight: 700;")
    else:
        labels["pzr_surge"].text = "— NEUTRAL —"
        labels["pzr_surge"].style("color: #90A4AE; font-size: 0.7rem; font-family: Orbitron; font-weight: 700;")

    # Rod mode
    labels["rod_mode_disp"].text = f"Mode: {ps.rod_mode.upper()}"

    # --- SVG animation updates ---
    # These JS calls are relatively expensive; `_tick` already throttles redraws,
    # so we keep them here (only called on redraw).
    ui.run_javascript(f"window.pwrSetRod && window.pwrSetRod({ps.rod_pos_pu});")
    ui.run_javascript(f"window.pwrSetPumpsRunning && window.pwrSetPumpsRunning({str(sim_running).lower()});")

    # --- Update trend plots ---
    _update_trend_plots()

_trend_update_counter = 0
_TREND_UPDATE_INTERVAL = 6  # only update charts every Nth display tick (lighter for slower machines)

def _update_trend_plots():
    """Push latest ring-buffer data into the active EChart slots (throttled)."""
    global _trend_update_counter
    _trend_update_counter += 1
    if _trend_update_counter % _TREND_UPDATE_INTERVAL != 0:
        # Still update the current-value overlay every tick (cheap)
        for i in range(4):
            key = _plot_slots[i]
            chart = _plot_charts[i]
            if chart is None or key is None or key not in PLOTTABLE_PARAMS:
                continue
            buf = _trend_data.get(key)
            if not buf:
                continue
            label, unit, color, _ = PLOTTABLE_PARAMS[key]
            current_val = buf[-1][1]
            if abs(current_val) >= 100:
                val_text = f"{current_val:.1f}"
            elif abs(current_val) >= 1:
                val_text = f"{current_val:.2f}"
            else:
                val_text = f"{current_val:.4f}"
            chart.options["graphic"] = [{
                "type": "text", "right": 14, "top": 6,
                "style": {
                    "text": f"{val_text} {unit}", "fill": color,
                    "fontSize": 16, "fontWeight": "bold",
                    "fontFamily": "Orbitron, monospace", "textAlign": "right",
                },
            }]
            chart.update()
        return

    # Full chart data update (every Nth tick)
    for i in range(4):
        key = _plot_slots[i]
        chart = _plot_charts[i]
        header = _plot_headers[i]
        if chart is None:
            continue
        if key is None or key not in PLOTTABLE_PARAMS:
            if header:
                header.text = "Click a value to plot"
                header.style("color: #555;")
            continue

        label, unit, color, _ = PLOTTABLE_PARAMS[key]
        buf = _trend_data.get(key)
        if not buf or len(buf) == 0:
            continue

        # Downsample in-place: step through the deque without copying
        n = len(buf)
        step = max(1, n // 150)
        times = []
        vals = []
        for j in range(0, n, step):
            t_val, y_val = buf[j]
            times.append(f"{t_val:.0f}")
            vals.append(round(y_val, 3))
        # Always include latest point
        last_t, last_v = buf[-1]
        if not times or times[-1] != f"{last_t:.0f}":
            times.append(f"{last_t:.0f}")
            vals.append(round(last_v, 3))

        if header:
            header.text = f"{label} ({unit})"
            header.style(f"color: {color};")

        current_val = last_v
        if abs(current_val) >= 100:
            val_text = f"{current_val:.1f}"
        elif abs(current_val) >= 1:
            val_text = f"{current_val:.2f}"
        else:
            val_text = f"{current_val:.4f}"

        chart.options["xAxis"]["data"] = times
        chart.options["series"] = [{
            "type": "line", "data": vals, "showSymbol": False,
            "lineStyle": {"color": color, "width": 2},
            "itemStyle": {"color": color},
            "areaStyle": {"color": f"{color}22"},
        }]
        chart.options["yAxis"]["name"] = unit
        chart.options["graphic"] = [{
            "type": "text", "right": 14, "top": 6,
            "style": {
                "text": f"{val_text} {unit}", "fill": color,
                "fontSize": 16, "fontWeight": "bold",
                "fontFamily": "Orbitron, monospace", "textAlign": "right",
            },
        }]
        chart.update()


# ── Build UI ────────────────────────────────────────────────────────────
@ui.page("/")
def main_page():
    # ── CSS ──────────────────────────────────────────────────────────
    ui.add_head_html("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&display=swap');

    /* Fit + center the SVG without clipping */
    #diagram_box #pwr_svg_wrap svg {
        width: 100% !important;
        height: 100% !important;
        max-width: 100% !important;
        max-height: 100% !important;
        display: block;
        margin: 0 auto;
        transform: none;
    }

    :root {
        --bg-dark: #1a1a2e;
        --bg-panel: #16213e;
        --bg-card: #0f3460;
        --accent-yellow: #FFD600;
        --accent-orange: #FF6D00;
        --accent-red: #FF1744;
        --accent-blue: #00B0FF;
        --accent-green: #00E676;
        --text-primary: #E0E0E0;
        --text-dim: #90A4AE;
    }

    body {
        background: var(--bg-dark) !important;
        font-family: 'Share Tech Mono', monospace !important;
        color: var(--text-primary);
    }

    .display-box {
        background: #111;
        border: 2px solid var(--accent-yellow);
        border-radius: 6px;
        padding: 3px 10px; /* smaller */
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        font-size: 1.15rem; /* smaller */
        color: var(--accent-yellow);
        text-align: center;
        min-width: 92px; /* smaller */
        text-shadow: 0 0 8px rgba(255,214,0,0.5);
        line-height: 1.35;
        transition: box-shadow 0.15s, border-color 0.15s;
    }

    .display-box[style*="cursor: pointer"]:hover {
        box-shadow: 0 0 12px 3px rgba(0,224,255,0.35);
        border-color: #00E0FF;
    }

    .mini-box {
        background: #111;
        border: 2px solid var(--accent-yellow);
        border-radius: 6px;
        padding: 3px 10px;
        font-family: 'Orbitron', monospace;
        font-weight: 700;
        font-size: 0.85rem;
        color: var(--accent-yellow);
        text-align: center;
        text-shadow: 0 0 8px rgba(255,214,0,0.35);
        line-height: 1.3;
        min-width: 130px;
    }

    .status-box {
        background: #111;
        border: 2px solid var(--accent-red);
        border-radius: 999px;
        padding: 3px 12px;
        font-family: 'Orbitron', monospace;
        font-weight: 900;
        font-size: 0.75rem;
        color: var(--accent-red);
        text-align: center;
        letter-spacing: 1px;
        line-height: 1.3;
        min-width: 120px;
    }

    .display-label {
        font-family: 'Orbitron', monospace;
        font-size: 0.60rem; /* smaller */
        font-weight: 700;
        color: var(--accent-yellow);
        text-transform: uppercase;
        letter-spacing: 1.4px;
        text-align: center;
        margin-bottom: 2px;
    }

    .component-card {
        background: linear-gradient(145deg, #0f3460, #0a2a50);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 12px; /* smaller cards */
        box-shadow: 0 4px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
    }

    .component-title {
        font-family: 'Orbitron', monospace;
        font-weight: 900;
        font-size: 0.78rem; /* smaller */
        color: white;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-align: center;
        margin-bottom: 8px;
    }

    .pipe-hot {
        background: linear-gradient(90deg, #B71C1C, #F44336, #B71C1C);
        height: 12px;
        border-radius: 6px;
        box-shadow: 0 0 10px rgba(244,67,54,0.4);
    }

    .pipe-cold {
        background: linear-gradient(90deg, #0D47A1, #2196F3, #0D47A1);
        height: 12px;
        border-radius: 6px;
        box-shadow: 0 0 10px rgba(33,150,243,0.4);
    }

    .header-bar {
        background: linear-gradient(90deg, #0a1628, #16213e, #0a1628);
        border-bottom: 2px solid var(--accent-yellow);
        padding: 8px 24px;
    }

    .control-btn {
        font-family: 'Orbitron', monospace !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
    }
    
    /* ---- SVG animation helpers ---- */
    #pwr_svg_wrap svg * {
        transform-box: fill-box;        /* makes CSS transforms work in inline SVG */
        transform-origin: center;       /* rotate around the element's center */
    }

    @keyframes pwr_spin {
        from { transform: rotate(0deg); }
        to   { transform: rotate(360deg); }
    }

    .pwr-pump-spinning {
        animation: pwr_spin 0.9s linear infinite;
    }

    /* smooth rod motion */
    #pwr_svg_wrap #rod_group {
        transition: transform 0.12s linear;
    }

    </style>
    """)

    labels = {}

    # ── HEADER BAR ───────────────────────────────────────────────────
    with ui.row().classes("header-bar w-full items-center justify-between"):
        ui.label("AP1000 PWR SIMULATOR").style(
            "font-family: Orbitron; font-weight: 900; font-size: 1.3rem; color: white; letter-spacing: 3px;"
        )
        with ui.row().classes("items-center gap-4"):
            ui.label("")

    # ── MAIN LAYOUT ───────────────────────────────────────────────────
    from pathlib import Path

    # ── MAIN LAYOUT ───────────────────────────────────────────────────
    with ui.row().classes("w-full p-4 gap-4 items-start flex-wrap"):

        # LEFT SIDE: 4 unit cards + diagnostics stacked under them
        with ui.column().classes("gap-4").style("flex: 0 0 auto; width: 820px;"):

            UNIT_COL_H = "440px"  # keep the 4 left columns the same height (tall enough to contain all readouts)
            UNIT_MIN_W = "180px"  # allow 4 columns to fit comfortably within 820px

            # Top row: the 4 unit cards (columns 1–4)
            with ui.row().classes("gap-4 flex-wrap").style("width: 820px;"):

                # COLUMN 1: REACTOR CORE
                with ui.column().classes("component-card items-center gap-2").style(f"height: {UNIT_COL_H}; flex: 1 1 {UNIT_MIN_W}; min-width: {UNIT_MIN_W};"):
                    ui.label("REACTOR CORE").classes("component-title")
                    _display("CORE POWER", "core_mw", "MW", labels)
                    #html(""" ... your reactor svg ... """)
                    _display("ROD INSERTION %", "rod_pct", "%", labels)
                    _display("AVG COOLANT TEMP", "tavg_f", "°F", labels)
                    ui.separator().style("background: rgba(255,255,255,0.1); margin: 6px 0 4px 0;")
                    ui.label("REACTIVITY").classes("component-title")
                    _display("Δk/k", "rho_dk", "", labels)

                # COLUMN 2: PRESSURIZER (single card, fixed height)
                with ui.column().classes("component-card items-center gap-2").style(f"height: {UNIT_COL_H}; flex: 1 1 {UNIT_MIN_W}; min-width: {UNIT_MIN_W};"):
                    ui.label("PRESSURIZER").classes("component-title")
                    _display("LEVEL", "pzr_level", "%", labels)
                    _display("PRESSURE", "pzr_psig", "psig", labels)

                    ui.separator().style("background: rgba(255,255,255,0.1); margin: 6px 0 4px 0;")

                    # --- Heater bar gauge + kW readout ---
                    ui.label("HEATER OUTPUT").style(
                        "font-family: Orbitron; font-size: 0.55rem; font-weight: 700; "
                        "color: var(--accent-yellow); letter-spacing: 1.4px; text-align: center;"
                    )
                    # Bar gauge container
                    html("""
                    <div style="width: 130px; height: 16px; background: #222; border: 1px solid #555;
                                border-radius: 3px; overflow: hidden; position: relative;">
                        <div id="pzr_heater_bar" style="height: 100%; width: 0%; transition: width 0.15s linear;
                             background: linear-gradient(90deg, #FF6D00, #FFD600); border-radius: 2px;"></div>
                    </div>
                    """)
                    labels["pzr_htr_kw"] = ui.label("0 kW").style(
                        "font-family: Orbitron; font-weight: 700; font-size: 0.85rem; "
                        "color: #FF6D00; text-align: center; min-width: 80px;"
                    )

                    ui.separator().style("background: rgba(255,255,255,0.1); margin: 4px 0;")

                    # --- Spray active indicator with shower icon ---
                    with ui.row().classes("items-center justify-center gap-2"):
                        ui.label("SPRAY").style(
                            "font-family: Orbitron; font-size: 0.55rem; font-weight: 700; "
                            "color: var(--accent-yellow); letter-spacing: 1.4px;"
                        )
                        # Shower spray SVG icon + status label
                        html("""
                        <div id="pzr_spray_icon" style="display: flex; align-items: center; gap: 6px;">
                            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#555"
                                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M4 4l2.5 2.5"/>
                                <path d="M13.5 6.5a4 4 0 0 0-5.5 0"/>
                                <path d="M15 5a7 7 0 0 0-9.5 0"/>
                                <path d="M8 10v.01"/><path d="M10 12v.01"/><path d="M6 12v.01"/>
                                <path d="M12 14v.01"/><path d="M8 14v.01"/><path d="M4 14v.01"/>
                                <path d="M10 16v.01"/><path d="M6 16v.01"/>
                                <path d="M8 18v.01"/>
                            </svg>
                            <span id="pzr_spray_label" style="font-family: Orbitron; font-size: 0.7rem;
                                  font-weight: 700; color: #555;">OFF</span>
                        </div>
                        """)

                    ui.separator().style("background: rgba(255,255,255,0.1); margin: 4px 0;")

                    # --- Surge direction indicator ---
                    ui.label("SURGE").style(
                        "font-family: Orbitron; font-size: 0.55rem; font-weight: 700; "
                        "color: var(--accent-yellow); letter-spacing: 1.4px; text-align: center;"
                    )
                    labels["pzr_surge"] = ui.label("— NEUTRAL —").style(
                        "font-family: Orbitron; font-weight: 700; font-size: 0.7rem; "
                        "color: #90A4AE; text-align: center; min-width: 110px;"
                    )

                # COLUMN 3: STEAM GENERATOR
                with ui.column().classes("component-card items-center gap-2").style(f"height: {UNIT_COL_H}; flex: 1 1 {UNIT_MIN_W}; min-width: {UNIT_MIN_W};"):
                    ui.label("STEAM GENERATOR").classes("component-title")
                    _display("STEAM FLOW", "steam_flow", "Mlb/hr", labels)
                    #html(""" ... your SG svg ... """)
                    with ui.row().classes("gap-6 justify-center w-full"):
                        _display("SECONDARY PRESSURE", "psec_psia", "psig", labels)
                        _display("STEAM TEMPERATURE", "tsteam_f", "°F", labels)
                    ui.separator().style("background: rgba(255,255,255,0.1); margin: 6px 0 4px 0;")
                    ui.label("SG LIMITED").classes("component-title")
                    _display("STATUS", "sg_lim", "", labels)

                # COLUMN 4: TURBINE + TEMPERATURES (fixed total height; equal card widths)
                with ui.column().classes("gap-3").style(f"height: {UNIT_COL_H}; flex: 1 1 {UNIT_MIN_W}; min-width: {UNIT_MIN_W};"):

                    # Turbine card (fills upper half)
                    with ui.column().classes("component-card items-center w-full").style("flex: 1;"):
                        ui.label("TURBINE").classes("component-title")
                        #html(""" ... your turbine svg ... """)
                        _display("TURBINE POWER", "turb_mw", "MW", labels)

                    # Temperatures card (fills lower half)
                    with ui.column().classes("component-card items-center justify-between w-full").style("flex: 1;"):
                        ui.label("TEMPERATURES").classes("component-title")
                        _display("HOT LEG", "thot_f", "°F", labels)
                        html('<div class="pipe-hot" style="width: 120px;"></div>')
                        html('<div class="pipe-cold" style="width: 120px;"></div>')
                        _display("COLD LEG", "tcold_f", "°F", labels)

            # ── CONTROL PANEL (moved under the 4 unit columns) ─────────────
            with ui.row().classes("gap-4 flex-wrap items-start").style("width: 820px;"):

                CTRL_MIN_W = "215px"  # allow 3 cards to fit comfortably within 820px

                # Simulation controls
                with ui.column().classes("component-card").style(f"flex: 1 1 {CTRL_MIN_W}; min-width: {CTRL_MIN_W};"):
                    ui.label("SIMULATION CONTROL").classes("component-title")
                    with ui.row().classes("justify-center items-center gap-3").style("margin-bottom: 8px;"):
                        labels["sim_time"] = ui.label("T = 0.0 s").classes("mini-box")
                        labels["sim_status"] = ui.label("STOPPED").classes("status-box")

                    def _start_sim():
                        global sim_running, _last_ui_update_wall_s
                        sim_running = True
                        _last_ui_update_wall_s = 0.0  # force immediate redraw
                        timer.active = True
                        labels["sim_status"].text = "RUNNING"
                        labels["sim_status"].style("color: #00E676; border-color: #00E676;")
                        ui.run_javascript("window.pwrSetPumpsRunning && window.pwrSetPumpsRunning(true);")
                    def _stop_sim():
                        global sim_running, _last_ui_update_wall_s
                        sim_running = False
                        timer.active = False
                        labels["sim_status"].text = "STOPPED"
                        labels["sim_status"].style("color: #FF1744; border-color: #FF1744;")
                        ui.run_javascript("window.pwrSetPumpsRunning && window.pwrSetPumpsRunning(false);")

                    with ui.row().classes("gap-2"):
                        ui.button("▶ START", on_click=_start_sim).classes("control-btn").props("color=green-9 dense")
                        ui.button("■ STOP", on_click=_stop_sim).classes("control-btn").props("color=red-9 dense")
                        ui.button("↺ RESET", on_click=lambda: _reset_sim()).classes("control-btn").props("color=grey-8 dense")

                    def _reset_sim():
                        global sim_running, _last_ui_update_wall_s
                        # stop
                        sim_running = False
                        _last_ui_update_wall_s = 0.0
                        timer.active = False
                        # reset the backend state
                        _reset_simulator_state()
                        # update status + displays
                        labels["sim_status"].text = "STOPPED"
                        labels["sim_status"].style("color: #FF1744; border-color: #FF1744;")
                        labels["sim_time"].text = "T = 0.0 s"
                        _update_displays(labels)
                        ui.run_javascript("window.pwrSetPumpsRunning && window.pwrSetPumpsRunning(false);")

                    ui.label("Speed (steps/tick):").style("color: #90A4AE; font-size: 0.75rem; margin-top: 8px;")
                    speed_slider = ui.slider(min=1, max=50, value=sim_speed, step=1).style("width: 100%;")
                    speed_slider.on("update:model-value", lambda e: _set_speed(_event_value(e)))

                # Rod control
                with ui.column().classes("component-card").style(f"flex: 1 1 {CTRL_MIN_W}; min-width: {CTRL_MIN_W};"):
                    ui.label("ROD CONTROL").classes("component-title")

                    with ui.row().classes("gap-2"):
                        ui.button("AUTO", on_click=lambda: _set_rod_mode("auto")).classes("control-btn").props("color=blue-9 dense")
                        ui.button("MANUAL", on_click=lambda: _set_rod_mode("manual")).classes("control-btn").props("color=orange-9 dense")

                    labels["rod_mode_disp"] = ui.label("Mode: AUTO").style(
                        "color: #4FC3F7; font-family: Orbitron; font-size: 0.7rem; margin-top: 4px;"
                    )
                    global _rod_mode_label
                    _rod_mode_label = labels["rod_mode_disp"]

                    ui.separator().style("background: rgba(255,255,255,0.1); margin: 6px 0;")

                    ui.label("Step Size (%):").style("color: #90A4AE; font-size: 0.75rem;")
                    step_size_slider = ui.slider(min=0.5, max=10.0, value=_rod_step_size_pct, step=0.5).style("width: 100%;")
                    step_size_slider.on("update:model-value", lambda e: _set_rod_step_size(_event_value(e)))

                    labels["step_size_disp"] = ui.label(f"{_rod_step_size_pct:.1f}%").style(
                        "color: #FFD600; font-family: Orbitron; font-size: 0.7rem; text-align: center;"
                    )
                    # force initial sync (and ensures label updates even if events behave differently)
                    _set_rod_step_size(_rod_step_size_pct)

                    with ui.row().classes("gap-3 mt-1"):
                        ui.button("▲ WITHDRAW", on_click=_rod_step_withdraw).classes("control-btn").props("color=green-9 dense").style("font-size: 0.75rem;")
                        ui.button("▼ INSERT", on_click=_rod_step_insert).classes("control-btn").props("color=red-9 dense").style("font-size: 0.75rem;")

                    labels["rod_move_status"] = ui.label("").style(
                        "color: #90A4AE; font-family: Share Tech Mono; font-size: 0.65rem; margin-top: 2px; min-height: 1.2em;"
                    )

                # Load demand
                with ui.column().classes("component-card").style(f"flex: 1 1 {CTRL_MIN_W}; min-width: {CTRL_MIN_W};"):
                    ui.label("LOAD DEMAND").classes("component-title")

                    labels["load_disp"] = ui.label("100.0%").classes("display-box")

                    ui.separator().style("background: rgba(255,255,255,0.1); margin: 6px 0;")

                    ui.label("Step Size (%):").style("color: #90A4AE; font-size: 0.75rem;")
                    load_step_slider = ui.slider(min=0.5, max=10.0, value=_load_step_size_pct, step=0.5).style("width: 100%;")
                    load_step_slider.on("update:model-value", lambda e: _set_load_step_size(_event_value(e)))

                    labels["load_step_disp"] = ui.label(f"{_load_step_size_pct:.1f}%").style(
                        "color: #FFD600; font-family: Orbitron; font-size: 0.7rem; text-align: center;"
                    )
                    _set_load_step_size(_load_step_size_pct)

                    with ui.row().classes("gap-3 mt-1"):
                        ui.button("▲ RAISE", on_click=_load_raise).classes("control-btn").props("color=green-9 dense").style("font-size: 0.75rem;")
                        ui.button("▼ LOWER", on_click=_load_lower).classes("control-btn").props("color=red-9 dense").style("font-size: 0.75rem;")

                    labels["load_status"] = ui.label("").style(
                        "color: #90A4AE; font-family: Share Tech Mono; font-size: 0.65rem; margin-top: 2px; min-height: 1.2em;"
                    )

        # RIGHT SIDE: Plant Diagram + Trend Plots
        with ui.column().classes("gap-4").style("flex: 1; min-width: 520px;"):

            # Plant Diagram card
            with ui.column().classes("component-card").style("width: 100%;"):
                ui.label("PLANT DIAGRAM").classes("component-title")

                svg_path = Path(__file__).with_name("pwr_diagram.svg")
                if svg_path.exists():
                    # Constrain the diagram area and scale the SVG to fit without requiring browser zoom-out
                    html(f"""
                    <div id=\"diagram_box\" style=\"
                        width: 100%;
                        height: 460px;
                        overflow: hidden;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    \">
                      <div id=\"pwr_svg_wrap\" style=\"width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;\">
                        {svg_path.read_text(encoding='utf-8')}
                      </div>
                    </div>
                    """)
                else:
                    ui.label(f"Missing {svg_path.name} (put it next to gui_app.py)").style("color: #FF1744;")

            # Ensure the embedded SVG uses a "contain"-style fit and crops internal padding
            ui.timer(0.1, lambda: ui.run_javascript("""
              const svg = document.querySelector('#pwr_svg_wrap svg');
              if (!svg) return;

              // Crop internal empty padding by tightening the viewBox to the actual drawn content.
              // This fixes the "extra space on the right" when the SVG viewBox is wider than the artwork.
              try {
                const vb = svg.getAttribute('viewBox');
                const box = svg.getBBox();

                // Add a tiny padding so strokes aren't clipped
                const pad = 8;
                const x = Math.max(box.x - pad, 0);
                const y = Math.max(box.y - pad, 0);
                const w = box.width + 2 * pad;
                const h = box.height + 2 * pad;

                // Only apply if bbox looks sane
                if (w > 0 && h > 0 && Number.isFinite(w) && Number.isFinite(h)) {
                  svg.setAttribute('viewBox', `${x} ${y} ${w} ${h}`);
                }
              } catch (e) {
                // If getBBox fails (rare), do nothing.
              }

              // Then fit + center
              svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
              svg.style.width = '100%';
              svg.style.height = '100%';
              svg.style.maxWidth = '100%';
              svg.style.maxHeight = '100%';
              svg.style.margin = '0 auto';
              svg.style.transform = 'none';
            """), once=True)

            # Define SVG control functions once
            ui.timer(0.2, lambda: ui.run_javascript(r"""
              // --- AP1000 SVG hooks (requires IDs in the SVG) ---
              // Expected IDs: rod_group, pump_cold_leg, pump_hot_leg

              window.pwrSetPumpsRunning = function(isRunning) {
                const p1 = document.querySelector('#pwr_svg_wrap #pump_cold_leg');
                const p2 = document.querySelector('#pwr_svg_wrap #pump_hot_leg');
                [p1, p2].forEach(p => {
                  if (!p) return;
                  if (isRunning) p.classList.add('pwr-pump-spinning');
                  else p.classList.remove('pwr-pump-spinning');
                });
              };

              // Rod movement: SVG baseline is FULLY INSERTED (100%). Withdrawing moves rods UP.
              // IMPORTANT: SVG `transform="translate(...)"` uses SVG user units (viewBox units), not CSS pixels.
              // So we scale travel based on the SVG viewBox height to make motion visible.
              const ROD_TRAVEL_FRAC = 0.25; // bigger travel so motion is obvious

              window.pwrSetRod = function(rod_pu) {
                const svg = document.querySelector('#pwr_svg_wrap svg');
                const rod = document.querySelector('#pwr_svg_wrap #rod_group');
                if (!svg || !rod) return;

                const pu = Math.max(0, Math.min(1, Number(rod_pu)));

                // Determine travel in SVG units
                let h = 0;
                try {
                  // Prefer viewBox height if present
                  if (svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.height) {
                    h = svg.viewBox.baseVal.height;
                  }
                } catch (e) {}

                // Fallback: use rod bbox height if viewBox is missing
                if (!h) {
                  try {
                    const b = rod.getBBox();
                    h = (b && b.height) ? (b.height * 6) : 800; // heuristic fallback
                  } catch (e) {
                    h = 800;
                  }
                }

                const travel = h * ROD_TRAVEL_FRAC;
                // SVG drawing baseline = 100% insertion (fully inserted). 
                // Therefore, decreasing insertion should move rods UP from the baseline.
                const dy = - (1 - pu) * travel; // 1.0 => 0, 0.0 => -travel (up)

                // Cache the original transform the first time we see the element.
                if (!rod.dataset.baseTransform) {
                  rod.dataset.baseTransform = rod.getAttribute('transform') || '';
                }

                const base = rod.dataset.baseTransform;
                const translate = `translate(0 ${dy})`;
                rod.setAttribute('transform', base ? `${base} ${translate}` : translate);

                // Also set CSS transform as a backup for browsers that don't animate attribute changes
                // (should not be relied on for movement, but can help in some cases)
                rod.style.transform = `translate(0px, ${dy}px)`;
              };
            """), once=True)

            # ── 2×2 TREND PLOT GRID ──────────────────────────────────────
            with ui.column().classes("component-card").style("width: 100%; margin-top: 8px;"):
                with ui.row().classes("items-center justify-between w-full"):
                    ui.label("TREND PLOTS").classes("component-title")
                    ui.label("Click any value readout to plot it here (up to 4)").style(
                        "color: #555; font-size: 0.65rem; font-family: Share Tech Mono;"
                    )

                ECHART_BASE = {
                    "animation": False,
                    "grid": {"top": 30, "right": 16, "bottom": 28, "left": 58, "containLabel": False},
                    "xAxis": {
                        "type": "category",
                        "data": [],
                        "axisLabel": {"color": "#666", "fontSize": 10},
                        "axisLine": {"lineStyle": {"color": "#333"}},
                        "splitLine": {"show": False},
                    },
                    "yAxis": {
                        "type": "value",
                        "name": "",
                        "nameTextStyle": {"color": "#888", "fontSize": 10},
                        "axisLabel": {"color": "#888", "fontSize": 10},
                        "axisLine": {"lineStyle": {"color": "#333"}},
                        "splitLine": {"lineStyle": {"color": "#222"}},
                    },
                    "series": [{"type": "line", "data": [], "showSymbol": False}],
                    "backgroundColor": "transparent",
                }

                # Two rows of two plots each
                for row_start in (0, 2):
                    with ui.row().classes("gap-3 w-full"):
                        for idx in (row_start, row_start + 1):
                            with ui.column().style(
                                "flex: 1; background: #0a0a14; border: 1px solid #222; border-radius: 6px; "
                                "padding: 4px 6px;"
                            ):
                                _plot_headers[idx] = ui.label("Click a value to plot").style(
                                    "color: #555; font-family: Orbitron; font-size: 0.6rem; "
                                    "font-weight: 700; letter-spacing: 1px; text-align: center; "
                                    "width: 100%;"
                                )
                                chart = ui.echart(copy.deepcopy(ECHART_BASE)).style(
                                    "width: 100%; height: 220px;"
                                )
                                _plot_charts[idx] = chart



    # ── Timer for periodic updates ───────────────────────────────────
    timer = ui.timer(0.08, lambda: _tick(labels), active=False)

    # Save label refs for callbacks
    global _rod_labels_ref, _load_labels_ref
    _rod_labels_ref = labels
    _load_labels_ref = labels

    # Initial display refresh
    _update_displays(labels)


# ── Launch ──────────────────────────────────────────────────────────────
ui.run(title="AP1000 Simulator", port=8080, reload=False)

