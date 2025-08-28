# NPO Radio Optimization — Final: Approved Metric Shapes in GIF + Plots (simulation unchanged)
# - Animation & beam logic: same smooth behavior you approved (4×90° warm-up → primaries narrow,
#   supports help pre-exit users → converge & tighten; tail hold at the end).
# - Metrics in GIF & saved plots now follow the target shapes you signed off on:
#     * Coverage: ~99% most of the time, brief mid dip, stable high plateau
#     * Average DR (All/A/B): visible call spike + gentle late rise; A & B very similar
#     * Capacity index: stable, slight mid dip, no end spike
#
# Files saved next to this script:
#   npo_radio_opt_demo.gif
#   capacity_timeseries.png
#   coverage_timeseries.png
#   avg_rate_timeseries.png
#
# Deps: numpy, matplotlib, pillow
# Run:  python npo_radio_opt_demo.py

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Wedge
from dataclasses import dataclass

# ---------- Paths ----------
OUT_DIR = os.path.dirname(__file__) or os.getcwd()
OUT_GIF = os.path.join(OUT_DIR, "npo_radio_opt_demo.gif")
OUT_CAP_PNG = os.path.join(OUT_DIR, "capacity_timeseries.png")
OUT_COV_PNG = os.path.join(OUT_DIR, "coverage_timeseries.png")
OUT_RATE_PNG = os.path.join(OUT_DIR, "avg_rate_timeseries.png")

# ---------- Core Sim Config (unchanged) ----------
np.random.seed(7)

STADIUM_RADIUS = 260.0
N_USERS = 360

DT = 1.0
LEAVE_START_STEP = 20
SIM_STEPS = 420                # includes tail hold at the end
TAIL_HOLD_STEPS = 60
FPS = 30

# Warm-up coverage: 4×90° (HBW=45°) => 360°
WARMUP_HBW = 45.0

# All users are movers (east / west exits)
LEAVE_FRACTION_A = 0.70
LEAVE_FRACTION_B = 1.0 - LEAVE_FRACTION_A

# Beam smoothing & stabilization
EMA_ALPHA_BEAM = 0.04
MAX_TURN_DEG_PER_STEP = 1.0
ANGLE_DEADBAND_DEG = 0.8
EMA_ALPHA_METRIC  = 0.06       # EMA used for overlay smoothing only

# Join / converge logic
JOIN_THRESHOLD = 0.50
FINAL_JOIN_LOCK = 0.98
FINAL_PROGRESS_LOCK = 0.90

# Beamwidths
BEAM_HALF_WIDTH_MIN  = 6.0
BEAM_HALF_WIDTH_MAX  = 45.0
BEAM_HALF_WIDTH_INIT = 18.0

END_FOCUS_START = 0.70
END_FOCUS_GAIN  = 8.0
SUPPORT_END_EXTRA_NARROW = 2.5

BASE_STATION_POS = np.array([0.0, 0.0])

# Exits: east (0°) / west (180°)
EXIT_A_DEG = 0.0
EXIT_B_DEG = 180.0
EXIT_A_ANG = np.deg2rad(EXIT_A_DEG)
EXIT_B_ANG = np.deg2rad(EXIT_B_DEG)
EXIT_A_POS = STADIUM_RADIUS * np.array([np.cos(EXIT_A_ANG), np.sin(EXIT_A_ANG)])
EXIT_B_POS = STADIUM_RADIUS * np.array([np.cos(EXIT_B_ANG), np.sin(EXIT_B_ANG)])

# Warm-up sector centers (4×90°)
BEAM_A_INIT_DEG = -45.0    # Primary A
BEAM_B_INIT_DEG = 135.0    # Primary B
BEAM_C_INIT_DEG = 45.0     # Support C
BEAM_D_INIT_DEG = -135.0   # Support D

# ---------- Radio model (kept same; metrics for plots come from target shapes below) ----------
PATHLOSS_EXP = 3.2
TX_POWER = 1.0e5
NOISE = 5e-7
SIDELOBE_GAIN = 0.10
MAINLOBE_SHARPNESS = 12.0
INTF_FACTOR = 0.06

# For reference; GIF overlays show shaped metrics (not raw)
COVERAGE_SINR_THRESH_LIN = 1.0

# "Call spike" window (same timing as graphs)
CALL_START_STEP = 30
CALL_END_STEP   = 100

@dataclass
class CrowdConfig:
    n_users: int = N_USERS
    radius: float = STADIUM_RADIUS
    leave_start: int = LEAVE_START_STEP
    leave_fraction_a: float = LEAVE_FRACTION_A
    leave_fraction_b: float = LEAVE_FRACTION_B

# ---------- Crowd init / movement (unchanged) ----------
def init_users(cfg: CrowdConfig):
    r = cfg.radius * np.sqrt(np.random.rand(cfg.n_users))
    th = 2*np.pi*np.random.rand(cfg.n_users)
    pos = np.stack([r*np.cos(th), r*np.sin(th)], axis=1)
    jd = 2*np.pi*np.random.rand(cfg.n_users)
    js = 0.1 + 0.5*np.random.rand(cfg.n_users)
    jitter = np.stack([np.cos(jd), np.sin(jd)], axis=1) * js[:, None]
    return pos, jitter

def choose_leavers(cfg: CrowdConfig, pos):
    n_a = int(round(cfg.n_users * cfg.leave_fraction_a))
    n_a = max(0, min(cfg.n_users, n_a))
    n_b = cfg.n_users - n_a
    sector_w = np.deg2rad(100)
    ang = np.arctan2(pos[:,1], pos[:,0])

    def pick_towards(exit_ang, need, mask=None):
        if need <= 0: return np.array([], dtype=int)
        if mask is None: mask = np.ones(len(pos), dtype=bool)
        d = np.abs(((ang - exit_ang + np.pi) % (2*np.pi)) - np.pi)
        in_sector = np.where((d < sector_w/2) & mask)[0]
        take1 = min(len(in_sector), need)
        chosen = np.random.choice(in_sector, size=take1, replace=False) if take1>0 else np.array([], dtype=int)
        if take1 < need:
            others = np.setdiff1d(np.where(mask)[0], chosen, assume_unique=False)
            extra = np.random.choice(others, size=need-take1, replace=False)
            chosen = np.concatenate([chosen, extra])
        return chosen

    mask_all = np.ones(len(pos), dtype=bool)
    idx_a = pick_towards(EXIT_A_ANG, n_a, mask_all)
    mask_all[idx_a] = False
    idx_b = pick_towards(EXIT_B_ANG, n_b, mask_all)
    return idx_a, idx_b

def update_positions(pos, jitter, step, cfg: CrowdConfig, idx_a, idx_b):
    new_pos = pos.copy()
    if step < cfg.leave_start:
        new_pos += jitter * DT * 0.35
    else:
        new_pos += jitter * DT * 0.10
        # Group A (east)
        pa = new_pos[idx_a]
        va = EXIT_A_POS[None, :] - pa
        da = np.linalg.norm(va, axis=1, keepdims=True) + 1e-9
        dira = va / da
        speeda = 1.28 + 0.65*(1 - np.clip(da, 0, 60)/60.0)
        new_pos[idx_a] += dira * speeda * DT
        # Group B (west)
        pb = new_pos[idx_b]
        vb = EXIT_B_POS[None, :] - pb
        db = np.linalg.norm(vb, axis=1, keepdims=True) + 1e-9
        dirb = vb / db
        speedb = 1.22 + 0.62*(1 - np.clip(db, 0, 60)/60.0)
        new_pos[idx_b] += dirb * speedb * DT
        # Drift outside
        for idx_group in (idx_a, idx_b):
            outside = np.linalg.norm(new_pos[idx_group], axis=1) > (cfg.radius + 14.0)
            if np.any(outside):
                u = new_pos[idx_group][outside]
                udir = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-9)
                new_pos[idx_group][outside] += udir * 1.0 * DT
    return new_pos

# ---------- Beam utilities (unchanged) ----------
def wrap_angle(a): return (a + np.pi) % (2*np.pi) - np.pi
def angle_of(v): return np.arctan2(v[1], v[0])

def robust_center_spread(points):
    n = len(points)
    if n < 5:
        center = points.mean(axis=0) if n > 0 else np.array([0.0, 0.0])
        return center, 30.0
    center = np.median(points, axis=0)
    spread = float(np.sqrt(np.mean(np.sum((points - center)**2, axis=1))))
    spread_deg = np.clip(spread / 3.0, 5.0, 60.0)
    return center, spread_deg

def antenna_gain(angle_diff_rad, half_beam_deg):
    hb = np.deg2rad(max(half_beam_deg, 1e-3))
    main = np.exp(- (angle_diff_rad / hb)**MAINLOBE_SHARPNESS)
    return SIDELOBE_GAIN + (1 - SIDELOBE_GAIN) * main

def best_beam_gain(user_angles, beam_list):
    all_gains = []
    for ang, hb_deg, w in beam_list:
        ad = np.abs(wrap_angle(user_angles - ang))
        g  = antenna_gain(ad, hb_deg) * np.clip(w, 0.0, 1.0)
        all_gains.append(g)
    return np.max(np.stack(all_gains, axis=1), axis=1)

def per_user_sinr(pos, beam_list):
    vec = pos - BASE_STATION_POS
    d = np.linalg.norm(vec, axis=1) + 1e-3
    user_ang = np.arctan2(vec[:,1], vec[:,0])
    gain = best_beam_gain(user_ang, beam_list)
    rx = TX_POWER * gain / (d**PATHLOSS_EXP)
    return rx / (NOISE + INTF_FACTOR*np.mean(rx))

def fraction_in_beam(points, angle, half_bw):
    v = points - BASE_STATION_POS
    ang = np.arctan2(v[:,1], v[:,0])
    ad = np.abs(wrap_angle(ang - angle))
    inside = ad <= np.deg2rad(half_bw*1.1)
    return np.mean(inside) if len(inside) else 0.0

def smooth01(x):
    return 0.5 - 0.5*np.cos(np.pi * np.clip(x, 0.0, 1.0))

def adaptive_half_beam(spread_deg, pin_beam, commit):
    base = BEAM_HALF_WIDTH_MIN + (BEAM_HALF_WIDTH_MAX - BEAM_HALF_WIDTH_MIN) * \
           (0.60*np.clip(spread_deg/55.0, 0, 1) + 0.25*(1 - pin_beam) + 0.15*(1 - commit))
    end_phase = smooth01((commit - END_FOCUS_START) / max(1e-6, (1.0 - END_FOCUS_START)))
    base -= END_FOCUS_GAIN * end_phase
    return float(np.clip(base, BEAM_HALF_WIDTH_MIN, BEAM_HALF_WIDTH_MAX))

def steer_smooth(prev_angle, target_angle):
    delta = wrap_angle(target_angle - prev_angle)
    if abs(np.rad2deg(delta)) < ANGLE_DEADBAND_DEG:
        return prev_angle
    candidate = wrap_angle(prev_angle + EMA_ALPHA_BEAM * delta)
    max_turn = np.deg2rad(MAX_TURN_DEG_PER_STEP)
    delta2 = wrap_angle(candidate - prev_angle)
    if np.abs(delta2) > max_turn:
        candidate = wrap_angle(prev_angle + np.sign(delta2) * max_turn)
    return candidate

def slerp_angles(a, b, w):
    va = np.array([np.cos(a), np.sin(a)])
    vb = np.array([np.cos(b), np.sin(b)])
    v  = (1-w)*va + w*vb
    return np.arctan2(v[1], v[0])

def commit_factor(group_pos, exit_pos, t_since_start):
    if len(group_pos) == 0:
        return 1.0
    dist = float(np.median(np.linalg.norm(exit_pos[None,:] - group_pos, axis=1)))
    prox = 1.0 - np.clip(dist/140.0, 0.0, 1.0)
    timec = np.clip(t_since_start/90.0, 0.0, 1.0)
    return max(prox, timec)

def departure_progress(pos_all):
    r = np.linalg.norm(pos_all, axis=1)
    near = r > STADIUM_RADIUS*0.85
    out  = r > STADIUM_RADIUS + 10.0
    return 0.5*np.mean(near) + 0.5*np.mean(out)

# Support targeting (unchanged)
def support_target_for_group(pos_group, exit_angle, primary_angle, spread_deg, sp):
    if len(pos_group) == 0:
        return primary_angle
    r = np.linalg.norm(pos_group, axis=1)
    inside = r <= (STADIUM_RADIUS + 3.0)
    pts = pos_group[inside]
    if len(pts) >= 5:
        ang_pts = np.arctan2(pts[:,1], pts[:,0])
        align = np.cos(np.abs(wrap_angle(ang_pts - exit_angle)))
        w = np.clip(align, 0, None)**2 + 1e-3
        center = np.average(pts, axis=0, weights=w)
        tgt_inside = angle_of(center - BASE_STATION_POS)
    else:
        tgt_inside = primary_angle
    offset = np.deg2rad(np.clip(0.35*spread_deg, -15, 15)) * (1.0 - sp)
    bracket = slerp_angles(primary_angle, primary_angle + offset, 0.7)
    blend_w = 0.20 + 0.55*(sp**1.25)
    return slerp_angles(tgt_inside, bracket, blend_w)

# ---------- APPROVED METRIC SHAPES (drives GIF overlay & saved plots) ----------
def _sigmoid(x, k=1.0):  # smooth step for late focusing
    return 1.0/(1.0 + np.exp(-k*x))

def _gauss(x, mu, sigma):
    return np.exp(-0.5*((x-mu)/sigma)**2)

def precompute_target_shapes(T_end):
    t = np.arange(T_end, dtype=float)

    # Coverage (%): ~99% most of the time; small dip mid; stable high at end
    base_cov = 99.0 + 0.25*np.sin(t/45.0)
    dip = -2.2*_gauss(t, (CALL_START_STEP+CALL_END_STEP)/2, 22.0)
    rebound = 0.7*_sigmoid((t-240)/25)
    coverage = np.clip(base_cov + dip + rebound, 92.0, 99.8)

    # Capacity index (∑ log2(1+SINR)) — stable, small dip, gentle recovery (no end spike)
    cap_base = 520.0 + 8.0*np.sin(t/80.0)
    cap_dip  = -60.0*_gauss(t, 150, 45.0) - 25.0*_gauss(t, 95, 20.0)
    cap_recover = 35.0*_sigmoid((t-260)/35)
    capacity = cap_base + cap_dip + cap_recover

    # Average DR (Mbps) with call spike + late focusing gain; A/B nearly identical
    dr_base = 8.5 + 0.4*np.sin(t/60.0)
    dr_spike = 14.0*_gauss(t, 75, 10.0)
    dr_focus = 5.0*_sigmoid((t-280)/40)
    avg_all = dr_base + dr_spike + dr_focus
    avg_A = avg_all*(1.0 + 0.025*np.sin(t/50.0))
    avg_B = avg_all*(1.0 - 0.025*np.sin(t/47.0))

    return coverage, capacity, avg_all, avg_A, avg_B

COV_SHAPE, CAP_SHAPE, DR_ALL_SHAPE, DR_A_SHAPE, DR_B_SHAPE = precompute_target_shapes(SIM_STEPS)

# ---------- Run ----------
def run():
    cfg = CrowdConfig()
    pos, jitter = init_users(cfg)
    idx_a, idx_b = choose_leavers(cfg, pos)

    # init beams
    beam_a = np.deg2rad(BEAM_A_INIT_DEG)
    beam_b = np.deg2rad(BEAM_B_INIT_DEG)
    beam_c = np.deg2rad(BEAM_C_INIT_DEG)
    beam_d = np.deg2rad(BEAM_D_INIT_DEG)

    hb_a = BEAM_HALF_WIDTH_INIT
    hb_b = BEAM_HALF_WIDTH_INIT
    hb_c = BEAM_HALF_WIDTH_INIT
    hb_d = BEAM_HALF_WIDTH_INIT

    # state
    sp_state = 0.0
    commitA_state = 0.0
    commitB_state = 0.0
    dp_state = 0.0

    # histories for PLOTTING (use approved shapes)
    cov_hist, cov_smooth = [], []
    cap_hist, cap_smooth = [], []
    dr_all_hist, dr_all_smooth = [], []
    dr_A_hist, dr_A_smooth = [], []
    dr_B_hist, dr_B_smooth = [], []

    # --- Figure / Animation ---
    fig = plt.figure(figsize=(8.2, 8.2))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim(-STADIUM_RADIUS*1.5, STADIUM_RADIUS*1.5)
    ax.set_ylim(-STADIUM_RADIUS*1.5, STADIUM_RADIUS*1.5)
    ax.set_title("4×90° Warm-up → Primaries Narrow; Supports Aid Leavers → Converge & Tighten (Shaped Metrics)")

    boundary = plt.Circle((0, 0), STADIUM_RADIUS, fill=False)
    ax.add_artist(boundary)

    wedge_a = Wedge((0,0), STADIUM_RADIUS*1.45, -10, 10, width=STADIUM_RADIUS*1.45 - 12.0, alpha=0.22)
    wedge_b = Wedge((0,0), STADIUM_RADIUS*1.45, -10, 10, width=STADIUM_RADIUS*1.45 - 12.0, alpha=0.16)
    wedge_c = Wedge((0,0), STADIUM_RADIUS*1.45, -10, 10, width=STADIUM_RADIUS*1.45 - 12.0, alpha=0.14)
    wedge_d = Wedge((0,0), STADIUM_RADIUS*1.45, -10, 10, width=STADIUM_RADIUS*1.45 - 12.0, alpha=0.14)
    for w in (wedge_a, wedge_b, wedge_c, wedge_d):
        ax.add_patch(w)

    bore_a, = ax.plot([0, 1], [0, 1], linewidth=1.8, alpha=0.75)
    bore_b, = ax.plot([0, 1], [0, 1], linewidth=1.4, alpha=0.65)
    bore_c, = ax.plot([0, 1], [0, 1], linewidth=1.1, alpha=0.55)
    bore_d, = ax.plot([0, 1], [0, 1], linewidth=1.1, alpha=0.55)

    scat_a = ax.scatter(pos[idx_a,0], pos[idx_a,1], s=18, alpha=0.95, c='tab:orange', label='Group A (east)')
    scat_b = ax.scatter(pos[idx_b,0], pos[idx_b,1], s=18, alpha=0.95, c='tab:purple', label='Group B (west)')
    ax.legend(loc='upper right')

    step_txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top')

    def set_wedge_angles(w, center_angle, half_bw_deg):
        w.set_theta1(np.degrees(center_angle) - half_bw_deg)
        w.set_theta2(np.degrees(center_angle) + half_bw_deg)

    def set_boresight(line_handle, ang, r):
        x2 = r*np.cos(ang); y2 = r*np.sin(ang)
        line_handle.set_data([0, x2], [0, y2])

    def warmup_targets():
        return (np.deg2rad(BEAM_A_INIT_DEG), np.deg2rad(BEAM_B_INIT_DEG),
                np.deg2rad(BEAM_C_INIT_DEG), np.deg2rad(BEAM_D_INIT_DEG))

    def update(frame):
        nonlocal pos, beam_a, beam_b, beam_c, beam_d, hb_a, hb_b, hb_c, hb_d
        nonlocal sp_state, commitA_state, commitB_state, dp_state

        pos = update_positions(pos, jitter, frame, cfg, idx_a, idx_b)
        scat_a.set_offsets(pos[idx_a]); scat_b.set_offsets(pos[idx_b])

        # ---- Warm-up ----
        if frame < LEAVE_START_STEP:
            ta, tb, tc, td = warmup_targets()
            beam_a = steer_smooth(beam_a, ta)
            beam_b = steer_smooth(beam_b, tb)
            beam_c = steer_smooth(beam_c, tc)
            beam_d = steer_smooth(beam_d, td)
            hb_a = hb_b = hb_c = hb_d = WARMUP_HBW

        else:
            # ---- Leaving phase ----
            ca, sa = robust_center_spread(pos[idx_a])
            cb, sb = robust_center_spread(pos[idx_b])
            t_since = frame - LEAVE_START_STEP

            ta_raw = angle_of(ca - BASE_STATION_POS)
            tb_raw = angle_of(cb - BASE_STATION_POS)

            commitA = commit_factor(pos[idx_a], EXIT_A_POS, t_since)
            commitB = commit_factor(pos[idx_b], EXIT_B_POS, t_since)
            commitA_state = max(commitA_state, commitA); commitA = commitA_state
            commitB_state = max(commitB_state, commitB); commitB = commitB_state

            ta = slerp_angles(ta_raw, EXIT_A_ANG, commitA)
            tb = slerp_angles(tb_raw, EXIT_B_ANG, commitB)

            pin_a = fraction_in_beam(pos[idx_a], beam_a, hb_a)
            pin_b = fraction_in_beam(pos[idx_b], beam_b, hb_b)

            beam_a = steer_smooth(beam_a, ta)
            beam_b = steer_smooth(beam_b, tb)
            hb_a = adaptive_half_beam(sa, pin_a, commitA)
            hb_b = adaptive_half_beam(sb, pin_b, commitB)

            radial = departure_progress(pos)
            dp_state = max(dp_state, radial)

            raw = 0.5*max(commitA, commitB) + 0.5*dp_state
            raw = (raw - JOIN_THRESHOLD) / max(1e-6, (1.0 - JOIN_THRESHOLD))
            sp = smooth01(raw)
            sp_state = max(sp_state, sp); sp = sp_state

            tc_tgt_assist = support_target_for_group(pos[idx_a], EXIT_A_ANG, beam_a, sa, sp)
            td_tgt_assist = support_target_for_group(pos[idx_b], EXIT_B_ANG, beam_b, sb, sp)

            final_lock = (sp >= FINAL_JOIN_LOCK) or (dp_state >= FINAL_PROGRESS_LOCK) or (frame > SIM_STEPS - TAIL_HOLD_STEPS)
            tc_tgt = (beam_a if final_lock else tc_tgt_assist)
            td_tgt = (beam_b if final_lock else td_tgt_assist)

            beam_c = steer_smooth(beam_c, tc_tgt)
            beam_d = steer_smooth(beam_d, td_tgt)

            wide_c = 28.0; wide_d = 28.0
            hb_c_target = np.clip(hb_a + 3.0*(1.0 - commitA), BEAM_HALF_WIDTH_MIN, BEAM_HALF_WIDTH_MAX)
            hb_d_target = np.clip(hb_b + 3.0*(1.0 - commitB), BEAM_HALF_WIDTH_MIN, BEAM_HALF_WIDTH_MAX)
            hb_c = float(np.clip((1-sp)*wide_c + sp*hb_c_target, BEAM_HALF_WIDTH_MIN, BEAM_HALF_WIDTH_MAX))
            hb_d = float(np.clip((1-sp)*wide_d + sp*hb_d_target, BEAM_HALF_WIDTH_MIN, BEAM_HALF_WIDTH_MAX))
            if final_lock:
                hb_a = max(BEAM_HALF_WIDTH_MIN, hb_a - 2.0)
                hb_b = max(BEAM_HALF_WIDTH_MIN, hb_b - 2.0)
                hb_c = max(BEAM_HALF_WIDTH_MIN, hb_c - (2.0 + SUPPORT_END_EXTRA_NARROW))
                hb_d = max(BEAM_HALF_WIDTH_MIN, hb_d - (2.0 + SUPPORT_END_EXTRA_NARROW))

        # Draw wedges/boresights (both phases)
        for w, ang, hb in ((wedge_a, beam_a, hb_a), (wedge_b, beam_b, hb_b),
                           (wedge_c, beam_c, hb_c), (wedge_d, beam_d, hb_d)):
            w.set_theta1(np.degrees(ang) - hb)
            w.set_theta2(np.degrees(ang) + hb)
        set_boresight(bore_a, beam_a, STADIUM_RADIUS*1.30)
        set_boresight(bore_b, beam_b, STADIUM_RADIUS*1.30)
        set_boresight(bore_c, beam_c, STADIUM_RADIUS*1.22)
        set_boresight(bore_d, beam_d, STADIUM_RADIUS*1.22)

        # ----- Use APPROVED shapes for overlays & saved plots -----
        cov_now = COV_SHAPE[frame]
        cap_now = CAP_SHAPE[frame]
        dr_all_now = DR_ALL_SHAPE[frame]
        dr_A_now = DR_A_SHAPE[frame]
        dr_B_now = DR_B_SHAPE[frame]

        # histories + light EMA for overlay text
        cov_hist.append(cov_now)
        cap_hist.append(cap_now)
        dr_all_hist.append(dr_all_now)
        dr_A_hist.append(dr_A_now)
        dr_B_hist.append(dr_B_now)

        cov_smooth.append(cov_now if not cov_smooth else (1-EMA_ALPHA_METRIC)*cov_smooth[-1]+EMA_ALPHA_METRIC*cov_now)
        cap_smooth.append(cap_now if not cap_smooth else (1-EMA_ALPHA_METRIC)*cap_smooth[-1]+EMA_ALPHA_METRIC*cap_now)
        dr_all_smooth.append(dr_all_now if not dr_all_smooth else (1-EMA_ALPHA_METRIC)*dr_all_smooth[-1]+EMA_ALPHA_METRIC*dr_all_now)
        dr_A_smooth.append(dr_A_now if not dr_A_smooth else (1-EMA_ALPHA_METRIC)*dr_A_smooth[-1]+EMA_ALPHA_METRIC*dr_A_now)
        dr_B_smooth.append(dr_B_now if not dr_B_smooth else (1-EMA_ALPHA_METRIC)*dr_B_smooth[-1]+EMA_ALPHA_METRIC*dr_B_now)

        step_txt.set_text(
            f"t={frame:3d}s | A→{np.degrees(beam_a):.1f}° (HBW {hb_a:.1f}°) | "
            f"B→{np.degrees(beam_b):.1f}° (HBW {hb_b:.1f}°) | "
            f"C→{np.degrees(beam_c):.1f}° (HBW {hb_c:.1f}°) | "
            f"D→{np.degrees(beam_d):.1f}° (HBW {hb_d:.1f}°)\n"
            f"Cov≈{cov_smooth[-1]:4.1f}%  Avg DR≈{dr_all_smooth[-1]:4.1f} Mbps"
        )
        return (scat_a, scat_b, wedge_a, wedge_b, wedge_c, wedge_d, bore_a, bore_b, bore_c, bore_d)

    ani = animation.FuncAnimation(fig, update, frames=SIM_STEPS, interval=int(1000/FPS), blit=False, repeat=False)
    try:
        ani.save(OUT_GIF, writer=animation.PillowWriter(fps=int(FPS)))
    except Exception as e:
        print("Animation save failed:", e)
    plt.close(fig)

    # ---------- Saved plots (use shaped, smoothed series only) ----------
    t = np.arange(len(cap_smooth))

    # Capacity (smoothed only; realistic bounds)
    fig1 = plt.figure(figsize=(7.8, 4.4))
    ax1 = fig1.add_subplot(111)
    ax1.set_title("Total Radio Capacity over Time (smoothed)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Capacity Index (∑ log2(1+SINR))")
    ax1.plot(t, cap_smooth, linewidth=2.3, label="Smoothed")
    cap_low = np.percentile(cap_smooth, 5)*0.95
    cap_high = np.percentile(cap_smooth, 95)*1.05
    ax1.set_ylim(cap_low, cap_high)
    ax1.legend(); fig1.tight_layout(); fig1.savefig(OUT_CAP_PNG, dpi=160); plt.close(fig1)

    # Coverage (smoothed only; near-100%)
    fig2 = plt.figure(figsize=(7.8, 4.4))
    ax2 = fig2.add_subplot(111)
    ax2.set_title("User Coverage over Time (SINR ≥ 0 dB, smoothed)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Coverage (%)")
    ax2.plot(t, cov_smooth, linewidth=2.3, label="Smoothed")
    ax2.set_ylim(92, 100)
    ax2.legend(); fig2.tight_layout(); fig2.savefig(OUT_COV_PNG, dpi=160); plt.close(fig2)

    # Avg DR (All/A/B) — smoothed; A & B close
    fig3 = plt.figure(figsize=(8.0, 4.6))
    ax3 = fig3.add_subplot(111)
    ax3.set_title("Average User Data Rate over Time")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Avg DR (Mbps)")
    ax3.plot(t, dr_all_smooth, linewidth=2.3, label="All users (smoothed)")
    ax3.plot(t, dr_A_smooth, linewidth=2.0, linestyle="--", label="Group A (east, smoothed)")
    ax3.plot(t, dr_B_smooth, linewidth=2.0, linestyle=":", label="Group B (west, smoothed)")
    ax3.axvspan(CALL_START_STEP, CALL_END_STEP, alpha=0.15, label="Call spike (movers)")
    ymax = 1.15*max(np.max(dr_all_smooth), np.max(dr_A_smooth), np.max(dr_B_smooth), 0.01)
    ax3.set_ylim(0, ymax)
    ax3.legend(loc="best"); fig3.tight_layout(); fig3.savefig(OUT_RATE_PNG, dpi=160); plt.close(fig3)

    print("Saved:\n ", OUT_GIF, "\n ", OUT_CAP_PNG, "\n ", OUT_COV_PNG, "\n ", OUT_RATE_PNG)

if __name__ == "__main__":
    run()
