import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation

# Replay 3D animation for IMU Jacket data
# Usage: python replay_3d.py run1_raw.jsonl

# ---- IMPORTANT: align with bench_fin.py (your live pipeline) ----
# bench_fin.py defines BodyModel + quat_mul + quat_normalize exactly as used online
from bench_fin import BodyModel, quat_mul, quat_normalize  # <-- aligned import

# ================== CONFIG ==================

CHEST_ID      = 4
SHOULDER_R_ID = 3
ELBOW_R_ID    = 2
SHOULDER_L_ID = 5
ELBOW_L_ID    = 6

FRAME_DT = 0.02  # seconds per frame (~50 Hz replay)

# --- Visual “lying on bench” staging (purely for display) ---
BODY_Z_OFFSET = 0.18   # lifts skeleton above bench pad so it doesn't clip
BENCH_Z       = 0.0

# Trunk/head proportions (meters) — purely visual, not used for ML/features
TORSO_HALF_LEN = 0.45  # chest -> hips along body axis
NECK_LEN       = 0.12  # chest -> neck
HEAD_LEN       = 0.18  # neck -> head top

# Bench dimensions (meters)
BENCH_LEN  = 1.90
BENCH_W    = 0.35
BENCH_THK  = 0.06

# Uprights (very simple)
UPRIGHT_Y_FROM_CHEST = 0.35
UPRIGHT_H            = 0.55
UPRIGHT_W            = 0.50  # distance between uprights (x)

# ================== LOAD CALIBRATION (optional) ==================

mount_quat = {}

def load_calibration(path="calibration.json"):
    global mount_quat
    try:
        with open(path, "r") as f:
            data = json.load(f)
        mount_quat = {int(k): np.array(v, dtype=float) for k, v in data.items()}
        print(f"[REPLAY] Loaded calibration for nodes: {sorted(mount_quat.keys())}")
    except FileNotFoundError:
        print("[REPLAY] No calibration.json found. Using raw quaternions.")
        mount_quat = {}
    except Exception as e:
        print("[REPLAY] Failed to load calibration.json:", e)
        print("[REPLAY] Using raw quaternions.")
        mount_quat = {}

def apply_mount(node_id, q_raw):
    # Matches your bench_fin pipeline convention: q_corrected = q_raw * q_mount
    if node_id in mount_quat:
        return quat_normalize(quat_mul(mount_quat[node_id], q_raw))
    return quat_normalize(q_raw)

# ================== LOAD LOG & BUILD FRAMES ==================

def build_frames(log_path):
    """
    Read NDJSON log and build a list of frames.
    Each frame is (t_s, pos_dict) where pos_dict has:
        p_chest_center, p_shL, p_elL, p_wrL, p_shR, p_elR, p_wrR
    """
    body = BodyModel()
    frames = []

    q_latest = {}     # node_id -> quat (after mount)
    t0 = None
    last_frame_t = None
    have_all_once = False

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pkt = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Extract time (controller or node timestamp)
            t_us = pkt.get("t_ctrl_us", pkt.get("t_node_us", None))
            if t_us is None:
                continue
            if t0 is None:
                t0 = t_us
            t_s = (t_us - t0) * 1e-6

            # Extract node + quat
            node = int(pkt.get("node", -1))
            if node < 0:
                continue

            qw = float(pkt.get("qw", 1.0))
            qx = float(pkt.get("qx", 0.0))
            qy = float(pkt.get("qy", 0.0))
            qz = float(pkt.get("qz", 0.0))
            q_raw = np.array([qw, qx, qy, qz], dtype=float)
            q_latest[node] = apply_mount(node, q_raw)

            # Decide whether to create a new frame
            if last_frame_t is None:
                last_frame_t = t_s

            if t_s - last_frame_t >= FRAME_DT:
                needed = {CHEST_ID, SHOULDER_L_ID, SHOULDER_R_ID, ELBOW_L_ID, ELBOW_R_ID}
                if not have_all_once:
                    if not needed.issubset(set(q_latest.keys())):
                        continue
                    have_all_once = True

                if CHEST_ID not in q_latest:
                    continue

                q_ch  = q_latest[CHEST_ID]
                q_sL  = q_latest.get(SHOULDER_L_ID, q_ch)
                q_eL  = q_latest.get(ELBOW_L_ID,    q_sL)
                q_sR  = q_latest.get(SHOULDER_R_ID, q_ch)
                q_eR  = q_latest.get(ELBOW_R_ID,    q_sR)

                pos = body.reconstruct_positions(q_ch, q_sL, q_eL, q_sR, q_eR)
                frames.append((t_s, pos))
                last_frame_t = t_s

    print(f"[REPLAY] Built {len(frames)} frames from log.")
    return frames

# ================== DRAW HELPERS ==================

def draw_bench(ax, chest_point):
    """
    Draw a simple bench centered under the person.
    We use chest_point only to place the bench nicely in the scene.
    """
    # Center bench so chest sits around the upper half of the pad
    cx, cy, _ = chest_point
    bench_center = np.array([cx, cy - 0.10, BENCH_Z])  # slight shift so head is closer to top

    # Bench pad corners (rectangle in X-Y plane at z=BENCH_Z)
    L = BENCH_LEN
    W = BENCH_W
    x0 = bench_center[0] - W/2
    x1 = bench_center[0] + W/2
    y0 = bench_center[1] - L/2
    y1 = bench_center[1] + L/2

    # Draw top perimeter
    ax.plot([x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            [BENCH_Z, BENCH_Z, BENCH_Z, BENCH_Z, BENCH_Z],
            linewidth=2)

    # Draw a “thickness” outline (simple vertical edges)
    z2 = BENCH_Z - BENCH_THK
    for (xx, yy) in [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]:
        ax.plot([xx, xx], [yy, yy], [BENCH_Z, z2], linewidth=1)

    ax.plot([x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            [z2, z2, z2, z2, z2],
            linewidth=1)

    # Uprights (very minimal): two vertical posts near where bar would be
    uy = cy + UPRIGHT_Y_FROM_CHEST
    ux0 = cx - UPRIGHT_W/2
    ux1 = cx + UPRIGHT_W/2
    ax.plot([ux0, ux0], [uy, uy], [BENCH_Z, BENCH_Z + UPRIGHT_H], linewidth=3)
    ax.plot([ux1, ux1], [uy, uy], [BENCH_Z, BENCH_Z + UPRIGHT_H], linewidth=3)

def compute_trunk_and_head(pC):
    """
    Create a simple lying-down trunk and head line.
    We treat +Y as “towards head” (along the bench), -Y as towards hips.
    """
    p_hips = pC + np.array([0.0, -TORSO_HALF_LEN, 0.0])
    p_neck = pC + np.array([0.0, +NECK_LEN,       0.0])
    p_head = p_neck + np.array([0.0, +HEAD_LEN,   0.0])
    return p_hips, p_neck, p_head

# ================== ANIMATION ==================

def animate_frames(frames):
    if not frames:
        print("[REPLAY] No frames to animate.")
        return

    fig = plt.figure("IMU Jacket Replay (Bench Press)")
    ax = fig.add_subplot(111, projection='3d')

    def setup_axes():
        # Wider Y for bench length; moderate Z since person is lying down
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-1.3, 1.3)
        ax.set_zlim(-0.35, 0.9)
        ax.set_xlabel("X (left/right)")
        ax.set_ylabel("Y (bench length)")
        ax.set_zlabel("Z (up)")
        ax.view_init(elev=18, azim=-60)

    def init():
        setup_axes()
        ax.set_title("Upper-body skeleton (offline replay)")
        return []

    def update(frame_idx):
        t_s, pos = frames[frame_idx]

        # --- base joints (from BodyModel) ---
        pC   = pos["p_chest_center"].copy()
        p_shL = pos["p_shL"].copy()
        p_elL = pos["p_elL"].copy()
        p_wrL = pos["p_wrL"].copy()
        p_shR = pos["p_shR"].copy()
        p_elR = pos["p_elR"].copy()
        p_wrR = pos["p_wrR"].copy()

        # --- visual-only offset so the person is above the bench pad ---
        lift = np.array([0.0, 0.0, BODY_Z_OFFSET])
        pC   += lift
        p_shL += lift; p_elL += lift; p_wrL += lift
        p_shR += lift; p_elR += lift; p_wrR += lift

        # trunk/head
        p_hips, p_neck, p_head = compute_trunk_and_head(pC)

        ax.cla()
        setup_axes()
        ax.set_title(f"Bench Press Replay (t = {t_s:.2f} s)")

        # --- Bench (draw first) ---
        draw_bench(ax, pC - lift)  # bench sits at BENCH_Z; use un-lifted chest for placement

        # --- Trunk & head ---
        ax.plot([p_hips[0], pC[0], p_neck[0], p_head[0]],
                [p_hips[1], pC[1], p_neck[1], p_head[1]],
                [p_hips[2], pC[2], p_neck[2], p_head[2]],
                linewidth=3)

        # --- chest to shoulders ---
        ax.plot([pC[0], p_shL[0]], [pC[1], p_shL[1]], [pC[2], p_shL[2]], linewidth=2)
        ax.plot([pC[0], p_shR[0]], [pC[1], p_shR[1]], [pC[2], p_shR[2]], linewidth=2)

        # --- left arm ---
        ax.plot([p_shL[0], p_elL[0], p_wrL[0]],
                [p_shL[1], p_elL[1], p_wrL[1]],
                [p_shL[2], p_elL[2], p_wrL[2]], linewidth=3)

        # --- right arm ---
        ax.plot([p_shR[0], p_elR[0], p_wrR[0]],
                [p_shR[1], p_elR[1], p_wrR[1]],
                [p_shR[2], p_elR[2], p_wrR[2]], linewidth=3)

        # --- points ---
        pts = np.stack([p_head, p_neck, pC, p_hips, p_shL, p_elL, p_wrL, p_shR, p_elR, p_wrR], axis=0)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=25)

        return []

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        init_func=init,
        interval=FRAME_DT * 1000,
        blit=False,
        repeat=True,
    )

    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python replay_3d.py run1_raw.jsonl")
        sys.exit(1)

    log_path = sys.argv[1]
    load_calibration("calibration.json")
    frames = build_frames(log_path)
    animate_frames(frames)

if __name__ == "__main__":
    main()
