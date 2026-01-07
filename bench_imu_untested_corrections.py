
"""
bench_imu_pipeline.py

Single script that:
  - Reads bilateral IMU NDJSON from ESP32-S3 over Serial
  - Has a calibration routine (top bench pose)
  - Reconstructs joint positions using a simple body model
  - Computes MLP features per frame:

        left_elbow_angle,  right_elbow_angle,
        left_shoulder_angle, right_shoulder_angle,
        left_elbow_flare,  right_elbow_flare,
        left_wrist_stack_diff, right_wrist_stack_diff,
        shoulder_diff_3d,
        elbow_angle_asym, shoulder_angle_asym

  - Optionally logs these to a CSV file

Usage example:

    python bench_fin.py --port COM5 --baud 115200 --log imu_raw.jsonl --feat_csv bench_features.csv

Commands in terminal:
    c  -> start calibration (hold top bench pose)
    f  -> finish calibration (prints avg quats)
    q  -> quit
"""

import argparse
import json
import sys
import time
from typing import Dict, Optional
import os
import serial
import numpy as np
import select

# ---------- Node mapping (must match ESP32 node IDs) ----------
NODE_NAME_TO_ID = {
    "chest": 0,
    "shoulder_L": 1,
    "shoulder_R": 2,
    "elbow_L": 3,
    "elbow_R": 4,
}

# ---------- Quaternion math utilities ----------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n

def quat_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q = q1 * q2, both [w,x,y,z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)

def quat_to_R_wxyz(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w,x,y,z] to 3x3 rotation matrix.
    v_global = R @ v_local
    """
    w, x, y, z = quat_normalize(q)
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),         2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),         2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])
    return R

def quat_average_wxyz(samples: np.ndarray) -> np.ndarray:
    """Average quaternions with sign correction (q and -q represent same rotation)."""
    if samples.ndim != 2 or samples.shape[1] != 4:
        raise ValueError("samples must be shape (N,4) [w,x,y,z]")
    ref = samples[0].copy()
    aligned = samples.copy()
    for i in range(aligned.shape[0]):
        if float(np.dot(aligned[i], ref)) < 0.0:
            aligned[i] = -aligned[i]
    return quat_normalize(aligned.mean(axis=0))

def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.array([1, 0, 0, 0], dtype=float)
    axis = axis / n
    s = np.sin(angle_rad / 2.0)
    return quat_normalize(np.array([np.cos(angle_rad / 2.0), axis[0]*s, axis[1]*s, axis[2]*s], dtype=float))

def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """XYZ intrinsic (roll x, pitch y, yaw z) -> quaternion [w,x,y,z]."""
    cr = np.cos(roll/2);  sr = np.sin(roll/2)
    cp = np.cos(pitch/2); sp = np.sin(pitch/2)
    cy = np.cos(yaw/2);   sy = np.sin(yaw/2)

    # q = qx * qy * qz
    qx = np.array([cr, sr, 0, 0], dtype=float)
    qy = np.array([cp, 0, sp, 0], dtype=float)
    qz = np.array([cy, 0, 0, sy], dtype=float)
    return quat_normalize(quat_mul(quat_mul(qx, qy), qz))

def yaw_from_quat_wxyz(q: np.ndarray) -> float:
    """Return yaw (about +Z) from quaternion [w,x,y,z]."""
    w, x, y, z = quat_normalize(q)
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    return float(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))

def rotate_about_z(q: np.ndarray, delta_yaw: float) -> np.ndarray:
    """Apply a world-frame Z rotation to quaternion q: q' = qz(delta)*q."""
    qz = quat_from_axis_angle(np.array([0, 0, 1], dtype=float), delta_yaw)
    return quat_normalize(quat_mul(qz, q))

# ---------- Simple body model + feature extraction ----------

class BodyModel:
    """
    Simple kinematic reconstruction using:
      - chest quaternion (defines torso orientation)
      - upper-arm quaternions (left/right)
      - forearm quaternions (left/right)

    NOTE: This assumes quaternions are already in a consistent world frame.
    """

    def __init__(self):
        # approximate anthropometric lengths (meters)
        self.upper_len = 0.30
        self.fore_len = 0.26

        # local offsets from chest to shoulders (in chest-local frame)
        self.shoulder_offset_y = 0.18
        self.shoulder_offset_z = 0.12

        # local bone direction in segment-local frame
        # +X forward, +Y left, +Z up (consistent with your existing assumptions)
        self.upper_dir_local = np.array([0.0, 0.0, -1.0], dtype=float)  # down in segment local
        self.fore_dir_local  = np.array([0.0, 0.0, -1.0], dtype=float)

    def reconstruct_positions(self,
                              q_chest: np.ndarray,
                              q_Lupper: np.ndarray,
                              q_Rupper: np.ndarray,
                              q_Lfore: np.ndarray,
                              q_Rfore: np.ndarray) -> Dict[str, np.ndarray]:
        # rotation matrices
        R_ch = quat_to_R_wxyz(q_chest)
        R_Lu = quat_to_R_wxyz(q_Lupper)
        R_Ru = quat_to_R_wxyz(q_Rupper)
        R_Lf = quat_to_R_wxyz(q_Lfore)
        R_Rf = quat_to_R_wxyz(q_Rfore)

        # shoulder anchors (world)
        shL_local = np.array([0.0, +self.shoulder_offset_y, +self.shoulder_offset_z], dtype=float)
        shR_local = np.array([0.0, -self.shoulder_offset_y, +self.shoulder_offset_z], dtype=float)

        p_shL = R_ch @ shL_local
        p_shR = R_ch @ shR_local

        # elbows (world)
        p_elL = p_shL + (R_Lu @ (self.upper_len * self.upper_dir_local))
        p_elR = p_shR + (R_Ru @ (self.upper_len * self.upper_dir_local))

        # wrists (world)
        p_wrL = p_elL + (R_Lf @ (self.fore_len * self.fore_dir_local))
        p_wrR = p_elR + (R_Rf @ (self.fore_len * self.fore_dir_local))

        return {
            "chest": np.zeros(3, dtype=float),
            "shL": p_shL, "elL": p_elL, "wrL": p_wrL,
            "shR": p_shR, "elR": p_elR, "wrR": p_wrR,
        }

def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    u = np.asarray(u, dtype=float); v = np.asarray(v, dtype=float)
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))

def compute_features(pos: Dict[str, np.ndarray]) -> Dict[str, float]:
    # vectors
    uL = pos["elL"] - pos["shL"]
    fL = pos["wrL"] - pos["elL"]
    uR = pos["elR"] - pos["shR"]
    fR = pos["wrR"] - pos["elR"]

    # angles
    left_elbow_angle  = angle_between(-uL, fL)
    right_elbow_angle = angle_between(-uR, fR)

    # shoulder angle relative to torso vertical (world Z)
    z = np.array([0.0, 0.0, 1.0], dtype=float)
    left_shoulder_angle  = angle_between(uL, -z)
    right_shoulder_angle = angle_between(uR, -z)

    # elbow flare: how far elbow goes "out" laterally relative to shoulder (Y axis)
    # (very rough; keep consistent with your prior definition)
    left_elbow_flare  = float(pos["elL"][1] - pos["shL"][1])
    right_elbow_flare = float(pos["elR"][1] - pos["shR"][1])

    # wrist stacking: compare wrist x,y relative to elbow (rough surrogate)
    left_wrist_stack_diff  = float(np.linalg.norm((pos["wrL"] - pos["elL"])[:2]))
    right_wrist_stack_diff = float(np.linalg.norm((pos["wrR"] - pos["elR"])[:2]))

    shoulder_diff_3d = float(np.linalg.norm(pos["shL"] - pos["shR"]))

    elbow_angle_asym = float(left_elbow_angle - right_elbow_angle)
    shoulder_angle_asym = float(left_shoulder_angle - right_shoulder_angle)

    return {
        "left_elbow_angle": left_elbow_angle,
        "right_elbow_angle": right_elbow_angle,
        "left_shoulder_angle": left_shoulder_angle,
        "right_shoulder_angle": right_shoulder_angle,
        "left_elbow_flare": left_elbow_flare,
        "right_elbow_flare": right_elbow_flare,
        "left_wrist_stack_diff": left_wrist_stack_diff,
        "right_wrist_stack_diff": right_wrist_stack_diff,
        "shoulder_diff_3d": shoulder_diff_3d,
        "elbow_angle_asym": elbow_angle_asym,
        "shoulder_angle_asym": shoulder_angle_asym,
    }

# ---------- Calibration and mount correction ----------

class CalibrationManager:
    """
    Collect reference quaternions during top bench pose.
    """

    def __init__(self, duration_s: float = 2.0):
        self.duration_s = duration_s
        self.active = False
        self.start_time: Optional[float] = None
        self.samples: Dict[int, list] = {}

    def start(self):
        print("[CAL] Starting calibration: hold ideal top bench position...")
        self.active = True
        self.start_time = time.time()
        self.samples = {}

    def maybe_collect(self, pkt: dict):
        if not self.active or self.start_time is None:
            return
        if time.time() - self.start_time > self.duration_s:
            return

        node = int(pkt.get("node", -1))
        qw = float(pkt.get("qw", 1.0))
        qx = float(pkt.get("qx", 0.0))
        qy = float(pkt.get("qy", 0.0))
        qz = float(pkt.get("qz", 0.0))
        q = quat_normalize(np.array([qw, qx, qy, qz], dtype=float))

        if node not in self.samples:
            self.samples[node] = []
        self.samples[node].append(q)

    def finish(self) -> Dict[str, np.ndarray]:
        self.active = False
        if not self.samples:
            print("[CAL] No samples collected, returning empty reference.")
            return {}

        ref_quats: Dict[str, np.ndarray] = {}
        for name, node_id in NODE_NAME_TO_ID.items():
            if node_id not in self.samples or len(self.samples[node_id]) == 0:
                print(f"[CAL] WARNING: no samples for {name} (node {node_id})")
                continue
            arr = np.stack(self.samples[node_id], axis=0)
            q_mean = quat_average_wxyz(arr)
            ref_quats[name] = q_mean
            print(f"[CAL] {name}: {q_mean}")
        print("[CAL] Calibration finished.")
        return ref_quats

def save_calibration(mount_quats: Dict[int, np.ndarray], path: str = "calibration.json"):
    out = {}
    for node_id, q in mount_quats.items():
        out[str(node_id)] = [float(x) for x in q.tolist()]
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[CAL] Saved mount calibration to {path}")

def load_calibration(path: str = "calibration.json") -> Dict[int, np.ndarray]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        data = json.load(f)
    mount_quats: Dict[int, np.ndarray] = {}
    for k, arr in data.items():
        node_id = int(k)
        q = quat_normalize(np.array(arr, dtype=float))
        mount_quats[node_id] = q
    print(f"[CAL] Loaded mount calibration from {path}")
    return mount_quats

def compute_mount_quats(ref_quats: Dict[str, np.ndarray],
                        desired_by_name: Dict[str, np.ndarray]) -> Dict[int, np.ndarray]:
    """
    Given a calibration reading q_cal (per segment), compute q_mount such that:
        q_segment = q_mount * q_raw
    and if q_raw == q_cal, then q_segment == q_desired.

    So:
        q_mount = q_desired * conj(q_cal)
    """
    mount_quats: Dict[int, np.ndarray] = {}
    for seg_name, q_cal in ref_quats.items():
        if seg_name not in desired_by_name:
            continue
        q_des = desired_by_name[seg_name]
        q_mount = quat_normalize(quat_mul(q_des, quat_conj(q_cal)))
        node_id = NODE_NAME_TO_ID.get(seg_name, None)
        if node_id is not None:
            mount_quats[node_id] = q_mount
    return mount_quats

def apply_mount(node_id: int, q_raw: np.ndarray, mount_quats: Dict[int, np.ndarray]) -> np.ndarray:
    """Apply mount correction if available, otherwise return q_raw.

    Convention: q_segment = q_mount * q_raw.
    With q_mount = q_desired * conj(q_cal), if q_raw == q_cal then q_segment == q_desired.
    """
    if node_id in mount_quats:
        return quat_normalize(quat_mul(mount_quats[node_id], q_raw))
    return q_raw


# ---------- Main loop ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM5)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--log", default=None, help="Append raw NDJSON lines to this file")
    parser.add_argument("--feat_csv", default=None, help="Append computed features to this CSV")
    parser.add_argument("--calib_s", type=float, default=2.0, help="Calibration capture duration (seconds)")
    args = parser.parse_args()

    # Serial open
    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.05)
    except Exception as e:
        print(f"[ERROR] Could not open serial port {args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    log_file = None
    if args.log is not None:
        log_file = open(args.log, "a", buffering=1)

    feat_file = None
    wrote_feat_header = False
    if args.feat_csv is not None:
        # If the file already exists and is non-empty, don't write a header again.
        try:
            wrote_feat_header = os.path.exists(args.feat_csv) and os.path.getsize(args.feat_csv) > 0
        except OSError:
            wrote_feat_header = False
        feat_file = open(args.feat_csv, "a", buffering=1)

    print(f"[INFO] Listening on {args.port} @ {args.baud}")
    print("[INFO] Commands: 'c' (start calib), 'f' (finish calib), 'q' (quit)")

    # latest (smoothed) orientation per node
    q_latest: Dict[int, np.ndarray] = {}

    # Optional raw accel + mag (for potential yaw correction)
    a_latest: Dict[int, np.ndarray] = {}
    m_latest: Dict[int, np.ndarray] = {}

    # EMA smoothing for quats
    quat_alpha = 0.2

    # Body model
    model = BodyModel()

    # Calibration
    calib = CalibrationManager(duration_s=args.calib_s)
    mount_quats: Dict[int, np.ndarray] = load_calibration("calibration.json")

    # Desired pose reference (identity unless you intentionally offset something)
    q_id = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    desired_by_name: Dict[str, np.ndarray] = {
        "chest":     q_id,
        "shoulder_R": q_id,
        "elbow_R":    q_id,
        "shoulder_L": q_id,   # (optional) left-only outward offset (currently identity)
        "elbow_L":    q_id,
    }

    # Feature smoothing / outlier handling
    feats_filt: Dict[str, float] = {}
    feat_smooth_alpha: float = 0.2     # EMA on feature values
    max_angle_jump_deg: float = 35.0   # reject sudden jumps in angles/asym
    max_stack_jump: float = 0.10       # reject sudden jumps in wrist stack scalar

    last_stdin_check = time.time()
    last_print = time.time()
    json_error_count = 0

    while True:
        # 1) Read from serial
        line = ser.readline()
        if line:
            raw = line.strip()
            if raw:
                try:
                    s = raw.decode("utf-8", errors="strict")
                    pkt = json.loads(s)
                except Exception:
                    json_error_count += 1
                    # Print the first few parse errors to help debugging, then stay quiet.
                    if json_error_count <= 5:
                        print(f"[WARN] Dropped malformed JSON line: {raw!r}", file=sys.stderr)
                    continue

                # raw log
                if log_file is not None:
                    log_file.write(s + "\\n")

                node = int(pkt.get("node", -1))
                if node in NODE_NAME_TO_ID.values():
                    # --- raw quaternion from ESP (Madgwick output) ---
                    qw = float(pkt.get("qw", 1.0))
                    qx = float(pkt.get("qx", 0.0))
                    qy = float(pkt.get("qy", 0.0))
                    qz = float(pkt.get("qz", 0.0))
                    q_raw = quat_normalize(np.array([qw, qx, qy, qz], dtype=float))

                    # EMA smooth (simple, keeps continuity)
                    if node in q_latest:
                        # align sign to avoid flips during EMA
                        if float(np.dot(q_latest[node], q_raw)) < 0.0:
                            q_raw = -q_raw
                        q_latest[node] = quat_normalize((1.0 - quat_alpha) * q_latest[node] + quat_alpha * q_raw)
                    else:
                        q_latest[node] = q_raw

                    # Optional accel/mag
                    if "ax" in pkt and "ay" in pkt and "az" in pkt:
                        a_latest[node] = np.array([float(pkt["ax"]), float(pkt["ay"]), float(pkt["az"])], dtype=float)
                    if "mx" in pkt and "my" in pkt and "mz" in pkt:
                        m_latest[node] = np.array([float(pkt["mx"]), float(pkt["my"]), float(pkt["mz"])], dtype=float)

                    # feed calibration collection
                    calib.maybe_collect(pkt)

        # 2) keyboard commands
        now = time.time()
        if now - last_stdin_check > 0.05:
            last_stdin_check = now
            if select.select([sys.stdin], [], [], 0)[0]:
                cmd = sys.stdin.readline().strip().lower()
                if cmd == "c":
                    calib.start()
                elif cmd == "f":
                    ref_quats = calib.finish()
                    if not ref_quats:
                        continue
                    mount_quats = compute_mount_quats(ref_quats, desired_by_name)
                    for name, node_id in NODE_NAME_TO_ID.items():
                        if node_id in mount_quats:
                            print(f"[CAL] mount[{name}] (node {node_id}) = {mount_quats[node_id]}")
                    save_calibration(mount_quats)
                elif cmd == "q":
                    print("[INFO] Quitting...")
                    break

        # 3) If we have all required segments, compute features
        have_all = all(k in q_latest for k in (
            NODE_NAME_TO_ID["chest"],
            NODE_NAME_TO_ID["shoulder_L"],
            NODE_NAME_TO_ID["shoulder_R"],
            NODE_NAME_TO_ID["elbow_L"],
            NODE_NAME_TO_ID["elbow_R"],
        ))

        if have_all:
            # Apply mount corrections with correct convention: q_seg = q_mount * q_raw
            q_ch = apply_mount(NODE_NAME_TO_ID["chest"],      q_latest[NODE_NAME_TO_ID["chest"]],      mount_quats)
            q_Lu = apply_mount(NODE_NAME_TO_ID["shoulder_L"], q_latest[NODE_NAME_TO_ID["shoulder_L"]], mount_quats)
            q_Ru = apply_mount(NODE_NAME_TO_ID["shoulder_R"], q_latest[NODE_NAME_TO_ID["shoulder_R"]], mount_quats)
            q_Lf = apply_mount(NODE_NAME_TO_ID["elbow_L"],    q_latest[NODE_NAME_TO_ID["elbow_L"]],    mount_quats)
            q_Rf = apply_mount(NODE_NAME_TO_ID["elbow_R"],    q_latest[NODE_NAME_TO_ID["elbow_R"]],    mount_quats)

            # Reconstruct positions and compute features
            pos = model.reconstruct_positions(q_ch, q_Lu, q_Ru, q_Lf, q_Rf)
            feats = compute_features(pos)

            # --- Feature smoothing (EMA) + outlier rejection ---
            if not feats_filt:
                feats_filt = {k: float(v) for k, v in feats.items()}
            else:
                for k, v in feats.items():
                    v = float(v)
                    prev = feats_filt.get(k, v)

                    # reject big jumps on angles/asym
                    if k in (
                        "left_elbow_angle", "right_elbow_angle",
                        "left_shoulder_angle", "right_shoulder_angle",
                        "elbow_angle_asym", "shoulder_angle_asym",
                    ):
                        if abs(v - prev) > max_angle_jump_deg:
                            continue

                    # reject big jumps on stack metrics
                    if k in ("left_wrist_stack_diff", "right_wrist_stack_diff"):
                        if abs(v - prev) > max_stack_jump:
                            continue

                    feats_filt[k] = (1.0 - feat_smooth_alpha) * prev + feat_smooth_alpha * v

            out = feats_filt if feats_filt else feats

            # 5) Console print at ~5 Hz
            if now - last_print > 0.2:
                last_print = now
                print(
                    f"[FEAT] L_elb={out['left_elbow_angle']:.1f}° "
                    f"R_elb={out['right_elbow_angle']:.1f}°  "
                    f"L_wrist_stack={out['left_wrist_stack_diff']:.3f} "
                    f"R_wrist_stack={out['right_wrist_stack_diff']:.3f}  "
                    f"elbow_asym={out['elbow_angle_asym']:.1f}° "
                    f"shoulder_asym={out['shoulder_angle_asym']:.1f}°"
                )

            # 6) Log to feature CSV if requested
            if feat_file is not None:
                if not wrote_feat_header:
                    header_cols = ["t_s"] + list(out.keys())
                    feat_file.write(",".join(header_cols) + "\n")
                    wrote_feat_header = True

                t_s = now
                values = [f"{t_s:.6f}"] + [f"{out[k]:.6f}" for k in out.keys()]
                feat_file.write(",".join(values) + "\n")

    if log_file is not None:
        log_file.close()
    if feat_file is not None:
        feat_file.close()
    ser.close()

if __name__ == "__main__":
    main()
