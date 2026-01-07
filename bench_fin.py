#!/usr/bin/env python3
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

    python bench_fin.py --port COM29 --baud 115200 --log imu_raw.jsonl --feat_csv bench_features.csv

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

# ---------- Node mapping (must match ESP32-S3 firmware) ----------

def save_calibration(mount_quats, path="calibration.json"):
    """
    Save mount_quats to a JSON file.

    mount_quats: dict {node_id (int) -> quat (np.ndarray or list of 4)}
    JSON format: { "2": [w, x, y, z], "3": [...], ... }
    """
    data = {}
    for node_id, q in mount_quats.items():
        q = np.asarray(q, dtype=float)  # ensure it's an array of floats
        data[str(node_id)] = q.tolist()

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[CAL] Saved {len(data)} mount quats to {path}")

NODE_NAME_TO_ID = {
    "elbow_R":    2,   # forearm IMU (right)
    "shoulder_R": 3,   # upper arm IMU (right)
    "chest":      4,   # chest IMU
    "shoulder_L": 5,   # upper arm IMU (left)
    "elbow_L":    6,   # forearm IMU (left)
}
CHEST_ID      = NODE_NAME_TO_ID["chest"]
SHOULDER_R_ID = NODE_NAME_TO_ID["shoulder_R"]
ELBOW_R_ID    = NODE_NAME_TO_ID["elbow_R"]
SHOULDER_L_ID = NODE_NAME_TO_ID["shoulder_L"]
ELBOW_L_ID    = NODE_NAME_TO_ID["elbow_L"]


# ---------- Quaternion + geometry helpers ----------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n

def quat_conj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate [w,x,y,z] → [w,-x,-y,-z]."""
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

def quat_slerp(q_prev: np.ndarray, q_new: np.ndarray, alpha: float) -> np.ndarray:
    """
    Spherical linear interpolation between quaternions.
    q_prev, q_new: [w,x,y,z]
    alpha in [0,1], weight toward q_new.
    """
    q_prev = quat_normalize(q_prev)
    q_new  = quat_normalize(q_new)

    dot = float(np.dot(q_prev, q_new))
    # Take shortest path
    if dot < 0.0:
        q_new = -q_new
        dot = -dot

    # If very close, fall back to lerp
    if dot > 0.9995:
        q = (1.0 - alpha) * q_prev + alpha * q_new
        return quat_normalize(q)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    w1 = np.sin((1.0 - alpha) * theta) / sin_theta
    w2 = np.sin(alpha * theta) / sin_theta
    return quat_normalize(w1 * q_prev + w2 * q_new)

def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Angle between v1 and v2 in DEGREES.
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return np.nan
    cosang = float(np.dot(v1, v2) / (n1 * n2))
    cosang = max(-1.0, min(1.0, cosang))
    return np.degrees(np.arccos(cosang))

def wrap_to_pi(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

def yaw_from_quat_wxyz(q: np.ndarray) -> float:
    """
    Extract yaw (rotation around Z) from quaternion [w,x,y,z]
    using ZYX (yaw-pitch-roll) convention.
    """
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))

def quat_z(delta_yaw: float) -> np.ndarray:
    """Quaternion for rotation of delta_yaw around +Z axis."""
    half = 0.5 * delta_yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)

# ---------- Simple upper-body model ----------

class BodyModel:
    """
    Simple symmetric upper body model with:
      - chest origin at (0,0,0)
      - shoulders at (+/- shoulder_width/2, 0, 0) in chest frame
      - upper arms & forearms pointing along local +Y at neutral

    Units in meters.
    """

    def __init__(self,
                 shoulder_width: float = 0.40,
                 upper_arm_length: float = 0.30,
                 forearm_length: float = 0.25):
        self.shoulder_width  = shoulder_width
        self.upper_arm_length = upper_arm_length
        self.forearm_length   = forearm_length
        # local "down" (towards elbow) direction for bone in its own frame
        self.upper_dir_local = np.array([0.0, 1.0, 0.0])
        self.fore_dir_local  = np.array([0.0, 1.0, 0.0])

    def reconstruct_positions(self,
                              q_chest: np.ndarray,
                              q_Lupper: np.ndarray,
                              q_Lfore: np.ndarray,
                              q_Rupper: np.ndarray,
                              q_Rfore: np.ndarray):
        """
        Given segment quaternions, reconstruct joint positions
        in the chest frame:

          p_chest_center, p_shL, p_elL, p_wrL, p_shR, p_elR, p_wrR
        """
        R_chest  = quat_to_R_wxyz(q_chest)
        R_Lupper = quat_to_R_wxyz(q_Lupper)
        R_Lfore  = quat_to_R_wxyz(q_Lfore)
        R_Rupper = quat_to_R_wxyz(q_Rupper)
        R_Rfore  = quat_to_R_wxyz(q_Rfore)

        p_chest_center = np.zeros(3)

        half_sw = self.shoulder_width / 2.0
        shoulder_L_local = np.array([+half_sw, 0.0, 0.0])
        shoulder_R_local = np.array([-half_sw, 0.0, 0.0])

        p_shL = R_chest @ shoulder_L_local + p_chest_center
        p_shR = R_chest @ shoulder_R_local + p_chest_center

        upper_vec_local = self.upper_dir_local * self.upper_arm_length
        fore_vec_local  = self.fore_dir_local  * self.forearm_length

        p_elL = p_shL + (R_Lupper @ upper_vec_local)
        p_wrL = p_elL + (R_Lfore  @ fore_vec_local)

        p_elR = p_shR + (R_Rupper @ upper_vec_local)
        p_wrR = p_elR + (R_Rfore  @ fore_vec_local)

        return {
            "p_chest_center": p_chest_center,
            "p_shL": p_shL,
            "p_elL": p_elL,
            "p_wrL": p_wrL,
            "p_shR": p_shR,
            "p_elR": p_elR,
            "p_wrR": p_wrR,
        }


def compute_mlp_features_from_positions(pos: dict) -> Dict[str, float]:
    """
    Compute all MLP features from joint positions:

    - left_elbow_angle,  right_elbow_angle
    - left_shoulder_angle, right_shoulder_angle
    - left_elbow_flare,  right_elbow_flare
    - left_wrist_stack_diff, right_wrist_stack_diff
    - shoulder_diff_3d
    - elbow_angle_asym, shoulder_angle_asym
    """
    p_chest = pos["p_chest_center"]
    p_shL   = pos["p_shL"]
    p_elL   = pos["p_elL"]
    p_wrL   = pos["p_wrL"]
    p_shR   = pos["p_shR"]
    p_elR   = pos["p_elR"]
    p_wrR   = pos["p_wrR"]

    # Segment vectors
    uL = p_elL - p_shL   # upper arm L
    fL = p_wrL - p_elL   # forearm L
    uR = p_elR - p_shR   # upper arm R
    fR = p_wrR - p_elR   # forearm R

    # Elbow angles (joint angle at elbow)
    left_elbow_angle  = angle_between(uL, fL)
    right_elbow_angle = angle_between(uR, fR)

    # Shoulder angles (angle between trunk and upper arm)
    trunk_L = p_shL - p_chest
    trunk_R = p_shR - p_chest
    left_shoulder_angle  = angle_between(trunk_L, uL)
    right_shoulder_angle = angle_between(trunk_R, uR)

    # Elbow flare (mirror CV code: abs(x_elbow - x_shoulder) - abs(x_wrist - x_shoulder))
    x_shL, x_elL, x_wrL = p_shL[0], p_elL[0], p_wrL[0]
    x_shR, x_elR, x_wrR = p_shR[0], p_elR[0], p_wrR[0]

    left_elbow_flare  = abs(x_elL - x_shL) - abs(x_wrL - x_shL)
    right_elbow_flare = abs(x_elR - x_shR) - abs(x_wrR - x_shR)

    # Wrist stacking (dist in y,z between wrist and elbow)
    dy_L = p_wrL[1] - p_elL[1]
    dz_L = p_wrL[2] - p_elL[2]
    left_wrist_stack_diff = float(np.sqrt(dy_L**2 + dz_L**2))

    dy_R = p_wrR[1] - p_elR[1]
    dz_R = p_wrR[2] - p_elR[2]
    right_wrist_stack_diff = float(np.sqrt(dy_R**2 + dz_R**2))

    # Shoulder stability (3D diff in y,z between L & R shoulders)
    dy_sh = p_shR[1] - p_shL[1]
    dz_sh = p_shR[2] - p_shL[2]
    shoulder_diff_3d = float(np.sqrt(dy_sh**2 + dz_sh**2))

    # Asymmetries
    elbow_angle_asym    = abs(left_elbow_angle  - right_elbow_angle)
    shoulder_angle_asym = abs(left_shoulder_angle - right_shoulder_angle)

    feats = {
        "left_elbow_angle":       float(left_elbow_angle),
        "right_elbow_angle":      float(right_elbow_angle),
        "left_shoulder_angle":    float(left_shoulder_angle),
        "right_shoulder_angle":   float(right_shoulder_angle),
        "left_elbow_flare":       float(left_elbow_flare),
        "right_elbow_flare":      float(right_elbow_flare),
        "left_wrist_stack_diff":  left_wrist_stack_diff,
        "right_wrist_stack_diff": right_wrist_stack_diff,
        "shoulder_diff_3d":       shoulder_diff_3d,
        "elbow_angle_asym":       float(elbow_angle_asym),
        "shoulder_angle_asym":    float(shoulder_angle_asym),
    }
    return feats


# ---------- Calibration manager ----------

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

    def update(self, pkt: dict):
        if not self.active or self.start_time is None:
            return

        now = time.time()
        if now - self.start_time > self.duration_s:
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
            ref = arr[0]
            for i in range(arr.shape[0]):
                if np.dot(arr[i], ref) < 0:
                    arr[i] = -arr[i]
            q_mean = quat_normalize(arr.mean(axis=0))
            ref_quats[name] = q_mean
            print(f"[CAL] {name}: {q_mean}")
        print("[CAL] Calibration finished.")
        return ref_quats


# ---------- Main loop ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM5 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--log",  type=str, default=None, help="Optional NDJSON log file path")
    parser.add_argument("--feat_csv", type=str, default=None,
                        help="Optional CSV to log per-frame MLP features")
    args = parser.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"[ERROR] Could not open serial port {args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    log_file = None
    if args.log is not None:
        log_file = open(args.log, "a", buffering=1)

    feat_file = None
    wrote_feat_header = False
    if args.feat_csv is not None:
        feat_file = open(args.feat_csv, "a", buffering=1)

    print(f"[INFO] Listening on {args.port} @ {args.baud}")
    print("[INFO] Commands: 'c' (start calib), 'f' (finish calib), 'q' (quit)")

    # latest (smoothed) orientation per node
    q_latest: Dict[int, np.ndarray] = {}
    a_latest: Dict[int, np.ndarray] = {}

    # Orientation smoothing (quaternions)
    q_filt: Dict[int, np.ndarray] = {}
    quat_smooth_alpha: float = 0.25  # 0=freeze, 1=no smoothing; try 0.2–0.3

    # Feature smoothing / outlier handling
    feats_filt: Dict[str, float] = {}
    feat_smooth_alpha: float = 0.2     # EMA on feature values
    max_angle_jump_deg: float = 60.0   # ignore single-frame angle jumps > this

    # Calibration + mount
    cal = CalibrationManager(duration_s=2.0)
    ref_quats: Dict[str, np.ndarray] = {}
    mount_quats: Dict[int, np.ndarray] = {}

    body_model = BodyModel()

    # Chest magnetometer (latest reading) and yaw-drift correction
    m_chest_latest: Optional[np.ndarray] = None
    yaw_err_filt: float = 0.0          # filtered yaw error (rad)
    yaw_alpha: float = 0.02            # smaller = slower, more stable

    # --- Desired segment orientations in top-bench pose (Option 3) ---
    # Identity for most, slight outward yaw for left upper arm
    q_id = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


    desired_by_name: Dict[str, np.ndarray] = {
        "chest":      q_id,
        "shoulder_R": q_id,
        "elbow_R":    q_id,
        "shoulder_L": q_id,   # left-only outward offset
        "elbow_L":    q_id,
    }

    last_stdin_check = time.time()
    last_print = time.time()

    while True:
        # 1) Read from serial
        line = ser.readline()
        if line:
            line = line.strip()
            if line:
                try:
                    pkt = json.loads(line.decode("utf-8"))
                except Exception:
                    continue

                # raw log
                if log_file is not None:
                    log_file.write(line.decode("utf-8") + "\n")

                node = int(pkt.get("node", -1))
                if node in NODE_NAME_TO_ID.values():
                    # --- raw quaternion from ESP (Madgwick output) ---
                    qw = float(pkt.get("qw", 1.0))
                    qx = float(pkt.get("qx", 0.0))
                    qy = float(pkt.get("qy", 0.0))
                    qz = float(pkt.get("qz", 0.0))
                    q_raw = quat_normalize(np.array([qw, qx, qy, qz], dtype=float))

                    # --- quaternion smoothing (EMA via slerp) ---
                    if node in q_filt:
                        q_filt[node] = quat_slerp(q_filt[node], q_raw, quat_smooth_alpha)
                    else:
                        q_filt[node] = q_raw

                    q_latest[node] = q_filt[node]

                    # raw accel (still useful if we need it later)
                    ax = float(pkt.get("ax", 0.0))
                    ay = float(pkt.get("ay", 0.0))
                    az = float(pkt.get("az", 0.0))
                    a_latest[node] = np.array([ax, ay, az], dtype=float)

                    # --- chest magnetometer (only for CHEST_ID) ---
                    if node == CHEST_ID:
                        mx = float(pkt.get("mx", 0.0))
                        my = float(pkt.get("my", 0.0))
                        mz = float(pkt.get("mz", 0.0))
                        if not (mx == 0.0 and my == 0.0 and mz == 0.0):
                            m_chest_latest = np.array([mx, my, mz], dtype=float)

                # calibration
                cal.update(pkt)

        now = time.time()

        # 2) Poll stdin for commands (cross-platform)
        if now - last_stdin_check > 0.2:
            last_stdin_check = now

            cmd = None

            # On Windows, use msvcrt for non-blocking keyboard input
            if os.name == "nt":
                import msvcrt
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    cmd = ch.strip()
            else:
                # POSIX: we can use select() on stdin
                import select
                rlist, _, _ = select.select([sys.stdin], [], [], 0)
                if rlist:
                    cmd = sys.stdin.readline().strip()

            if cmd == "c":
                cal.start()
            elif cmd == "f":
                # Finish calibration and compute mount corrections
                ref_quats = cal.finish()
                print(f"[INFO] Stored ref_quats for: {list(ref_quats.keys())}")

                mount_quats.clear()
                for name, node_id in NODE_NAME_TO_ID.items():
                    if name not in ref_quats:
                        print(f"[CAL] No ref quat for {name}, skipping mount.")
                        continue

                    q_cal = ref_quats[name]
                    q_desired = desired_by_name.get(name, q_id)
                    # IMU->bone mapping: q_segment = q_desired when IMU reads q_cal
                    q_mount = quat_mul(q_desired, quat_conj(q_cal))
                    mount_quats[node_id] = quat_normalize(q_mount)
                    print(f"[CAL] mount_quat[{name}] (node {node_id}) = {mount_quats[node_id]}")
                save_calibration(mount_quats)
            elif cmd == "q":
                print("[INFO] Quitting...")
                break

        def apply_mount(node_id: int, q_raw: np.ndarray) -> np.ndarray:
            """Apply mount correction if available, otherwise return q_raw."""
            if node_id in mount_quats:
                return quat_normalize(quat_mul(mount_quats[node_id], q_raw))
            return q_raw

        # 3) If we have all required segments, compute features
        have_all = all(k in q_latest for k in (
            CHEST_ID, SHOULDER_L_ID, ELBOW_L_ID, SHOULDER_R_ID, ELBOW_R_ID))

        if have_all:
            q_chest_raw  = q_latest[CHEST_ID]
            q_Lupper_raw = q_latest[SHOULDER_L_ID]
            q_Lfore_raw  = q_latest[ELBOW_L_ID]
            q_Rupper_raw = q_latest[SHOULDER_R_ID]
            q_Rfore_raw  = q_latest[ELBOW_R_ID]

            # --- 1) Apply mount corrections ---
            q_chest  = apply_mount(CHEST_ID,      q_chest_raw)
            q_Lupper = apply_mount(SHOULDER_L_ID, q_Lupper_raw)
            q_Lfore  = apply_mount(ELBOW_L_ID,    q_Lfore_raw)
            q_Rupper = apply_mount(SHOULDER_R_ID, q_Rupper_raw)
            q_Rfore  = apply_mount(ELBOW_R_ID,    q_Rfore_raw)

            # --- 2) Global yaw drift correction using chest magnetometer ---
            if m_chest_latest is not None:
                psi_q = yaw_from_quat_wxyz(q_chest)

                R_ch = quat_to_R_wxyz(q_chest)
                m_w = R_ch @ m_chest_latest
                mx_w, my_w, mz_w = m_w

                psi_m = float(np.arctan2(my_w, mx_w))
                dpsi = wrap_to_pi(psi_q - psi_m)

                yaw_err_filt = (1.0 - yaw_alpha) * yaw_err_filt + yaw_alpha * dpsi

                q_corr = quat_z(-yaw_err_filt)

                # Apply same global yaw correction to all segments
                q_chest  = quat_mul(q_corr, q_chest)
                q_Lupper = quat_mul(q_corr, q_Lupper)
                q_Lfore  = quat_mul(q_corr, q_Lfore)
                q_Rupper = quat_mul(q_corr, q_Rupper)
                q_Rfore  = quat_mul(q_corr, q_Rfore)

            # --- 2.6) Reduce RELATIVE yaw drift forearm vs its own upper arm ---
            k_elbow = 0.7  # 0=no effect, 1=forearm yaw == upper arm yaw

            def align_forearm_to_upper(q_upper: np.ndarray,
                                       q_fore: np.ndarray,
                                       k: float) -> np.ndarray:
                psi_u = yaw_from_quat_wxyz(q_upper)
                psi_f = yaw_from_quat_wxyz(q_fore)
                dpsi_uf = wrap_to_pi(psi_f - psi_u)
                q_align = quat_z(-k * dpsi_uf)
                return quat_mul(q_align, q_fore)

            q_Lfore = align_forearm_to_upper(q_Lupper, q_Lfore, k_elbow)
            q_Rfore = align_forearm_to_upper(q_Rupper, q_Rfore, k_elbow)

            # --- 3) Body model on yaw-corrected quats ---
            pos = body_model.reconstruct_positions(
                q_chest, q_Lupper, q_Lfore, q_Rupper, q_Rfore
            )
            feats = compute_mlp_features_from_positions(pos)

            # --- 4) Feature smoothing (EMA) + outlier rejection ---
            if not feats_filt:
                feats_filt = {k: float(v) for k, v in feats.items()}
            else:
                for k, v in feats.items():
                    v = float(v)
                    prev = feats_filt.get(k, v)

                    if k in (
                        "left_elbow_angle", "right_elbow_angle",
                        "left_shoulder_angle", "right_shoulder_angle",
                    ):
                        if abs(v - prev) > max_angle_jump_deg:
                            continue

                    feats_filt[k] = (1.0 - feat_smooth_alpha) * prev + feat_smooth_alpha * v

            # 5) Console print at ~5 Hz
            if now - last_print > 0.2:
                last_print = now
                print(
                    f"[FEAT] L_elb={feats['left_elbow_angle']:.1f}° "
                    f"R_elb={feats['right_elbow_angle']:.1f}°  "
                    f"L_wrist_stack={feats['left_wrist_stack_diff']:.3f} "
                    f"R_wrist_stack={feats['right_wrist_stack_diff']:.3f}  "
                    f"elbow_asym={feats['elbow_angle_asym']:.1f}° "
                    f"shoulder_asym={feats['shoulder_angle_asym']:.1f}°"
                )

            # 6) Log to feature CSV if requested
            if feat_file is not None:
                if not wrote_feat_header:
                    header_cols = ["t_s"] + list(feats.keys())
                    feat_file.write(",".join(header_cols) + "\n")
                    wrote_feat_header = True

                t_s = now
                values = [f"{t_s:.6f}"] + [f"{feats[k]:.6f}" for k in feats.keys()]
                feat_file.write(",".join(values) + "\n")

    if log_file is not None:
        log_file.close()
    if feat_file is not None:
        feat_file.close()
    ser.close()

if __name__ == "__main__":
    main()
