"""Shared GEAR-SONIC bridge-validation plan: the SAME segments, start, timing and statistics in MuJoCo
(sonic_mujoco_ref.py) and Isaac (sonic_isaac_val.py). numpy only (importable from Isaac kit python).

Per segment: ground start (default pose, soles on the floor, 0.1 s default-pose PD hold) -> controller reset
(planner enabled, IDLE) -> SETTLE s idle -> command for CMD s; statistics over the last half of the command phase,
from one row per 50 Hz control tick (logged after the 4 physics sub-steps of that tick)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math
import numpy as np

SETTLE = 2.0
CMD = 10.0
HOLD_STEPS = 20          # 0.1 s default-pose PD hold at 200 Hz before control starts
START_PELVIS_Z = 0.7574907452582463   # export_sonic_npz.py: default pose with soles on z = 0

# name: (mode, SonicController.set_command kwargs)
SEGMENTS = {
    "idle": ("idle", dict()),
    "slow_walk_0.3": ("slow_walk", dict(vx=0.3)),
    "slow_walk_0.5": ("slow_walk", dict(vx=0.5)),
    "slow_walk_0.8": ("slow_walk", dict(vx=0.8)),
    "walk_default": ("walk", dict(vx=1.0, speed=-1.0)),
    "run_1.5": ("run", dict(vx=1.5)),
    "lateral_left_0.4": ("slow_walk", dict(vy=0.4)),
    "lateral_right_0.4": ("slow_walk", dict(vy=-0.4)),
    "backward_0.4": ("slow_walk", dict(vx=-0.4)),
    "turn_left_idle_0.5": ("idle", dict(wz=0.5)),
    "turn_right_idle_0.5": ("idle", dict(wz=-0.5)),
    "walk_turn_0.5_0.3": ("slow_walk", dict(vx=0.5, wz=0.3)),
    "squat_h0.2": ("squat", dict(height=0.2)),
    "squat_h0.4": ("squat", dict(height=0.4)),
    "kneel_two_legs_h0.2": ("kneel_two_legs", dict(height=0.2)),
    "crouch_walk_default": ("crouch_walk", dict(vx=1.0, speed=-1.0)),
    "crouch_walk_0.5": ("crouch_walk", dict(vx=0.5)),
    "hand_crawl_0.6_h0.8": ("hand_crawl", dict(vx=0.6, height=0.8)),
    "hand_crawl_0.4_h0.2": ("hand_crawl", dict(vx=0.4, height=0.2)),
    "elbow_crawl_0.8_h0.8": ("elbow_crawl", dict(vx=0.8, height=0.8)),
    "elbow_crawl_0.7_h0.2": ("elbow_crawl", dict(vx=0.7, height=0.2)),
}
CRAWL = {"hand_crawl", "elbow_crawl"}


def row(t, pos, quat_wxyz, v_world, w_body, R_torso, head_top, ref_z=float("nan"), robot_top=float("nan"),
        torso_floor=float("nan")):
    """One log row. pos/quat: pelvis world pose (z relative to the floor); v_world: pelvis linear velocity (world);
    w_body: pelvis angular velocity (body frame); R_torso: torso_link world rotation; head_top relative to floor."""
    w, x, y, z = quat_wxyz
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    yaw = math.atan2(R[1, 0], R[0, 0]); c, s = math.cos(yaw), math.sin(yaw)
    ww = R @ np.asarray(w_body, float)
    Rt = np.asarray(R_torso, float)
    return dict(t=t, x=float(pos[0]), y=float(pos[1]), z=float(pos[2]), yaw=yaw,
                vx=c * v_world[0] + s * v_world[1], vy=-s * v_world[0] + c * v_world[1], wz=float(ww[2]),
                head_top=float(head_top), robot_top=float(robot_top),
                torso_pitch=math.atan2(-Rt[2, 0], math.hypot(Rt[2, 1], Rt[2, 2])),
                pelvis_pitch=math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])), torso_up_z=float(Rt[2, 2]),
                ref_z=float(ref_z), torso_floor=float(torso_floor))


def summarize(name, log, n_settle, n_cmd):
    mode, kw = SEGMENTS[name]
    L = log[n_settle + n_cmd // 2:]
    full = log[n_settle:]
    arr = lambda key: np.array([e[key] for e in L], float)
    head_min = min(e["head_top"] for e in full)
    up_min = min(e["torso_up_z"] for e in full)
    if mode in CRAWL:
        fall = bool(up_min < -0.2 or head_min < 0.12)
        crit = "crawl: torso up-axis z < -0.2 (upside down) or head_top < 0.12 m"
    else:
        fall = bool(head_min < 0.45 or up_min < 0.3)
        crit = "head_top < 0.45 m or torso up-axis z < 0.3"
    nanmean = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else None
    yaw = np.unwrap([e["yaw"] for e in full])
    return dict(segment=name, mode=mode, command=kw, settle_s=n_settle * 0.02, cmd_s=n_cmd * 0.02,
                complete=len(log) == n_settle + n_cmd,
                vx_mean=float(arr("vx").mean()), vy_mean=float(arr("vy").mean()), wz_mean=float(arr("wz").mean()),
                speed_planar_mean=float(np.hypot(arr("vx"), arr("vy")).mean()),
                pelvis_z_mean=float(arr("z").mean()), pelvis_z_min=float(arr("z").min()),
                head_top_mean=float(arr("head_top").mean()), head_top_min=float(arr("head_top").min()),
                head_top_max=float(arr("head_top").max()),
                robot_top_mean=nanmean(arr("robot_top")),
                torso_pitch_deg_mean=float(np.degrees(arr("torso_pitch")).mean()),
                pelvis_pitch_deg_mean=float(np.degrees(arr("pelvis_pitch")).mean()),
                ref_pelvis_z_mean=nanmean(arr("ref_z")),
                torso_or_pelvis_floor_contact_frac=nanmean((arr("torso_floor") > 0).astype(float))
                if np.isfinite(arr("torso_floor")).any() else None,
                displacement_xy_cmd_phase=[float(log[-1]["x"] - full[0]["x"]), float(log[-1]["y"] - full[0]["y"])],
                yaw_change_cmd_phase_deg=float(np.degrees(yaw[-1] - yaw[0])),
                min_head_top_cmd_phase=float(head_min), min_pelvis_z_cmd_phase=float(min(e["z"] for e in full)),
                fall=fall, fall_criterion=crit)
