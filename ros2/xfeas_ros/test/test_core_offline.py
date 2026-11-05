"""Offline test of the XFeas core with the real robot configs, no ROS and no robot.

For G1 + GEAR-SONIC and Spot + ReLIC: load config/<robot>.yaml, feed 3 s of a synthetic robot standing in an empty hall
(flat floor seen by its head camera, joint / odometry streams) and ask the real model server (CPU). Checks the camera
convention (the floor lands at z = 0), that every configured joint name is known to the model, and that a command comes
back. Needs the repo's data/ (controller specs and self-test probes).

    XFEAS_ROOT=/path/to/XFeas python ros2/xfeas_ros/test/test_core_offline.py
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from xfeas_ros import core as XC  # noqa: E402

ROOT = os.environ.setdefault("XFEAS_ROOT", os.path.abspath(os.path.join(HERE, "..", "..", "..")))
os.environ.setdefault("XFEAS_PY", sys.executable)
CAMERA = dict(g1_sonic=(1.2, 27.0, 0.79, 1.30), spot_relic=(0.62, 26.0, 0.52, 0.97))  # cam z, pitch, base z, body top


def head_camera(cam_z, pitch_deg):
    """optical-frame pose (R world<-optical, t) of a camera at (0, 0, cam_z) looking +x, pitched down"""
    p = math.radians(pitch_deg)
    fwd = np.array([math.cos(p), 0.0, -math.sin(p)])
    right = np.array([0.0, -1.0, 0.0])
    return np.stack([right, np.cross(fwd, right), fwd], 1), np.array([0.0, 0.0, cam_z])


def floor_depth(R_opt, t, N=224):
    """z-depth image of the plane z = 0 in the simulator camera model"""
    f = XC.SIM_F * N
    u, v = np.meshgrid(np.arange(N) + 0.5, np.arange(N) + 0.5)
    rays = np.stack([(u - N / 2) / f, (v - N / 2) / f, np.ones_like(u)], -1) @ R_opt.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(rays[..., 2] < -1e-6, -t[2] / rays[..., 2], np.inf).astype(np.float32)


def run(robot):
    cfg = XC.load_config(os.path.join(HERE, "..", "config", f"{robot}.yaml"))
    cfg.update(device="cpu", cuda_device="", model_log=None)
    core = XC.XFeasCore(cfg)
    import gen4_common as G

    unknown = [n for n in cfg["joint_names"] if n not in G.CANON_JOINTS]
    assert not unknown, f"{robot}: joint names the model does not know: {unknown}"

    cam_z, pitch, base_z, top = CAMERA[robot]
    R, t = head_camera(cam_z, pitch)
    M = XC.camera_matrix_from_optical(R, t)
    dep = floor_depth(R, t)
    P = core.FM.mem_points([(core.CLM.min_pool_depth(np.clip(dep, 0, 20), 128), M, (0.0, 0.0, 0.0))], (0.0, 0.0, 0.0), 0.0,
                           self_x=0.0)
    z = np.median(np.asarray(P)[:, 2])
    print(f"[test] {robot}: floor points {len(P)}, z median {z:.3f} m (expect ~0)")
    assert abs(z) < 0.05, "camera convention: the floor is not at z = 0"

    core.start_model()
    core.set_goal((8.0, 0.0))
    q0, rgb, ch = np.asarray(cfg["default_pos"]), np.full((224, 224, 3), 90, np.uint8), None
    for k in range(150):  # 3 s at 50 Hz, standing still
        tt = k / 50.0
        core.on_base(tt, (0.0, 0.0, base_z), (0, 0, 0, 1), (0, 0, 0), (0, 0, 0))
        core.on_joints(tt, q0, np.zeros_like(q0))
        core.on_body(top_above_floor=top, floor_z=0.0)
        if k % 5 == 0:
            core.on_frame(tt, rgb, dep, M)
        ch = core.decide(tt) or ch
    core.client.close()
    assert ch is not None, "no decision was made"
    cmd = core.command(tt)
    print(f"[test] {robot}: class {ch['cls']} (model alone: {ch.get('model_cls')}), skill {ch['rule']['skill']}, "
          f"U {ch['U']:.3f}, {len(ch['table'])} candidates -> {cmd}")
    assert {"lin_vel_x", "lin_vel_y", "yaw_rate"} <= set(cmd)


if __name__ == "__main__":
    rs = XC.CameraResampler([[615.0, 0, 320.0], [0, 615.0, 240.0], [0, 0, 1]], 640, 480)
    print(f"[test] a D435i-like 640x480 colour camera covers {100 * rs.coverage:.0f} % of the simulator image")
    for r in sys.argv[1:] or ("g1_sonic", "spot_relic"):
        run(r)
    print("[test] OK")
