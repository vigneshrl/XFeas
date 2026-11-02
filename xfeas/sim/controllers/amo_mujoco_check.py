#!/usr/bin/env python
"""
Reference check of the RELEASED AMO policy in its OWN deployment simulator (MuJoCo, play_amo.py), headless with
scripted commands. The observation/action/PD code below is play_amo.py's HumanoidEnv, copied line for line
(viewer and keyboard removed, arm toggle off); commands are set per segment instead of by keys.

  $PLAYGROUND_PY amo_mujoco_check.py [--video out.mp4] [--json out.json]

commands vector (play_amo.py): [vx, target_yaw, vy, height_offset (height = 0.75 + offset), torso_yaw,
torso_pitch, torso_roll, arm_toggle]
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import argparse, json, os, sys
from collections import deque
import numpy as np
import torch

AMO = os.environ.get("AMO_DIR", xfeas.paths.BASELINES + "/AMO")


def quatToEuler(quat):
    qw, qx, qy, qz = quat
    e = np.zeros(3)
    e[0] = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    sinp = 2 * (qw * qy - qz * qx)
    e[1] = np.copysign(np.pi / 2, sinp) if np.abs(sinp) >= 1 else np.arcsin(sinp)
    e[2] = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return e


class AMOController:
    """play_amo.py HumanoidEnv observation / action logic, simulator-agnostic.
    Inputs per control tick: dof_pos(23), dof_vel(23) in AMO order, quat (w,x,y,z) and gyro of imu_in_pelvis."""

    stiffness = np.array(
        [150, 150, 150, 300, 80, 20, 150, 150, 150, 300, 80, 20, 400, 400, 400, 80, 80, 40, 60, 80, 80, 40, 60], float
    )
    damping = np.array([2, 2, 2, 4, 2, 1, 2, 2, 2, 4, 2, 1, 15, 15, 15, 2, 2, 1, 1, 2, 2, 1, 1], float)
    default_dof_pos = np.array(
        [
            -0.1,
            0.0,
            0.0,
            0.3,
            -0.2,
            0.0,
            -0.1,
            0.0,
            0.0,
            0.3,
            -0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.0,
            0.2,
            0.3,
            0.5,
            0.0,
            -0.2,
            0.3,
        ]
    )
    torque_limits = np.array(
        [88, 139, 88, 139, 50, 50, 88, 139, 88, 139, 50, 50, 88, 50, 50, 25, 25, 25, 25, 25, 25, 25, 25], float
    )
    dof_names = [
        "left_hip_pitch",
        "left_hip_roll",
        "left_hip_yaw",
        "left_knee",
        "left_ankle_pitch",
        "left_ankle_roll",
        "right_hip_pitch",
        "right_hip_roll",
        "right_hip_yaw",
        "right_knee",
        "right_ankle_pitch",
        "right_ankle_roll",
        "waist_yaw",
        "waist_roll",
        "waist_pitch",
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
    ]
    num_actions, num_dofs, action_scale = 15, 23, 0.25
    sim_dt, sim_decimation = 0.002, 10
    control_dt = sim_dt * sim_decimation

    def __init__(self, device=None):
        # play_amo.py: device = "cuda" if available; the released jit hard-codes a cuda:0 tensor, so CPU cannot run it
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = self.device
        self.policy_jit = torch.jit.load(os.path.join(AMO, "amo_jit.pt"), map_location=device)
        self.adapter = torch.jit.load(os.path.join(AMO, "adapter_jit.pt"), map_location=device).eval()
        ns = torch.load(
            os.path.join(AMO, "adapter_norm_stats.pt"), weights_only=False
        )  # released file holds numpy arrays (play_amo.py: torch 2.3 default)
        f = lambda k: torch.tensor(ns[k], device=device, dtype=torch.float32)
        self.input_mean, self.input_std, self.output_mean, self.output_std = (
            f("input_mean"),
            f("input_std"),
            f("output_mean"),
            f("output_std"),
        )
        self.scales_ang_vel, self.scales_dof_vel = 0.25, 0.05
        self.nj, self.n_priv = 23, 3
        self.n_proprio = 3 + 2 + 2 + 23 * 3 + 2 + 15
        self.history_len, self.extra_history_len, self._n_demo_dof = 10, 25, 8
        self.commands = np.zeros(8, dtype=np.float32)
        self.reset()

    def reset(self):
        self.last_action = np.zeros(self.nj)
        self.demo_obs_template = np.zeros((8 + 3 + 3 + 3,))
        self.demo_obs_template[: self._n_demo_dof] = self.default_dof_pos[15:]
        self.demo_obs_template[self._n_demo_dof + 6 : self._n_demo_dof + 9] = 0.75
        self._in_place_stand_flag = True
        self.gait_cycle = np.array([0.25, 0.25])
        self.gait_freq = 1.3
        self.proprio_history_buf = deque([np.zeros(self.n_proprio)] * self.history_len, maxlen=self.history_len)
        self.extra_history_buf = deque(
            [np.zeros(self.n_proprio)] * self.extra_history_len, maxlen=self.extra_history_len
        )
        self.arm_action = self.default_dof_pos[15:].copy()
        self.prev_arm_action = self.default_dof_pos[15:].copy()
        self.arm_blend = 0.0
        self.pd_target = self.default_dof_pos.copy()

    def get_observation(self, dof_pos, dof_vel, quat, ang_vel):
        c = self.commands
        rpy = quatToEuler(quat)
        dyaw = rpy[2] - c[1]
        dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
        if self._in_place_stand_flag:
            dyaw = 0.0
        obs_dof_vel = dof_vel.copy()
        obs_dof_vel[[4, 5, 10, 11, 13, 14]] = 0.0  # computed but (as in play_amo.py) NOT used below
        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)
        ai = np.concatenate([np.zeros(4), dof_pos[15:]])
        ai[0], ai[1], ai[2], ai[3] = 0.75 + c[3], c[4], c[5], c[6]
        ai = torch.tensor(ai).to(self.device, dtype=torch.float32).unsqueeze(0)
        ai = (ai - self.input_mean) / (self.input_std + 1e-8)
        with torch.no_grad():
            ao = self.adapter(ai.view(1, -1)) * self.output_std + self.output_mean
        obs_prop = np.concatenate(
            [
                ang_vel * self.scales_ang_vel,
                rpy[:2],
                (np.sin(dyaw), np.cos(dyaw)),
                dof_pos - self.default_dof_pos,
                dof_vel * self.scales_dof_vel,
                self.last_action,
                gait_obs,
                ao.cpu().numpy().squeeze(),
            ]
        )
        obs_priv = np.zeros((self.n_priv,))
        obs_hist = np.array(self.proprio_history_buf).flatten()
        obs_demo = self.demo_obs_template.copy()
        obs_demo[: self._n_demo_dof] = dof_pos[15:]
        obs_demo[self._n_demo_dof] = c[0]
        obs_demo[self._n_demo_dof + 1] = c[2]
        self._in_place_stand_flag = np.abs(c[0]) < 0.1
        obs_demo[self._n_demo_dof + 3] = c[4]
        obs_demo[self._n_demo_dof + 4] = c[5]
        obs_demo[self._n_demo_dof + 5] = c[6]
        obs_demo[self._n_demo_dof + 6 : self._n_demo_dof + 9] = 0.75 + c[3]
        self.proprio_history_buf.append(obs_prop)
        self.extra_history_buf.append(obs_prop)
        return np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist))

    def control_tick(self, dof_pos, dof_vel, quat, ang_vel):
        """One 50 Hz tick of play_amo.py run(); returns the PD target (23)."""
        obs = self.get_observation(dof_pos, dof_vel, quat, ang_vel)
        with torch.no_grad():
            extra = (
                torch.tensor(np.array(self.extra_history_buf).flatten().copy(), dtype=torch.float)
                .view(1, -1)
                .to(self.device)
            )
            raw = (
                self.policy_jit(torch.from_numpy(obs).float().unsqueeze(0).to(self.device), extra)
                .cpu()
                .numpy()
                .squeeze()
            )
        raw = np.clip(raw, -40.0, 40.0)
        self.last_action = np.concatenate([raw.copy(), (dof_pos - self.default_dof_pos)[15:] / self.action_scale])
        pd = np.concatenate([raw * self.action_scale, np.zeros(8)]) + self.default_dof_pos
        pd[15:] = (1 - self.arm_blend) * self.prev_arm_action + self.arm_blend * self.arm_action
        self.arm_blend = min(1.0, self.arm_blend + 0.01)
        self.gait_cycle = np.remainder(self.gait_cycle + self.control_dt * self.gait_freq, 1.0)
        if self._in_place_stand_flag and (
            (np.abs(self.gait_cycle[0] - 0.25) < 0.05) or (np.abs(self.gait_cycle[1] - 0.25) < 0.05)
        ):
            self.gait_cycle = np.array([0.25, 0.25])
        if (not self._in_place_stand_flag) and (
            (np.abs(self.gait_cycle[0] - 0.25) < 0.05) and (np.abs(self.gait_cycle[1] - 0.25) < 0.05)
        ):
            self.gait_cycle = np.array([0.25, 0.75])
        self.pd_target = pd
        return pd

    def torque(self, dof_pos, dof_vel):
        t = (self.pd_target - dof_pos) * self.stiffness - dof_vel * self.damping
        return np.clip(t, -self.torque_limits, self.torque_limits)


SEGMENTS = [  # (label, commands[8] , seconds)
    ("stand", [0, 0, 0, 0, 0, 0, 0, 0], 6.0),
    ("vx 0.3", [0.3, 0, 0, 0, 0, 0, 0, 0], 6.0),
    ("vx 0.5", [0.5, 0, 0, 0, 0, 0, 0, 0], 6.0),
    ("vx -0.3", [-0.3, 0, 0, 0, 0, 0, 0, 0], 6.0),
    ("vy 0.3 (vx 0)", [0, 0, 0.3, 0, 0, 0, 0, 0], 6.0),
    ("vy 0.3 (vx 0.15)", [0.15, 0, 0.3, 0, 0, 0, 0, 0], 6.0),
    ("turn: target yaw +0.8 (vx 0.2)", [0.2, 0.8, 0, 0, 0, 0, 0, 0], 6.0),
    ("height -0.15 (0.60)", [0, 0, 0, -0.15, 0, 0, 0, 0], 6.0),
    ("height -0.30 (0.45)", [0, 0, 0, -0.30, 0, 0, 0, 0], 6.0),
    ("height -0.45 (0.30)", [0, 0, 0, -0.45, 0, 0, 0, 0], 6.0),
    ("height +0.05 (0.80)", [0, 0, 0, 0.05, 0, 0, 0, 0], 6.0),
    ("torso pitch +0.5", [0, 0, 0, 0, 0, 0.5, 0, 0], 6.0),
    ("torso pitch +1.0", [0, 0, 0, 0, 0, 1.0, 0, 0], 6.0),
    ("torso pitch -0.3", [0, 0, 0, 0, 0, -0.3, 0, 0], 6.0),
    ("torso roll +0.4", [0, 0, 0, 0, 0, 0, 0.4, 0], 6.0),
    ("torso roll -0.4", [0, 0, 0, 0, 0, 0, -0.4, 0], 6.0),
    ("torso yaw +0.8", [0, 0, 0, 0, 0.8, 0, 0, 0], 6.0),
    ("walk 0.3 + height -0.15", [0.3, 0, 0, -0.15, 0, 0, 0, 0], 6.0),
    ("walk 0.3 + pitch 0.3", [0.3, 0, 0, 0, 0, 0.3, 0, 0], 6.0),
]


def torso_rpy_world(m, d):  # noqa
    """torso_link orientation in the world (roll, pitch, yaw) -- what the torso commands act on."""
    q = d.xquat[m.body("torso_link").id]
    return quatToEuler(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    import mujoco

    globals()['mujoco'] = mujoco
    m = mujoco.MjModel.from_xml_path(os.path.join(AMO, "g1.xml"))
    m.opt.timestep = AMOController.sim_dt
    d = mujoco.MjData(m)
    C = AMOController()
    renderer = None
    frames = []
    if a.video:
        renderer = mujoco.Renderer(m, 480, 640)
        cam = mujoco.MjvCamera()
        cam.distance = 2.8
        cam.elevation = -12
        cam.azimuth = 135
    res = {}
    pel = m.body("pelvis").id
    for label, cmd, dur in SEGMENTS:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_step(m, d)
        C.reset()
        C.commands[:] = 0
        n = int(dur / C.sim_dt)
        settle = int(2.0 / C.sim_dt)  # 2 s standing before each segment
        rows, fell = [], False
        for i in range(settle + n):
            if i == settle:
                C.commands[:] = np.array(cmd, np.float32)
                y0 = quatToEuler(d.sensor("orientation").data)[2]
                if cmd[1] != 0:
                    C.commands[1] = y0 + cmd[1]  # target yaw is ABSOLUTE in play_amo.py
            q = d.qpos.astype(np.float32)[-23:]
            dq = d.qvel.astype(np.float32)[-23:]
            quat = d.sensor("orientation").data.astype(np.float32)
            gyr = d.sensor("angular-velocity").data.astype(np.float32)
            if i % C.sim_decimation == 0:
                C.control_tick(q, dq, quat, gyr)
                if renderer is not None and i >= settle and (i // C.sim_decimation) % 2 == 0:
                    cam.lookat[:] = d.qpos[:3]
                    renderer.update_scene(d, cam)
                    im = renderer.render().copy()
                    import cv2

                    cv2.putText(im, f"MuJoCo AMO: {label}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(
                        im,
                        f"pelvis z {d.qpos[2]:.2f}  vx {d.qvel[0]:+.2f}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    frames.append(im)
            d.ctrl[:] = C.torque(q, dq)
            mujoco.mj_step(m, d)
            if d.qpos[2] < 0.3 or not np.isfinite(d.qpos).all():
                fell = True
                break
            if i >= settle + n // 2 and i % 10 == 0:
                R = d.xmat[pel].reshape(3, 3)
                v_b = R.T @ d.qvel[:3]
                rows.append(
                    dict(
                        z=float(d.qpos[2]),
                        vx=float(v_b[0]),
                        vy=float(v_b[1]),
                        wz=float(d.qvel[5]),
                        pelvis_rpy=quatToEuler(d.qpos[3:7]).tolist(),
                        torso_rpy=torso_rpy_world(m, d).tolist(),
                        waist=d.qpos[7 + 12 : 7 + 15].tolist(),
                    )
                )
        r = dict(cmd=cmd, fell=fell)
        if rows and not fell:
            for k in ("z", "vx", "vy", "wz"):
                r[k] = float(np.mean([x[k] for x in rows]))
            for k in ("pelvis_rpy", "torso_rpy", "waist"):
                r[k] = np.mean([x[k] for x in rows], 0).tolist()
            r["yaw_change"] = float(quatToEuler(d.qpos[3:7])[2] - y0)
        res[label] = r
        print(
            f"[amo-mj] {label:32s} fell={fell} "
            + " ".join(f"{k}={r[k]:+.3f}" for k in ("z", "vx", "vy", "wz") if k in r)
            + (
                f" torso_rpy={np.round(r['torso_rpy'], 3).tolist()} pelvis_rpy={np.round(r['pelvis_rpy'], 3).tolist()}"
                f" waist(y,r,p)={np.round(r['waist'], 3).tolist()} dyaw={r['yaw_change']:+.2f}"
                if "z" in r
                else ""
            ),
            flush=True,
        )
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    if frames:
        import imageio_ffmpeg

        gen = imageio_ffmpeg.write_frames(
            a.video, (640, 480), fps=25, codec="libx264", pix_fmt_in="rgb24", output_params=["-crf", "20"]
        )
        gen.send(None)
        for f in frames:
            gen.send(np.ascontiguousarray(f))
        gen.close()
        print("[amo-mj] video", a.video)


if __name__ == "__main__":
    main()
