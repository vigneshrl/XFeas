"""Headless MuJoCo sim2sim reference for GEAR-SONIC using the release MJCF + sonic_ref.SonicController.

Reproduces gear_sonic/scripts/run_sim_loop.py physics (gear_sonic/utils/mujoco_sim/base_sim.py):
  scene gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml, timestep 0.005 (sim_frequency=200,
  configs.py:139 -> SIMULATE_DT), PD torque tau = kp*(q*-q) + kd*(0-dq) recomputed every physics step
  (base_sim.py:240-273), clipped to motor_effort_limit_list (wbc_configs/g1_29dof_sonic_model12.yaml),
  elastic band on pelvis (unitree_sdk2py_bridge.py:366-395) during the 3 s INIT ramp
  (g1_deploy_onnx_ref.cpp:2762-2794), then dropped ('9' key) after control start
  (--start band; NOTE: this falls in our runs because the policy flails while hanging, so the default
  is --start ground: robot placed on the floor in the default pose, 0.1 s PD hold, then control).
The C++ deploy binary itself (TensorRT + CUDA graphs + DDS) is NOT run (no TensorRT here).

Usage: python sonic_mujoco_ref.py --segments idle slow_walk_0.3 ... --out_json X --video_dir D
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sonic_ref import (SonicController, KPS, KDS, DEFAULT_ANGLES, DEX3_LEFT_DEFAULT, DEX3_RIGHT_DEFAULT,  # noqa: E402
                       DEX3_KP, DEX3_KD, MODES)

REPO = xfeas_paths.BASELINES + "/GR00T-WholeBodyControl"
SCENE = f"{REPO}/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml"
WBC_YAML = f"{REPO}/gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml"
SIM_DT = 0.005
DECIM = 4  # 50 Hz control

from sonic_plan import SEGMENTS, CRAWL, HOLD_STEPS, START_PELVIS_Z, row, summarize  # noqa: E402


def quat_to_mat(q):
    m = np.zeros(9); mujoco.mju_quat2Mat(m, np.asarray(q, dtype=np.float64)); return m.reshape(3, 3)


class Sim:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.m.opt.timestep = SIM_DT
        self.d = mujoco.MjData(self.m)
        m = self.m
        names = [m.joint(i).name for i in range(m.njnt)]
        self.body_j = [i for i, n in enumerate(names) if any(p in n for p in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"])]
        self.lh_j = [i for i, n in enumerate(names) if "left_hand" in n]
        self.rh_j = [i for i, n in enumerate(names) if "right_hand" in n]
        assert len(self.body_j) == 29 and len(self.lh_j) == 7
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.body_j]); self.vadr = np.array([m.jnt_dofadr[j] for j in self.body_j])
        self.lq = np.array([m.jnt_qposadr[j] for j in self.lh_j]); self.lv = np.array([m.jnt_dofadr[j] for j in self.lh_j])
        self.rq = np.array([m.jnt_qposadr[j] for j in self.rh_j]); self.rv = np.array([m.jnt_dofadr[j] for j in self.rh_j])
        with open(WBC_YAML) as f:
            self.tlim = np.array(yaml.safe_load(f)["motor_effort_limit_list"], dtype=np.float64)
        assert len(self.tlim) == m.nu == 43
        self.pelvis = m.body("pelvis").id; self.torso = m.body("torso_link").id
        # actuator index for each joint (joint idx - 1, base_sim.py:420)
        self.body_act = np.array(self.body_j) - 1; self.lh_act = np.array(self.lh_j) - 1; self.rh_act = np.array(self.rh_j) - 1
        # mesh vertex caches for top-height metrics
        self.geom_verts = {}
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == 0:
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]; a = m.mesh_vertadr[mid]; n = m.mesh_vertnum[mid]
                self.geom_verts[g] = m.mesh_vert[a:a + n].astype(np.float64)
            else:
                c = m.geom_aabb[g][:3]; h = m.geom_aabb[g][3:]
                self.geom_verts[g] = np.array([[c[0] + sx * h[0], c[1] + sy * h[1], c[2] + sz * h[2]]
                                               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        self.torso_geoms = [g for g in self.geom_verts if m.geom_bodyid[g] == self.torso]
        self.floor = m.geom("floor").id
        self.band = True
        self.q_target = DEFAULT_ANGLES.copy(); self.kp = KPS.copy(); self.kd = KDS.copy()
        self.lh_target = np.zeros(7); self.rh_target = np.zeros(7)

    def top_z(self, geoms):
        z = -1e9
        for g in geoms:
            v = self.geom_verts[g] @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]
            z = max(z, float(v[:, 2].max()))
        return z

    def state(self):
        d = self.d
        return (d.qpos[self.qadr].copy(), d.qvel[self.vadr].copy(), d.qpos[3:7].copy(), d.qvel[3:6].copy())

    def step(self):
        d = self.d
        if self.band:  # ElasticBand.Advance on pelvis, point (0,0,1), length 0
            pos = d.xpos[self.pelvis]; quat = d.xquat[self.pelvis]
            vel = np.zeros(6); mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, self.pelvis, vel, 0)
            lin, ang = vel[3:], vel[:3]
            f = 10000 * (np.array([0, 0, 1.0]) - pos) + 1000 * (0 - lin)
            rv = np.zeros(3); mujoco.mju_quat2Vel(rv, quat, 1.0)
            tq = -1000 * rv - 10 * ang
            d.xfrc_applied[self.pelvis] = np.concatenate([f, tq])
        else:
            d.xfrc_applied[self.pelvis] = 0
        tau = np.zeros(self.m.nu)
        tau[self.body_act] = self.kp * (self.q_target - d.qpos[self.qadr]) + self.kd * (0 - d.qvel[self.vadr])
        tau[self.lh_act] = DEX3_KP * (self.lh_target - d.qpos[self.lq]) - DEX3_KD * d.qvel[self.lv]
        tau[self.rh_act] = DEX3_KP * (self.rh_target - d.qpos[self.rq]) - DEX3_KD * d.qvel[self.rv]
        d.ctrl[:] = np.clip(tau, -self.tlim, self.tlim)
        mujoco.mj_step(self.m, d)

    def floor_contacts(self, bodies):
        n = 0
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g = c.geom2 if c.geom1 == self.floor else (c.geom1 if c.geom2 == self.floor else None)
            if g is not None and self.m.geom_bodyid[g] in bodies:
                n += 1
        return n


def ground_start(sim):
    """Robot placed standing in the default pose with the soles on the floor, then 0.1 s of
    default-pose PD hold (= WAIT_FOR_CONTROL after INIT, robot already on the floor)."""
    m, d = sim.m, sim.d
    sim.band = False
    d.qpos[:] = 0; d.qpos[3] = 1.0; d.qpos[sim.qadr] = DEFAULT_ANGLES; d.qpos[2] = START_PELVIS_Z; d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    sim.q_target = DEFAULT_ANGLES.copy()
    sim.lh_target = DEX3_LEFT_DEFAULT.copy(); sim.rh_target = DEX3_RIGHT_DEFAULT.copy()
    for _ in range(HOLD_STEPS):
        sim.step()


def run_segment(name, ctrl, settle_s=2.0, cmd_s=10.0, video_dir=None, frames_dir=None, start="ground"):
    mode, kw = SEGMENTS[name]
    sim = Sim(); m, d = sim.m, sim.d
    mujoco.mj_forward(m, d)
    if start == "band":
        # INIT: 3 s ramp to default pose hanging on the elastic band (cpp:2762-2794)
        q0 = d.qpos[sim.qadr].copy(); t = 0.0
        for k in range(150):
            t += 0.02; r = min(t / 3.0, 1.0)
            sim.q_target = q0 * (1 - r) + DEFAULT_ANGLES * r
            for _ in range(DECIM):
                sim.step()
    else:
        ground_start(sim)
    # WAIT_FOR_CONTROL: holds default pose; then CONTROL + planner enabled
    sim.lh_target = DEX3_LEFT_DEFAULT.copy(); sim.rh_target = DEX3_RIGHT_DEFAULT.copy()
    q, dq, quat, gyro = sim.state()
    ctrl.reset(q, quat)
    n_plan0 = len(ctrl.timing["planner_ms"]) - 1  # count the reset() initialisation call too
    renderer = None
    if video_dir:
        renderer = mujoco.Renderer(m, height=360, width=480)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; cam.trackbodyid = sim.pelvis
        cam.distance = 3.0; cam.azimuth = 135; cam.elevation = -15
        frames = []
    n_settle = int(settle_s / 0.02); n_cmd = int(cmd_s / 0.02); drop_tick = 25  # band released 0.5 s after start
    log = []
    tw = time.time()
    for k in range(n_settle + n_cmd):
        if k == drop_tick and start == "band":
            sim.band = False
        if k == n_settle:
            ctrl.set_command(mode, **kw)
        q, dq, quat, gyro = sim.state()
        sim.q_target = ctrl.control_tick(q, dq, quat, gyro)
        for _ in range(DECIM):
            sim.step()
        ref_pos, _, _ = ctrl.reference_frame()
        log.append(row((k + 1) * 0.02, d.qpos[0:3], d.qpos[3:7], d.qvel[0:3], d.qvel[3:6], d.xmat[sim.torso].reshape(3, 3),
                       sim.top_z(sim.torso_geoms), ref_z=ref_pos[2], robot_top=sim.top_z(list(sim.geom_verts.keys())),
                       torso_floor=sim.floor_contacts({sim.torso, sim.pelvis})))
        if renderer is not None and k % 2 == 0:
            renderer.update_scene(d, camera=cam); frames.append(renderer.render().copy())
        if not np.isfinite(d.qpos).all():
            break
    wall = time.time() - tw
    res = summarize(name, log, n_settle, n_cmd)
    res.update(n_planner_calls=len(ctrl.timing["planner_ms"]) - n_plan0,
               planner_ms_median=float(np.median(ctrl.timing["planner_ms"][n_plan0:])),
               policy_ms_median=float(np.median(ctrl.timing["policy_ms"])), wall_s=wall)
    if renderer is not None:
        import imageio
        os.makedirs(video_dir, exist_ok=True)
        imageio.mimsave(os.path.join(video_dir, f"{name}.mp4"), frames, fps=25, macro_block_size=8)
        if frames_dir:
            os.makedirs(frames_dir, exist_ok=True)
            for frac in (0.15, 0.6, 0.98):
                i = min(int(frac * len(frames)), len(frames) - 1)
                imageio.imwrite(os.path.join(frames_dir, f"{name}_{int(frac*100):02d}.png"), frames[i])
        res["video"] = os.path.join(video_dir, f"{name}.mp4")
    return res, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", nargs="+", default=list(SEGMENTS))
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--video_dir", default=None)
    ap.add_argument("--frames_dir", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--cmd", type=float, default=10.0)
    ap.add_argument("--start", choices=["ground", "band"], default="ground")
    a = ap.parse_args()
    ctrl = SonicController(intra_op_threads=a.threads)
    out = []
    for s in a.segments:
        r, _ = run_segment(s, ctrl, a.settle, a.cmd, a.video_dir, a.frames_dir, a.start)
        r["start"] = a.start
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        out.append(r)
    with open(a.out_json, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
