#!/usr/bin/env python
"""
URMA ("One Policy to Run Them All: an End-to-end Learning Approach to Multi-Embodiment Locomotion", Bohlinger et al.,
CoRL 2024; github.com/nico-bohlinger/one_policy_to_run_them_all) -- the RELEASED multi-embodiment checkpoint
(experiments/pre_trained_model) driving one of ITS OWN training robots in Isaac Sim 5.1, for episode_runner.py and
scripts/terrain/run_terrain.py:

    --robot g1u  --robot-factory urma_robot:make_urma_g1        (Unitree G1, URMA's g1.xml)
    --robot anyu --robot-factory urma_robot:make_urma_anymal    (ANYmal C, URMA's anymal_c.xml; closest to Spot)
    (scripts/urma on PYTHONPATH)

Spot is NOT one of the 16 training robots of the release (A1, Go1, Go2, ANYmal B/C, Barkour v0/vB, Badger, Bittle,
H1, G1, Talos, OP3, NAO, Cassie, hexapod), so the Spot scenes are run with ANYmal C, its closest released embodiment.

Everything is reproduced from the release's MuJoCo env in TEST mode (environments/<robot>/environment.py,
control_functions/rudin2022.py); nothing is tuned:
  physics dt 0.005 s (200 Hz), policy 50 Hz (4 substeps); torques are computed ONCE per policy step and HELD for the
     4 substeps (the release sets data.ctrl once, then mj_step(model, data, 4)), then clipped to the XML ctrlrange
  PD: tau = Kp (q0 + 0.5 a - q) - Kd dq; G1 Kp 45 Kd 1.0, ANYmal C Kp 80 Kd 2.0, action scaling 0.5 (rudin2022.py)
  passive joint terms of the XML: damping (PhysX drive damping, stiffness 0), armature (MJCF importer),
     frictionloss emulated as -fl * clip(dq / 0.05, -1, 1) every physics step (as for the AMO bridge)
  joint velocity cap = env.max_joint_velocities, clipped in the joint state after each control step exactly as the
     release does (NOT PhysX max joint velocity, which in Isaac 5.1 also derates the available effort)
  observation (environment.py get_observation, normalised exactly as released):
     per joint  [static 23-d description | (q - q0)/4.6, dq/35, previous action/10]
     per foot   [static 10-d description | contact/0.5 - 1, clip(t_since_touchdown/2.5 - 1, -1, 1)]
     general    clip(body-frame ang vel/50, -1, 1), goal (vx, vy, wz), projected gravity, static gains/mass/dims
     (the policy does NOT read linear velocity or base height). The static description vectors are exported verbatim
     from the release's own MuJoCo env (export_urma.py -> data/urma/<robot>_static.json).
  foot contact = net contact force on the foot body > 1 N (the release: any MuJoCo contact between foot geom and
     floor; here any surface). The touchdown timer is updated AFTER the observation, as in the release.
  G1 foot spheres (2026-10-11 fix): contactOffset 0.02 m / restOffset 0 and radius 1 cm with the lowest point kept,
     else the 1 mm spheres tunnel through triangle-mesh floors and compliant ground (see _foot_contact_offset);
     URMA_FOOT_COFF=0 URMA_FOOT_R=0 = pre-fix bridge.
  commands: (vx, vy, wz) body frame; training range +-1 m/s, +-1 m/s, +-1 rad/s (command_functions/random.py).
Robot models: the release's MuJoCo XMLs (floor/lights stripped) imported with Isaac 5.1's MJCF importer
(scripts/urma/convert_urma_mjcf.py -> data/urma/usd/<robot>.usd). Action = deterministic policy mean, as in the
release's test(); policy forward pass = urma_np.py (numpy port, max |diff| vs flax 1.4e-6).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, math, os, sys
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
D = os.path.join(ROOT, "data", "urma")
import urma_np

PHYS_HZ, CTRL_HZ = int(os.environ.get("URMA_PHYS_HZ", 200)), 50  # release: dt 0.005 s
SUB = PHYS_HZ // CTRL_HZ
FRICTION_V = 0.05
CONTACT_N = 1.0
ROBOTS = dict(
    # kind: (urma name, foot link name per URMA foot, spawn_dz, fall_tilt, min_z, cam, local-nav footprint, laser heights)
    g1u=dict(
        urma="unitree_g1",
        feet=dict(left_foot="left_ankle_roll_link", right_foot="right_ankle_roll_link"),
        spawn_dz=0.74,
        fall_tilt=0.8,
        min_z=0.3,
        cam=(0.1, 0.55, -0.5),
        foot=(0.18, 0.22),
        laser=(0.15, 0.5, 0.9, 1.2),
        foot_coff=0.02,
        foot_r=0.01,
    ),  # 2026-10-11 fix: contact offset + radius of the 1 mm foot spheres (_foot_contact_offset)
    anyu=dict(
        urma="anymal_c",
        feet=dict(LF_foot="LF_SHANK", RF_foot="RF_SHANK", LH_foot="LH_SHANK", RH_foot="RH_SHANK"),
        spawn_dz=0.60,
        fall_tilt=0.8,
        min_z=0.25,
        cam=(0.5, 0.2, -0.35),
        foot=(0.52, 0.30),
        laser=(0.15, 0.4, 0.7),
    ),
)


def _spec(kind):
    import robots as RB

    c = ROBOTS[kind]
    return RB.RobotSpec(
        kind,
        "legged",
        (1.0, 1.0, 1.0, 0.0, 1.0),
        c["spawn_dz"],
        c["fall_tilt"],
        PHYS_HZ,
        usd=os.path.join(D, "usd", c["urma"] + ".usd"),
        foot_expr=".*(ankle_roll_link|SHANK)",
        v_min=0.1,
        cam=c["cam"],
        min_z=c["min_z"],
        extra=dict(controller="URMA (released, One Policy to Run Them All)", urma_robot=c["urma"]),
    )


class URMAAdapter:
    def __init__(self, kind, prim_path, position, yaw=0.0):
        import omni.usd
        from pxr import Usd, UsdPhysics
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.prims import SingleArticulation, RigidPrim
        import robots as RB

        self.RB = RB
        self.kind, self.prim_path = kind, prim_path
        self.cfg = ROBOTS[kind]
        self.spec = _spec(kind)
        self.S = json.load(open(os.path.join(D, self.cfg["urma"] + "_static.json")))
        self.pol = urma_np.URMAPolicy(os.path.join(D, "urma_params.npz"))
        self.cmd = np.zeros(4)
        self.cmd_req = np.zeros(4)
        self.clipped = False
        self.ready = False
        self._variant = dict(effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0)
        add_reference_to_stage(self.spec.usd, prim_path)
        st = omni.usd.get_context().get_stage()
        root = None
        for p in Usd.PrimRange(st.GetPrimAtPath(prim_path)):
            if p.HasAPI(UsdPhysics.ArticulationRootAPI) and p.GetName() != "worldBody":
                root = str(p.GetPath())
                break
        self.links = {}
        for p in Usd.PrimRange(st.GetPrimAtPath(prim_path)):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                self.links.setdefault(p.GetName(), str(p.GetPath()))
        q = Rotation.from_euler("z", yaw).as_quat()
        self._art = SingleArticulation(
            prim_path=root,
            name=kind,
            position=np.asarray(position, float),
            orientation=np.array([q[3], q[0], q[1], q[2]]),
        )
        self.robot = self._art
        self.ctrl = None
        self.n_mat_fixed = RB.RobotAdapter._fix_contact_materials(prim_path)
        self.foot_paths = [self.links[self.cfg["feet"][f]] for f in self.S["foot_names"]]
        self.foot_coff = self._foot_contact_offset(st)
        self.compliant = (
            self._contact_model(st)
            if self.cfg.get("compliant") and os.environ.get("URMA_COMPLIANT", "1") != "0"
            else None
        )
        parent = os.path.dirname(self.foot_paths[0])  # one regex (a path LIST breaks the contact view)
        expr = parent + "/(" + "|".join(sorted(set(os.path.basename(p) for p in self.foot_paths))) + ")"
        self._feet = RigidPrim(expr, name=kind + "_feet", track_contact_forces=True, reset_xform_properties=False)
        found = list(self._feet.prim_paths)
        self.foot_order = np.array([found.index(p) for p in self.foot_paths])  # view order -> URMA foot order
        self._floor_z = 0.0
        print(f"[urma] {kind}: root {root}, feet {self.foot_paths}", flush=True)

    def _foot_contact_offset(self, st):
        """2026-10-11 bridge fix (G1 only). URMA's G1 touches the ground with four 1 mm SPHERES per foot (g1.xml). PhysX's
        automatic contact offset for a 1 mm sphere is ~1 mm, so on TRIANGLE-MESH ground (every scene floor, the terrain
        field's asphalt) the spheres tunnel through within a few steps: the feet sank 3-12 cm into the mesh, the robot
        stood on its ankle hulls, ran away and fell (the old 'falls at >= 0.5 m/s' / 'fell on the approach' result).
        An analytic ground plane is a half-space, so it never showed this (diag: same bridge tracked 0.3-0.8 m/s there).
        Fix: contactOffset 0.02 m (speculative contact distance only) and restOffset 0 (contact at the true sphere
        surface) on the foot spheres. Geometry, friction and the contact point positions are unchanged.
        Part 2: on COMPLIANT ground (the terrain field's grass / sand / mud / clay / rubber materials) the contact is
        meant to penetrate a few mm, more than the 1 mm radius, so the sphere centre still passed below the triangle and
        the foot fell through (diag: 2-3 mm sink -> through, robot fell at once). The spheres are therefore given a
        1 cm radius with their LOWEST point kept where it was (centre raised by r - 1 mm along the link's z): same four
        contact locations, same sole height, only the sphere curvature differs (contact shift <= r*sin(ankle tilt)).
        URMA_FOOT_COFF=0 / URMA_FOOT_R=0 reproduce the pre-fix bridge."""
        coff = float(os.environ.get("URMA_FOOT_COFF", self.cfg.get("foot_coff", 0.0)))
        rad = float(os.environ.get("URMA_FOOT_R", self.cfg.get("foot_r", 0.0)))
        if coff <= 0 and rad <= 0:
            return 0.0
        from pxr import Usd, UsdGeom, PhysxSchema, Gf

        n = 0
        for fp in self.foot_paths:
            sph = [
                str(p.GetPath())
                for p in Usd.PrimRange(st.GetPrimAtPath(fp), Usd.TraverseInstanceProxies())
                if p.IsA(UsdGeom.Sphere)
            ]
            for sp in sph:  # colliders are instance proxies: un-instance their ancestors (local copy)
                q = st.GetPrimAtPath(sp)
                while q.GetPath().pathString != "/":
                    if q.IsInstanceable():
                        q.SetInstanceable(False)
                    q = q.GetParent()
            for sp in sph:
                pr = st.GetPrimAtPath(sp)
                if coff > 0:
                    pc = PhysxSchema.PhysxCollisionAPI.Apply(pr)
                    pc.CreateContactOffsetAttr().Set(coff)
                    pc.CreateRestOffsetAttr().Set(0.0)
                if rad > 0:
                    g = UsdGeom.Sphere(pr)
                    r0 = float(g.GetRadiusAttr().Get())
                    g.GetRadiusAttr().Set(rad)
                    t = [
                        o
                        for o in UsdGeom.Xformable(pr.GetParent()).GetOrderedXformOps()
                        if o.GetOpType() == UsdGeom.XformOp.TypeTranslate
                    ]
                    cur = t[0].Get()
                    t[0].Set(Gf.Vec3d(cur[0], cur[1], cur[2] + (rad - r0)))
                n += 1
        print(
            f"[urma] {self.kind}: foot spheres contactOffset {coff} restOffset 0, radius {rad} (lowest point kept) on {n} colliders",
            flush=True,
        )
        return coff

    def _contact_model(self, st):
        """The release's MuJoCo contact for G1's foot points: solref = (-1000, -80) = DIRECT acceleration-level spring
        (aref = -1000 r - 80 v), default solimp. PhysX equivalent: compliant contact with an ACCELERATION spring of the
        same stiffness / damping on the foot links (rigid PhysX contact on four 1 mm spheres is a different contact
        model from the one the policy was trained on). Friction: mu 10 with combine 'min' -> the SURFACE's friction
        governs, exactly as for every other robot (RobotAdapter._fix_contact_materials)."""
        from pxr import UsdShade, UsdPhysics, PhysxSchema

        k, c = self.cfg["compliant"]
        mp = f"/World/URMA_{self.kind}_FootContact"
        mat = UsdShade.Material.Define(st, mp)
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr().Set(10.0)
        api.CreateDynamicFrictionAttr().Set(10.0)
        px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        px.CreateFrictionCombineModeAttr().Set("min")
        px.CreateCompliantContactStiffnessAttr().Set(float(k))
        px.CreateCompliantContactDampingAttr().Set(float(c))
        px.CreateCompliantContactAccelerationSpringAttr().Set(True)
        for fp in self.foot_paths:
            UsdShade.MaterialBindingAPI.Apply(st.GetPrimAtPath(fp)).Bind(
                mat, UsdShade.Tokens.weakerThanDescendants, "physics"
            )
        print(
            f"[urma] {self.kind}: foot contact = compliant acceleration spring k={k} c={c} (MuJoCo solref -{k} -{c})",
            flush=True,
        )
        return (k, c)

    # ------------------------------------------------------------------ lifecycle
    def initialize(self):
        self._art.initialize()
        try:
            self._feet.initialize()
        except Exception as e:
            print("[urma] feet view init:", e, flush=True)
        S = self.S
        names = list(self._art.dof_names)
        n = len(names)
        self.idx = np.array([names.index(j) for j in S["joint_names"]])
        av = self._art._articulation_view
        allj = S["all_joint_names"]
        kd = np.zeros(n)
        fl = np.zeros(n)
        arm = np.zeros(n)
        for i, nm in enumerate(names):
            if nm in allj:
                k = allj.index(nm)
                kd[i], fl[i], arm[i] = S["all_dof_damping"][k], S["all_dof_frictionloss"][k], S["all_dof_armature"][k]
        av.set_gains(kps=np.zeros((1, n)), kds=kd.reshape(1, -1))  # XML passive damping (implicit)
        try:
            av.set_armatures(arm.reshape(1, -1))
        except Exception as e:
            print("[urma] set_armatures:", e, flush=True)
        self.fl = fl
        cap = np.full(n, 1e6)
        cap[self.idx] = S["torque_limits"]
        av.set_max_efforts(cap.reshape(1, -1))
        self._effort_cap = cap
        # NOT PhysX max joint velocity: in Isaac 5.1 that also derates the available effort near the cap (the knee of
        # ANYmal saturated at ~25 of 80 N*m). The release clips qvel after every control step -> done in _policy_tick.
        self.vcap = np.asarray(S["max_joint_velocities"], float)
        self._base_masses = np.asarray(av.get_body_masses(), float).reshape(-1)
        try:  # solver accuracy (MuJoCo's Newton solver converges each step)
            pi0 = int(np.asarray(av.get_solver_position_iteration_counts()).reshape(-1)[0])
            vi0 = int(np.asarray(av.get_solver_velocity_iteration_counts()).reshape(-1)[0])
            PI, VI = (
                int(os.environ.get("URMA_POS_IT", pi0)),
                int(os.environ.get("URMA_VEL_IT", vi0)),
            )  # default: unchanged
            av.set_solver_position_iteration_counts(np.array([PI]))
            av.set_solver_velocity_iteration_counts(np.array([VI]))
            print(f"[urma] solver iterations pos {pi0}->{PI} vel {vi0}->{VI}", flush=True)
        except Exception as e:
            print("[urma] solver iterations:", e, flush=True)
        # full nominal pose (incl. passive G1 hand joints) from the release's home keyframe / nominal joints
        self._q0 = np.zeros(n)
        if S["complete_nominal"]:
            comp = np.asarray(S["complete_nominal"], float)[7:]
            for k, nm in enumerate(allj):
                if nm in names:
                    self._q0[names.index(nm)] = comp[k]
        else:
            self._q0[self.idx] = S["nominal_joint_positions"]
        self.q_nom = np.asarray(S["nominal_joint_positions"], float)
        dpos = np.zeros(n)
        dpos[self.idx] = self.q_nom
        self.ctrl = SimpleNamespace(default_pos=dpos)
        self.jdesc = np.asarray(S["joint_desc"], float)
        self.fdesc = np.asarray(S["foot_desc"], float)
        self.gstat = np.asarray(S["general_static"], float)
        self.kp, self.kd, self.scale = float(S["p_gain"]), float(S["d_gain"]), float(S["scaling_factor"])
        self.tlim = np.asarray(S["torque_limits"], float)
        print(
            f"[urma] {self.kind}: dofs {n} (actuated {len(self.idx)}), mass {self._base_masses.sum():.2f} kg "
            f"(release {S['mass']:.2f}), Kp {self.kp} Kd {self.kd} scale {self.scale}",
            flush=True,
        )
        self._reset_policy()
        self.ready = True

    def _reset_policy(self):
        self.sub = 0
        self.action = np.zeros(len(self.S["joint_names"]))
        self.t_td = np.zeros(len(self.S["foot_names"]))
        self.tau_hold = np.zeros(len(self.S["joint_names"]))
        self.n_ticks = 0

    def _contacts(self):
        try:
            f = np.asarray(self._feet.get_net_contact_forces(dt=1.0 / PHYS_HZ), float).reshape(-1, 3)
            return (np.linalg.norm(f, axis=1) > CONTACT_N).astype(float)[self.foot_order]
        except Exception as e:
            if not getattr(self, "_cwarn", False):
                print("[urma] CONTACT VIEW FAILED:", e, flush=True)
                self._cwarn = True
            return np.zeros(len(self.S["foot_names"]))

    def _policy_tick(self):
        _, quat = self._art.get_world_pose()
        quat = np.asarray(quat, float)
        R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        w_b = R.T @ np.asarray(self._art.get_angular_velocity(), float)
        g_b = R.T @ np.array([0.0, 0.0, -1.0])
        qa = np.asarray(self._art.get_joint_positions(), float).reshape(-1)[self.idx]
        dq_all = np.asarray(self._art.get_joint_velocities(), float).reshape(-1)
        dqa = dq_all[self.idx]
        if np.any(np.abs(dqa) > self.vcap):  # release: qvel[joints] = clip(qvel, -max_v, max_v) after mj_step
            dqa = np.clip(dqa, -self.vcap, self.vcap)
            dq_all[self.idx] = dqa
            self._art.set_joint_velocities(dq_all)
        c = self._contacts()
        js = np.stack([(qa - self.q_nom) / 4.6, dqa / 35.0, self.action / 10.0], 1)
        fs = np.stack([c / 0.5 - 1.0, np.clip(self.t_td / 2.5 - 1.0, -1.0, 1.0)], 1)
        gen = np.concatenate([np.clip(w_b / 50.0, -1.0, 1.0), [self.cmd[0], self.cmd[1], self.cmd[3]], g_b, self.gstat])
        a = self.pol(self.jdesc, js, self.fdesc, fs, gen)
        self.t_td = np.where(c > 0.5, 0.0, self.t_td + 1.0 / CTRL_HZ)  # reward_function.step after the obs
        tau = self.kp * (self.q_nom + self.scale * a - qa) - self.kd * dqa
        self.tau_hold = np.clip(tau, -self.tlim, self.tlim) * self._variant["effort_scale"]
        self.action = a
        self.n_ticks += 1

    def _apply(self):
        from isaacsim.core.utils.types import ArticulationAction

        dq = np.asarray(self._art.get_joint_velocities(), float).reshape(-1)
        full = -self.fl * np.clip(dq / FRICTION_V, -1.0, 1.0)
        full[self.idx] += self.tau_hold
        self._art.apply_action(ArticulationAction(joint_efforts=full))

    def step(self, dt):
        if not self.ready:
            self.initialize()
            return
        if self.sub % SUB == 0:
            self._policy_tick()
        self._apply()
        self.sub += 1

    # ------------------------------------------------------------------ command / state
    def limits(self):
        f, r, y, z, w = (np.array(self.spec.max_cmd) * self._variant["cmd_scale"]).tolist()
        return dict(vx_fwd=f, vx_rev=r, vy=y, vz=z, wz=w)

    def command(self, twist):
        if isinstance(twist, dict):  # posture setpoints are NOT part of URMA's interface
            twist = [twist.get("vx", self.cmd[0]), twist.get("vy", self.cmd[1]), 0.0, twist.get("wz", self.cmd[3])]
        return self.RB.RobotAdapter.command(self, twist)

    def setpoint(self):
        return dict(vx=float(self.cmd[0]), vy=float(self.cmd[1]), wz=float(self.cmd[3]))

    def reset_setpoints(self):  # velocity-only interface: nothing else to reset
        self.command(np.zeros(4))

    def stop(self):
        self.command(np.zeros(4))

    def set_variant(self, effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0):
        self._variant = dict(effort_scale=float(effort_scale), payload_kg=float(payload_kg), cmd_scale=float(cmd_scale))

    def teleport(self, x, y, yaw, floor_z):
        self._floor_z = float(floor_z)
        q = Rotation.from_euler("z", yaw).as_quat()
        self._art.set_world_pose(np.array([x, y, floor_z + self.spec.spawn_dz]), np.array([q[3], q[0], q[1], q[2]]))
        self._art.set_linear_velocity(np.zeros(3))
        self._art.set_angular_velocity(np.zeros(3))
        if getattr(self, "_q0", None) is not None:
            self._art.set_joint_positions(self._q0)
            self._art.set_joint_velocities(np.zeros(len(self._q0)))
            self._reset_policy()
        self.stop()

    def base_state(self):
        return self.RB.RobotAdapter.base_state(self)

    def fallen(self, bs=None):
        bs = bs or self.base_state()
        return bs is None or bs["tilt"] > self.spec.fall_tilt or bs["pos"][2] - self._floor_z < self.spec.min_z

    def _joint_limits(self):
        return self.RB.RobotAdapter._joint_limits(self)

    def proprio(self, bs=None):
        return self.RB.RobotAdapter.proprio(self, bs)

    def descriptor(self, stage=None):
        d = self.RB.RobotAdapter.descriptor(self, stage)
        d.update(
            controller="URMA (released)",
            setpoints_supported=["vx", "vy", "wz"],
            setpoints_unsupported=["body_height", "pitch", "roll", "gait"],
        )
        return d


def _register_local_nav(kind):
    try:
        import local_nav

        local_nav.FOOT.setdefault(kind, ROBOTS[kind]["foot"])
        local_nav.LASER_H.setdefault(kind, ROBOTS[kind]["laser"])
    except Exception:
        pass


def make_urma_g1(kind, prim_path, position, yaw=0.0):
    _register_local_nav(kind)
    return URMAAdapter(kind, prim_path, position, yaw)


def make_urma_anymal(kind, prim_path, position, yaw=0.0):
    _register_local_nav(kind)
    return URMAAdapter(kind, prim_path, position, yaw)


try:
    make_urma_g1.SPEC = _spec("g1u")
    make_urma_anymal.SPEC = _spec("anyu")
except Exception:
    make_urma_g1.SPEC = make_urma_anymal.SPEC = None
