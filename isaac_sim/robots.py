#!/usr/bin/env python
"""
ROBOT ADAPTERS -- one interface over every body the feasibility model is asked about.

The model never sees a robot's native command or observation layout. Everything goes through
RobotAdapter:

    command(twist)   body-frame [vx, vy, vz, wz] in PHYSICAL units (m/s, rad/s). Clipped to the
                     robot's own max_cmd; the clip is reported, never silent -- a selector that
                     asks for 1.5 m/s on a Jetbot must learn that it got 0.2.
    proprio()        per-joint pos/vel/effort + base IMU + cmd-vs-achieved, AND the same joint
                     signals normalised by that joint's own limits (effort/tau_max, vel/v_max,
                     position within range). The normalised "actuator utilisation" channels are
                     what make a 12-joint Spot, a 19-joint H1 and a 2-wheel Carter comparable.
    descriptor()     static embodiment description read from the live articulation (joint
                     limits, effort caps, mass, footprint, height, max_cmd). Nothing hand-typed
                     per robot except the controller choice and command limits.

Legged bodies run NVIDIA's shipped flat-terrain RL policies (all take (vx, vy, wz)); wheeled
bodies run a differential-drive controller; the aerial body (Pegasus Iris, DroneAdapter) runs
Pegasus' geometric controller driven as a velocity tracker. Use make_robot(kind, ...) -- it
returns the right adapter; all expose the same interface.

Import AFTER SimulationApp is created (needs omni / isaacsim modules).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

ASSET = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac"


@dataclass
class RobotSpec:
    kind: str                    # key used everywhere: "spot", "h1", ...
    family: str                  # "legged" | "wheeled" | "aerial"
    max_cmd: tuple               # (vx_fwd, vx_rev, vy, vz, wz) magnitudes the controller accepts
    spawn_dz: float              # base height above floor at spawn (just above resting height)
    fall_tilt: float             # body-z vs world-up angle (rad) beyond which the body is down
    phys_hz: int                 # physics rate the controller was TRAINED at (env yaml sim.dt)
    usd: str = ""
    foot_expr: str = ""          # regex of contact links, relative to the robot prim
    v_min: float = 0.15          # lowest forward command the controller reliably EXECUTES
    cam: tuple = (0.4, 0.2, -0.35)   # head camera (forward, up from base, pitch rad)
    min_z: float = 0.0           # base height above floor below which the body is DOWN (0 = off)
    extra: dict = field(default_factory=dict)


# max_cmd values are the command ranges each policy was TRAINED on (env yaml / Isaac examples),
# not wishes: commanding a flat-terrain policy outside them is itself a failure mode.
# phys_hz MATTERS: the policies count physics steps (decimation) to run at 50 Hz, and ANYmal's
# actuator net is stepped every physics tick. Spot was trained at 500 Hz, H1/ANYmal at 200 Hz;
# running H1 at 500 Hz makes it fall within 2 s. Each probe/collector world uses the robot's
# own rate -- one robot per World.
SPECS = {
    # v_min from deadband_probe.py: Spot's flat policy stands still below ~0.4 m/s (0.3: 3/4
    # trials move, 0.2: 1/4); ANYmal / H1 / Carter track down to 0.1 m/s.
    "spot":   RobotSpec("spot", "legged", (1.5, 1.0, 0.8, 0.0, 1.5), 0.8, 0.8, 500,
                        foot_expr=".*foot", v_min=0.4, cam=(0.45, 0.15, -0.35), min_z=0.25),
    "anymal": RobotSpec("anymal", "legged", (1.0, 1.0, 0.8, 0.0, 1.0), 0.7, 0.8, 200,
                        foot_expr=".*FOOT", v_min=0.15, cam=(0.5, 0.2, -0.35), min_z=0.25),
    "h1":     RobotSpec("h1", "legged", (1.0, 0.3, 0.3, 0.0, 1.0), 1.05, 0.6, 200,
                        foot_expr=".*ankle_link", v_min=0.15, cam=(0.15, 0.65, -0.5), min_z=0.55),
    "carter": RobotSpec("carter", "wheeled", (1.2, 0.6, 0.0, 0.0, 1.2), 0.27, 0.6, 200,
                        v_min=0.15, cam=(0.3, 0.55, -0.35),
                        usd=ASSET + "/Robots/NVIDIA/Carter/carter_v1.usd",
                        extra=dict(wheel_radius=0.24, wheel_base=0.54, wheel_tau_max=30.0)),
    # Clearpath Jackal: 17 kg 4-wheel SKID-STEER (all left wheels share one speed, all right
    # wheels the other). Ships a wheel material of mu=0.2; with the 'min' combine rule that
    # would cap its grip at 0.2 on every floor, so wheel_mu sets realistic rubber (0.9) and the
    # floor governs, as for Carter. wheel_tau_max 8 N*m/wheel is ASSUMED (USD ships inf):
    # 4 x 8 / 0.098 m ~ 330 N on 17 kg. Track 0.376 m, wheel radius 0.098 m (Clearpath URDF).
    "jackal": RobotSpec("jackal", "wheeled", (2.0, 1.0, 0.0, 0.0, 2.0), 0.12, 0.6, 200,
                        v_min=0.1, cam=(0.2, 0.25, -0.3),
                        usd=ASSET + "/Robots/Clearpath/Jackal/jackal.usd",
                        extra=dict(wheel_radius=0.098, wheel_base=0.376, wheel_tau_max=8.0,
                                   wheel_mu=0.9, skid=True)),
    # Iris flies at a fixed altitude above the floor (extra.alt); spawn_dz is that altitude.
    # Rotor "joints": 4 actuators, vel = commanded rotor speed, limit = max rotor speed.
    "iris":   RobotSpec("iris", "aerial", (2.0, 2.0, 1.5, 1.0, 1.5), 1.0, 1.0, 200,
                        v_min=0.05, cam=(0.15, -0.05, -0.35),
                        extra=dict(alt=1.0, max_rotor=1100.0, mass=1.5)),
    "jetbot": RobotSpec("jetbot", "wheeled", (0.25, 0.25, 0.0, 0.0, 2.0), 0.08, 0.6, 200,
                        v_min=0.05, cam=(0.08, 0.1, -0.3), usd=ASSET + "/Robots/NVIDIA/Jetbot/jetbot.usd",
                        extra=dict(wheel_radius=0.0325, wheel_base=0.1125, wheel_tau_max=0.15)),
}
UNBOUNDED = 1e6          # PhysX reports "no limit" as FLT_MAX; anything above this is unknown
# wheel_tau_max is an ASSUMED nominal motor limit (the USDs ship FLT_MAX): 30 N*m/wheel lets
# the 76 kg Carter climb ~19 deg, 0.15 N*m gives the 2.5 kg Jetbot ~0.37 g. Without a finite
# cap, set_variant(effort_scale) has nothing to scale on wheeled bodies.

N_BASE = 13      # base token: lin_vel_b(3) ang_vel_b(3) gravity_b(3) cmd(4)
N_JOINT = 7      # joint token: pos_rel, vel, eff, pos_in_range, vel/vmax, eff/taumax, is_wheel


def _yaw_quat(yaw):
    q = Rotation.from_euler("z", yaw).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


class RobotAdapter:
    """Spawn + drive + sense one robot. Call initialize() on the FIRST physics step (the
    policy controllers need a live physics view), then step(dt) every physics step."""

    def __init__(self, kind, prim_path, position, yaw=0.0):
        if kind not in SPECS:
            raise KeyError(f"unknown robot {kind!r}; have {sorted(SPECS)}")
        self.spec = SPECS[kind]
        self.kind = kind
        self.prim_path = prim_path
        self.cmd = np.zeros(4)                       # [vx, vy, vz, wz] after clipping
        self.cmd_req = np.zeros(4)                   # what the caller asked for
        self.clipped = False
        self.ready = False
        self._effort_cap = None                      # nominal max efforts, for variants
        self._variant = dict(effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0)
        pos = np.asarray(position, float)
        orient = _yaw_quat(yaw)
        if kind == "spot":
            from isaacsim.robot.policy.examples.robots import SpotFlatTerrainPolicy
            self.ctrl = SpotFlatTerrainPolicy(prim_path=prim_path, name=kind, position=pos,
                                              orientation=orient)
        elif kind == "anymal":
            from isaacsim.robot.policy.examples.robots import AnymalFlatTerrainPolicy
            self.ctrl = AnymalFlatTerrainPolicy(prim_path=prim_path, name=kind, position=pos,
                                                orientation=orient)
        elif kind == "h1":
            from isaacsim.robot.policy.examples.robots import H1FlatTerrainPolicy
            self.ctrl = H1FlatTerrainPolicy(prim_path=prim_path, name=kind, position=pos,
                                            orientation=orient)
        else:  # wheeled
            from isaacsim.core.utils.stage import add_reference_to_stage
            from isaacsim.core.prims import SingleArticulation
            add_reference_to_stage(self.spec.usd, prim_path)
            self.ctrl = None
            self._art = SingleArticulation(prim_path=prim_path, name=kind, position=pos,
                                           orientation=orient)
        self.robot = self.ctrl.robot if self.ctrl is not None else self._art
        self.n_mat_fixed = self._fix_contact_materials(prim_path, self.spec.extra.get("wheel_mu"))

    @staticmethod
    def _fix_contact_materials(prim_path, wheel_mu=None):
        """Force every physics material shipped INSIDE the robot to friction-combine 'min'.
        Carter's WheelMaterial ships mu=1.0 with combine 'max'; PhysX resolves combine-mode
        conflicts by priority (max > multiply > min > average), so the wheel's 1.0 overrode
        every floor -- Carter braked at ~5 m/s^2 on mu=0.08 oil, i.e. never saw oil at all.
        With 'min' on both sides the contact uses the lower of the two, so the floor governs.
        wheel_mu (if given) also resets the shipped friction -- Jackal ships 0.2, which under
        'min' would make every floor look like oil to it.
        Must run before world.reset() (physics parses materials at start)."""
        import omni.usd
        from pxr import Usd, UsdPhysics, PhysxSchema
        stage = omni.usd.get_context().get_stage()
        n = 0
        for p in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
            if p.HasAPI(UsdPhysics.MaterialAPI):
                PhysxSchema.PhysxMaterialAPI.Apply(p).CreateFrictionCombineModeAttr().Set("min")
                if wheel_mu is not None:
                    m = UsdPhysics.MaterialAPI(p)
                    m.CreateStaticFrictionAttr().Set(float(wheel_mu))
                    m.CreateDynamicFrictionAttr().Set(float(wheel_mu) * 0.8)
                n += 1
        # OPT-IN (XFEAS_FOOT_MATERIAL=1): rubber on the contact links + the 'multiply' convention for the scene
        # (foot_material.py). Off by default, so every existing result is reproduced unchanged.
        import foot_material
        if foot_material.enabled():
            foot_material.apply(prim_path)
        return n

    # ------------------------------------------------------------------ lifecycle
    def initialize(self):
        if self.ctrl is not None:
            self.ctrl.initialize()
        else:
            from isaacsim.robot.wheeled_robots.controllers.differential_controller import \
                DifferentialController
            self._art.initialize()
            names = list(self._art.dof_names)
            wl = [i for i, n in enumerate(names) if "wheel" in n.lower() and "left" in n.lower()]
            wr = [i for i, n in enumerate(names) if "wheel" in n.lower() and "right" in n.lower()]
            if not wl or not wr:
                raise RuntimeError(f"{self.kind}: no left/right wheel joints in {names}")
            # every left wheel gets the left speed, every right wheel the right speed: Carter
            # has one of each (+ casters), Jackal is 4-wheel skid-steer
            self._wl, self._wr = wl, wr
            self._wheels = wl + wr
            self._diff = DifferentialController(name=f"{self.kind}_diff",
                                                wheel_radius=self.spec.extra["wheel_radius"],
                                                wheel_base=self.spec.extra["wheel_base"])
            eff = np.asarray(self._art._articulation_view.get_max_efforts(), float).reshape(-1)
            eff[self._wheels] = self.spec.extra["wheel_tau_max"]
            self._art._articulation_view.set_max_efforts(eff.reshape(1, -1))
        av = self.robot._articulation_view
        # joint pose to restore on teleport. Legged: the policy's default stance. Wheeled: the
        # spawn pose -- Carter's rear caster otherwise keeps its angle from the previous turn
        # and the next 'straight' run veers off at ~0.65 rad.
        self._q0 = (np.asarray(self.ctrl.default_pos, float) if self.ctrl is not None
                    and hasattr(self.ctrl, "default_pos")
                    else np.asarray(self.robot.get_joint_positions(), float).reshape(-1).copy())
        try:
            self._effort_cap = np.asarray(av.get_max_efforts(), float).reshape(-1)
        except Exception:
            self._effort_cap = None
        if self.kind == "anymal":
            # ANYmal's actuator net writes joint torques directly (effort mode), which bypasses
            # set_max_efforts -- clip its output so effort_scale actually weakens the body.
            net = self.ctrl._actuator_network
            raw = net.compute_torques

            def capped(q, dq, a, _raw=raw):
                tau, extra = _raw(q, dq, a)
                if self._effort_cap is not None:
                    cap = self._effort_cap * self._variant["effort_scale"]
                    tau = np.clip(tau, -cap, cap)
                return tau, extra
            net.compute_torques = capped
        self._base_masses = None
        try:
            self._base_masses = np.asarray(av.get_body_masses(), float).reshape(-1)
        except Exception:
            pass
        self.ready = True

    def step(self, dt):
        if not self.ready:
            self.initialize()
            return
        if self.ctrl is not None:
            self.ctrl.forward(dt, np.array([self.cmd[0], self.cmd[1], self.cmd[3]]))
        else:
            from isaacsim.core.utils.types import ArticulationAction
            wz = self.cmd[3]
            if self.spec.extra.get("skid"):
                # skid-steer wheels slip sideways, so open-loop diff-drive under-turns badly
                # (Jackal reached <50% of every commanded turn). Close the yaw-rate loop with an
                # integral term, as the real Jackal base controller does.
                try:
                    q = np.asarray(self._art.get_world_pose()[1], float)
                    R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
                    w_meas = float((R.T @ np.asarray(self._art.get_angular_velocity(), float))[2])
                except Exception:
                    w_meas = wz
                self._wz_i = float(np.clip(getattr(self, "_wz_i", 0.0) + 4.0 * (wz - w_meas) * dt, -3.0, 3.0))
                if abs(wz) < 1e-3 and abs(w_meas) < 0.05:
                    self._wz_i *= 0.98
                wz = wz + self._wz_i
            wv = self._diff.forward(np.array([self.cmd[0], wz])).joint_velocities
            full = np.zeros(len(self._art.dof_names))
            full[self._wl], full[self._wr] = wv[0], wv[1]
            self._art.apply_action(ArticulationAction(joint_velocities=full))

    # ------------------------------------------------------------------ command
    def limits(self):
        """Effective command limits after the current variant's cmd_scale."""
        f, r, y, z, w = (np.array(self.spec.max_cmd) * self._variant["cmd_scale"]).tolist()
        return dict(vx_fwd=f, vx_rev=r, vy=y, vz=z, wz=w)

    def command(self, twist):
        """twist = [vx, vy, vz, wz] body frame. Returns the command actually applied."""
        t = np.zeros(4)
        t[:len(twist)] = np.asarray(twist, float)[:4]
        L = self.limits()
        c = np.array([np.clip(t[0], -L["vx_rev"], L["vx_fwd"]),
                      np.clip(t[1], -L["vy"], L["vy"]),
                      np.clip(t[2], -L["vz"], L["vz"]),
                      np.clip(t[3], -L["wz"], L["wz"])])
        self.cmd_req, self.cmd = t, c
        self.clipped = bool(np.any(np.abs(c - t) > 1e-6))
        return c

    def stop(self):
        self.command(np.zeros(4))
        self._wz_i = 0.0

    # ------------------------------------------------------------------ embodiment variants
    def set_variant(self, effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0):
        """'Virtual embodiment': weaker actuators, extra mass on the base, narrower command
        range. The shipped policy is unchanged -- which is the point: the SAME controller on a
        degraded body is a different feasibility problem, and the model has to tell from
        proprioception alone (the descriptor reports nominal, not degraded, limits unless the
        caller says otherwise)."""
        av = self.robot._articulation_view
        if self._effort_cap is not None:
            av.set_max_efforts(self._effort_cap * float(effort_scale))
        if self._base_masses is not None:
            m = self._base_masses.copy()
            m[0] += float(payload_kg)                # body 0 = articulation root (the base)
            av.set_body_masses(m.reshape(1, -1))
        self._variant = dict(effort_scale=float(effort_scale), payload_kg=float(payload_kg),
                             cmd_scale=float(cmd_scale))

    # ------------------------------------------------------------------ state
    def base_state(self):
        pos, q = self.robot.get_world_pose()
        q = np.asarray(q, float)
        if not np.all(np.isfinite(q)) or np.linalg.norm(q) < 1e-6:
            return None
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]])
        r, p, y = R.as_euler("xyz")
        RT = R.as_matrix().T
        v_b = RT @ np.asarray(self.robot.get_linear_velocity(), float)
        w_b = RT @ np.asarray(self.robot.get_angular_velocity(), float)
        g_b = RT @ np.array([0.0, 0.0, -1.0])
        # tilt = angle between body z and world up, from gravity in the body frame. The old
        # |roll|+|pitch| double-counts diagonal leans and is frame-convention dependent.
        return dict(pos=np.asarray(pos, float), quat=q, roll=float(r), pitch=float(p),
                    yaw=float(y), tilt=float(np.arccos(np.clip(-g_b[2], -1.0, 1.0))),
                    v_b=v_b, w_b=w_b, g_b=g_b)

    def fallen(self, bs=None):
        """Down = tipped past fall_tilt OR the base sank below min_z. The height check matters
        for H1: it can collapse to the floor with its pelvis still near-upright, which the tilt
        test alone labelled 'under_turn' / 'stopped' instead of a fall."""
        bs = bs or self.base_state()
        return (bs is None or bs["tilt"] > self.spec.fall_tilt
                or (self.spec.min_z > 0 and bs["pos"][2] - self._floor_z < self.spec.min_z))

    _floor_z = 0.0

    def teleport(self, x, y, yaw, floor_z):
        self._floor_z = float(floor_z)
        self.robot.set_world_pose(np.array([x, y, floor_z + self.spec.spawn_dz]), _yaw_quat(yaw))
        try:
            self.robot.set_linear_velocity(np.zeros(3))
            self.robot.set_angular_velocity(np.zeros(3))
            n = len(self.robot.dof_names)
            if getattr(self, "_q0", None) is not None:
                self.robot.set_joint_positions(self._q0)
            self.robot.set_joint_velocities(np.zeros(n))
        except Exception:
            pass
        self.stop()

    # ------------------------------------------------------------------ proprioception
    def _joint_limits(self):
        av = self.robot._articulation_view
        lim = np.asarray(av.get_dof_limits(), float).reshape(-1, 2)
        try:
            vmax = np.asarray(av.get_max_joint_velocities(), float).reshape(-1)
        except Exception:
            vmax = np.full(len(lim), np.inf)
        # NOMINAL limits on purpose: a degraded body (set_variant) must be inferred from
        # proprioception (saturation, tracking error), not read off its own descriptor.
        tmax = (self._effort_cap if self._effort_cap is not None
                else np.full(len(lim), np.inf))
        vmax = np.where(np.abs(vmax) < UNBOUNDED, vmax, np.inf)
        tmax = np.where(np.abs(tmax) < UNBOUNDED, tmax, np.inf)
        return lim, vmax, tmax

    def proprio(self, bs=None):
        """One proprioceptive sample. Returns (base[N_BASE], joints[J, N_JOINT], extras).

        Joint token = [pos - default, vel, effort, pos_in_range(-1..1), vel/vmax,
                       effort/tau_max, is_wheel]. Unbounded joints (wheels) get
        pos_in_range=0 and rely on the vel/effort ratios."""
        bs = bs or self.base_state()
        if bs is None:
            return None
        av = self.robot._articulation_view
        q = np.asarray(self.robot.get_joint_positions(), float).reshape(-1)
        dq = np.asarray(self.robot.get_joint_velocities(), float).reshape(-1)
        try:
            tau = np.asarray(av.get_measured_joint_efforts(), float).reshape(-1)
        except Exception:
            try:
                tau = np.asarray(av.get_applied_joint_efforts(), float).reshape(-1)
            except Exception:
                tau = np.zeros_like(q)
        lim, vmax, tmax = self._joint_limits()
        q0 = (np.asarray(self.ctrl.default_pos, float) if self.ctrl is not None
              and hasattr(self.ctrl, "default_pos") else np.zeros_like(q))
        span = lim[:, 1] - lim[:, 0]
        bounded = np.isfinite(span) & (span > 1e-6) & (span < 2 * math.pi + 1e-3)
        in_rng = np.where(bounded, 2 * (q - lim[:, 0]) / np.where(bounded, span, 1) - 1, 0.0)
        wheel = np.array([("wheel" in n.lower()) for n in self.robot.dof_names], float)
        with np.errstate(divide="ignore", invalid="ignore"):
            vr = np.where(np.isfinite(vmax) & (vmax > 0), dq / vmax, 0.0)
            tr = np.where(np.isfinite(tmax) & (tmax > 0), tau / tmax, 0.0)
        joints = np.stack([q - q0, dq, tau, in_rng, vr, tr, wheel], 1).astype(np.float32)
        base = np.concatenate([bs["v_b"], bs["w_b"], bs["g_b"], self.cmd]).astype(np.float32)
        v_plan = float(np.hypot(bs["v_b"][0], bs["v_b"][1]))
        c_plan = float(np.hypot(self.cmd[0], self.cmd[1]))
        extras = dict(tilt=bs["tilt"], track_err=float(np.linalg.norm(bs["v_b"][:2] - self.cmd[:2])),
                      v_ratio=v_plan / c_plan if c_plan > 0.05 else 1.0,
                      yaw_err=float(bs["w_b"][2] - self.cmd[3]),
                      clipped=self.clipped)
        return base, joints, extras

    # ------------------------------------------------------------------ descriptor
    def descriptor(self, stage=None):
        """Static embodiment description, read from the live articulation + USD bounds."""
        lim, vmax, tmax = self._joint_limits()
        names = list(self.robot.dof_names)
        mass = float(np.sum(self._base_masses)) if self._base_masses is not None else float("nan")
        dims = [float("nan")] * 3
        if stage is not None:
            try:
                from pxr import UsdGeom
                cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
                rng = cache.ComputeWorldBound(stage.GetPrimAtPath(self.prim_path)).ComputeAlignedRange()
                d = rng.GetSize()
                dims = [float(d[0]), float(d[1]), float(d[2])]     # measured at spawn yaw
            except Exception:
                pass
        L = self.limits()
        return dict(kind=self.kind, family=self.spec.family, n_joints=len(names),
                    joint_names=names,
                    joint_lower=lim[:, 0].tolist(), joint_upper=lim[:, 1].tolist(),
                    # -1 = no limit reported (wheels, passive casters)
                    joint_vmax=[float(x) if np.isfinite(x) else -1.0 for x in vmax],
                    joint_tau_max=[float(x) if np.isfinite(x) else -1.0 for x in tmax],
                    mass_kg=mass, bbox_xyz_m=dims, max_cmd=L, variant=dict(self._variant),
                    fall_tilt=self.spec.fall_tilt, phys_hz=self.spec.phys_hz,
                    v_min=self.spec.v_min)


def _velocity_backend_cls():
    """Pegasus backend that makes the Iris a VELOCITY-commanded body. The shipped geometric
    controller tracks a position reference; we move that reference (a leashed carrot) at the
    commanded body-frame velocity and feed the velocity forward. Degradation: rotor speeds are
    capped at effort_scale * max_rotor, and payload is an extra downward force the controller
    does not know about (it keeps assuming 1.5 kg)."""
    import os, sys
    for p in (xfeas_paths.PEGASUS + "/examples/utils",
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "legacy", "ros2_mission")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from drone_waypoint_controller import WaypointController

    class VelocityBackend(WaypointController):
        LEASH = 0.6

        def __init__(self, max_rotor):
            super().__init__([[0.0, 0.0, 1.0, 0.0]])
            self.cmd = np.zeros(4)
            self.p_ref, self.yaw_ref = None, 0.0
            self.max_rotor, self.effort, self.payload = max_rotor, 1.0, 0.0
            self.rotor = np.zeros(4)

        def reset_ref(self, p, yaw):
            self.p_ref, self.yaw_ref = np.asarray(p, float).copy(), float(yaw)
            self.int = np.zeros(3)

        def update(self, dt):
            if not self.reveived_first_state:
                return
            if self.p_ref is None:
                self.reset_ref(self.p, float(self.R.as_euler("xyz")[2]))
            vx, vy, vz, wz = self.cmd
            self.yaw_ref += wz * dt
            c, s = math.cos(self.yaw_ref), math.sin(self.yaw_ref)
            v_w = np.array([c * vx - s * vy, s * vx + c * vy, vz])
            self.p_ref = self.p_ref + v_w * dt
            d = self.p_ref - self.p
            if np.linalg.norm(d) > self.LEASH:              # don't wind up against a wall
                self.p_ref = self.p + d / np.linalg.norm(d) * self.LEASH
            ep = self.p - self.p_ref; ev = self.v - v_w
            self.int = self.int + ep * dt
            F_des = -(self.Kp @ ep) - (self.Kd @ ev) - (self.Ki @ self.int) \
                + np.array([0.0, 0.0, self.m * self.g])
            Z_B = self.R.as_matrix()[:, 2]
            u_1 = F_des @ Z_B
            Z_b_des = F_des / np.linalg.norm(F_des)
            X_c_des = np.array([c, s, 0.0])
            Y_b_des = np.cross(Z_b_des, X_c_des); Y_b_des /= np.linalg.norm(Y_b_des)
            X_b_des = np.cross(Y_b_des, Z_b_des)
            R_des = np.c_[X_b_des, Y_b_des, Z_b_des]; R = self.R.as_matrix()
            e_R = 0.5 * self.vee((R_des.T @ R) - (R.T @ R_des))
            e_w = self.w - np.array([0.0, 0.0, wz * Z_b_des[2]])
            tau = -(self.Kr @ e_R) - (self.Kw @ e_w)
            if self.vehicle:
                ref = np.asarray(self.vehicle.force_and_torques_to_velocities(u_1, tau), float)
                self.rotor = np.clip(ref, 0.0, self.effort * self.max_rotor)
                self.input_ref = self.rotor.tolist()
                if self.payload > 0:
                    g_b = R.T @ np.array([0.0, 0.0, -self.payload * self.g])
                    self.vehicle.apply_force(g_b.tolist())

    return VelocityBackend


class DroneAdapter:
    """Pegasus Iris behind the RobotAdapter interface. Flies at spec.extra['alt'] above the
    floor; teleport() puts it there hovering. step() is a no-op: Pegasus drives the vehicle
    from its own physics callbacks."""

    def __init__(self, kind, prim_path, position, yaw=0.0):
        from omni.isaac.core.world import World
        from pegasus.simulator.params import ROBOTS
        from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
        from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
        self.spec, self.kind, self.prim_path = SPECS[kind], kind, prim_path
        self.cmd, self.cmd_req, self.clipped, self.ready = np.zeros(4), np.zeros(4), False, False
        self._variant = dict(effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0)
        PegasusInterface()._world = World.instance()        # Pegasus spawns into OUR world
        self.backend = _velocity_backend_cls()(self.spec.extra["max_rotor"])
        cfg = MultirotorConfig(); cfg.backends = [self.backend]; cfg.sensors = []
        pos = np.asarray(position, float).copy(); pos[2] = max(pos[2], self.spec.extra["alt"])
        q = Rotation.from_euler("z", yaw).as_quat()
        self.vehicle = Multirotor(prim_path, ROBOTS["Iris"], 0, pos.tolist(), q.tolist(), config=cfg)
        self.robot = self.vehicle
        self.n_mat_fixed = 0

    def initialize(self):
        self.ready = True

    def step(self, dt):
        self.ready = True

    limits = RobotAdapter.limits

    def command(self, twist):
        c = RobotAdapter.command(self, twist)
        self.backend.cmd = c.copy()
        return c

    def stop(self):
        self.command(np.zeros(4))

    def set_variant(self, effort_scale=1.0, payload_kg=0.0, cmd_scale=1.0):
        self.backend.effort, self.backend.payload = float(effort_scale), float(payload_kg)
        self._variant = dict(effort_scale=float(effort_scale), payload_kg=float(payload_kg),
                             cmd_scale=float(cmd_scale))

    def base_state(self):
        st = self.vehicle.state
        qa = np.asarray(st.attitude, float)                  # [qx, qy, qz, qw]
        if not np.all(np.isfinite(qa)) or np.linalg.norm(qa) < 1e-6:
            return None
        R = Rotation.from_quat(qa)
        r, p, y = R.as_euler("xyz"); RT = R.as_matrix().T
        g_b = RT @ np.array([0.0, 0.0, -1.0])
        return dict(pos=np.asarray(st.position, float), quat=np.array([qa[3], qa[0], qa[1], qa[2]]),
                    roll=float(r), pitch=float(p), yaw=float(y),
                    tilt=float(np.arccos(np.clip(-g_b[2], -1.0, 1.0))),
                    v_b=RT @ np.asarray(st.linear_velocity, float),
                    w_b=np.asarray(st.angular_velocity, float), g_b=g_b)

    def fallen(self, bs=None):
        """Down = tipped past fall_tilt, or dropped below 40% of flight altitude (crash)."""
        bs = bs or self.base_state()
        return (bs is None or bs["tilt"] > self.spec.fall_tilt
                or bs["pos"][2] - self._floor < 0.4 * self.spec.extra["alt"])

    _floor = 0.0

    def teleport(self, x, y, yaw, floor_z):
        self._floor = float(floor_z)
        p = np.array([x, y, floor_z + self.spec.extra["alt"]])
        # Through Pegasus' own dynamic-control handle, NOT the articulation tensor view: the
        # Pegasus vehicle's tensor view gets invalidated after the first Replicator frame
        # ("Simulation view object is invalidated"), dynamic control does not.
        from omni.isaac.dynamic_control import _dynamic_control as _dc
        dc = self.vehicle.get_dc_interface()
        rb = dc.get_rigid_body(self.prim_path + "/body")
        q = Rotation.from_euler("z", yaw).as_quat()             # x, y, z, w
        dc.set_rigid_body_pose(rb, _dc.Transform(p.tolist(), q.tolist()))
        dc.set_rigid_body_linear_velocity(rb, [0.0, 0.0, 0.0])
        dc.set_rigid_body_angular_velocity(rb, [0.0, 0.0, 0.0])
        self.backend.reset_ref(p, yaw)
        self.stop()

    def proprio(self, bs=None):
        """Rotors are the actuators: joint token = [0, rotor_vel, thrust_frac, 0,
        rotor_vel/max, thrust_frac, is_wheel=0]; thrust ~ rotor^2."""
        bs = bs or self.base_state()
        if bs is None:
            return None
        mx = self.spec.extra["max_rotor"]
        w = np.asarray(self.backend.rotor, float)
        th = (w / mx) ** 2
        joints = np.stack([np.zeros(4), w, th, np.zeros(4), w / mx, th, np.zeros(4)], 1).astype(np.float32)
        base = np.concatenate([bs["v_b"], bs["w_b"], bs["g_b"], self.cmd]).astype(np.float32)
        v_plan = float(np.hypot(bs["v_b"][0], bs["v_b"][1])); c_plan = float(np.hypot(*self.cmd[:2]))
        return base, joints, dict(tilt=bs["tilt"], track_err=float(np.linalg.norm(bs["v_b"][:2] - self.cmd[:2])),
                                  v_ratio=v_plan / c_plan if c_plan > 0.05 else 1.0,
                                  yaw_err=float(bs["w_b"][2] - self.cmd[3]), clipped=self.clipped)

    def descriptor(self, stage=None):
        mx = self.spec.extra["max_rotor"]
        dims = [0.55, 0.55, 0.2]
        if stage is not None:
            try:
                from pxr import UsdGeom
                cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
                d = cache.ComputeWorldBound(stage.GetPrimAtPath(self.prim_path)).ComputeAlignedRange().GetSize()
                dims = [float(d[0]), float(d[1]), float(d[2])]
            except Exception:
                pass
        return dict(kind=self.kind, family="aerial", n_joints=4,
                    joint_names=[f"rotor{i}" for i in range(4)],
                    joint_lower=[0.0] * 4, joint_upper=[0.0] * 4, joint_vmax=[mx] * 4,
                    joint_tau_max=[-1.0] * 4, mass_kg=self.spec.extra["mass"], bbox_xyz_m=dims,
                    max_cmd=self.limits(), variant=dict(self._variant), fall_tilt=self.spec.fall_tilt,
                    phys_hz=self.spec.phys_hz, v_min=self.spec.v_min)


def make_robot(kind, prim_path, position, yaw=0.0):
    """The right adapter for the body family."""
    return (DroneAdapter if SPECS[kind].family == "aerial" else RobotAdapter)(kind, prim_path, position, yaw)


class ProprioBuffer:
    """Fixed-rate ring buffer of proprio samples (the in-context history the model reads).
    Record at `hz` from a physics loop running at `phys_hz`."""

    def __init__(self, seconds=2.0, hz=50, phys_hz=500):
        self.n = int(round(seconds * hz))
        self.every = max(1, int(round(phys_hz / hz)))
        self.k = 0
        self.base, self.joints, self.extras = [], [], []

    def tick(self, adapter, bs=None):
        self.k += 1
        if self.k % self.every:
            return
        s = adapter.proprio(bs)
        if s is None:
            return
        b, j, e = s
        self.base.append(b); self.joints.append(j); self.extras.append(e)
        if len(self.base) > self.n:
            self.base.pop(0); self.joints.pop(0); self.extras.pop(0)

    def full(self):
        return len(self.base) >= self.n

    def arrays(self):
        """(T, N_BASE), (T, J, N_JOINT) float32 -- oldest first."""
        return np.stack(self.base), np.stack(self.joints)

    def reset(self):
        self.k = 0
        self.base.clear(); self.joints.clear(); self.extras.clear()
