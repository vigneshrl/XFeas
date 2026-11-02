"""
gen5 adapters for the WHEELED and AERIAL bodies (2026-10-25, GEN5_DESIGN 10a: Jackal + Iris, one controller each).
Thin adapters over the existing off-the-shelf controllers in xfeas/sim/robots.py (nothing is trained or retuned):

  jackal  Clearpath Jackal, robots.RobotAdapter kind 'jackal' = the shipped diff-drive controller (Isaac
          DifferentialController, skid-steer yaw-rate integral as in robots.py). Commands {vx, wz}.
  iris    Pegasus Iris, robots.DroneAdapter = Pegasus' geometric controller driven as a velocity tracker
          (robots._velocity_backend_cls). Commands {vx, vy, wz, alt}: 'alt' is a height-above-floor SETPOINT; it is
          turned into the vertical velocity of the tracker reference by a fixed P loop (vz = clip(K_ALT (alt - z), +-VZ_MAX)),
          i.e. the same reference-moving mechanism the velocity backend already uses for vx / vy / wz.

What the gen3 / gen4 / gen5 harness needs from an adapter and is added here (no behaviour change of the controller):
  * command() also accepts a dict in the spec's adapter names (the VRob proxy sends dicts);
  * reset_setpoints() (nominal command, every episode start);
  * iris: a `_art` shim over Pegasus' dynamic-control handles (the Pegasus vehicle's articulation tensor view is
    invalidated after the first Replicator frame, robots.DroneAdapter notes): world pose / velocities of the body,
    rotor speeds as joint velocities, link transforms of body + 4 rotors (the harness' link geometry / camera mount /
    fall judge read `_art._articulation_view._physics_view.get_link_transforms()`);
  * iris: fallen() = crash rule for a variable-altitude flight (body below CRASH_Z above the floor, or tilt > fall_tilt)
    instead of DroneAdapter's "below 40 % of the fixed 1 m altitude";
  * iris: teleport() to far below the floor (gensim5 parks the robot 30 m under the floor for the audit view) parks the
    drone 400 m away at floor height instead (Pegasus does not recover from the deep park; gensim_v1 notes).
Also patched process-locally (only processes that import this module, i.e. jackal / iris runs): cam4.OK_MOUNTS accepts
the jackal_front / iris_front mounts of xfeas/sim/camera_mount.py.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
import robots as RB

try:
    import cam4 as _C4

    _C4.OK_MOUNTS = dict(_C4.OK_MOUNTS)
    _C4.OK_MOUNTS["spot"] = tuple(_C4.OK_MOUNTS["spot"]) + (
        "jackal_front",
        "iris_front",
    )  # Cam4 labels non-G1 robots 'spot'
except Exception:  # noqa: BLE001
    pass

K_ALT, VZ_MAX = 1.5, 0.8  # altitude P loop (1/s, m/s)
JACKAL_FALL_TILT = 1.0  # rad (57 deg): the Jackal is tipping over
CRASH_Z = 0.12  # m: body origin below this above the floor = on the floor (body half height 0.067)


def _twist_from(d, cur):
    return [float(d.get("vx", cur[0])), float(d.get("vy", cur[1])), 0.0, float(d.get("wz", cur[3]))]


# ====================================================================== Jackal
class Jackal5(RB.RobotAdapter):
    def __init__(self, kind, prim_path, position, yaw=0.0):
        RB.RobotAdapter.__init__(self, "jackal", prim_path, position, yaw)
        self.kind_harness = kind

    def command(self, twist=None, **kw):
        if isinstance(twist, dict) or kw:
            twist = _twist_from(dict(twist or {}, **kw), self.cmd_req)
        if twist is None:
            twist = self.cmd_req
        return RB.RobotAdapter.command(self, twist)

    def reset_setpoints(self):
        RB.RobotAdapter.command(self, np.zeros(4))

    def fallen(self, bs=None):
        """tipped over: tilt > JACKAL_FALL_TILT. robots.py's 0.6 rad (34 deg) fired while the Jackal's front wheels were
        mounting a curb with all four wheels still on the ground side (rivermark smoke 10-26: tilt 0.62, body 0.14 m up)."""
        bs = bs or self.base_state()
        return bs is None or bs["tilt"] > JACKAL_FALL_TILT


def _quiet_physx_material_warning():
    """The Jackal's wheel meshes make PhysX print 'PxShape::getMaterialFromInternalFaceIndex received 0xFFFFffff' on every
    contact report (smoke 10-26: 70-100 MB of log per 8-situation job, which the queue re-reads every 20 s). Log only
    errors from the omni.physx.plugin channel in jackal processes (no effect on physics)."""
    try:
        import carb.settings

        st = carb.settings.get_settings()
        st.set("/log/channels/omni.physx.plugin", "error")
    except Exception as e:  # noqa: BLE001
        print(f"[xbody5] log filter failed: {e!r}", flush=True)


def make_jackal5(kind, prim_path, position, yaw=0.0):
    _quiet_physx_material_warning()
    return Jackal5(kind, prim_path, position, yaw)


make_jackal5.SPEC = RB.SPECS["jackal"]


# ====================================================================== Iris
class _PV:
    def __init__(self, ad):
        self.ad = ad

    def get_link_transforms(self):
        return self.ad._link_T()


class _AV:
    def __init__(self, ad):
        self.ad = ad
        self.body_names = list(ad.BODIES)
        self._physics_view = _PV(ad)

    def get_max_efforts(self):
        return np.full(4, -1.0)

    def get_dof_limits(self):
        return np.zeros((4, 2))


class _DroneArt:
    """articulation-like read access to the Pegasus vehicle through dynamic control"""

    def __init__(self, ad):
        self.ad = ad
        self.dof_names = [f"rotor{i}_joint" for i in range(4)]
        self._articulation_view = _AV(ad)

    def get_world_pose(self):
        T = self.ad._dc_pose("body")
        return np.array(T.p, float), np.array([T.r[3], T.r[0], T.r[1], T.r[2]], float)

    def get_linear_velocity(self):
        dc, rb = self.ad._rb("body")
        return np.array(dc.get_rigid_body_linear_velocity(rb), float)

    def get_angular_velocity(self):
        dc, rb = self.ad._rb("body")
        return np.array(dc.get_rigid_body_angular_velocity(rb), float)

    def get_joint_positions(self):
        return np.zeros(4, np.float32)

    def get_joint_velocities(self):
        return np.asarray(self.ad.backend.rotor, np.float32).reshape(-1)[:4]


class Iris5(RB.DroneAdapter):
    BODIES = ("body", "rotor0", "rotor1", "rotor2", "rotor3")

    def __init__(self, kind, prim_path, position, yaw=0.0):
        RB.DroneAdapter.__init__(self, "iris", prim_path, position, yaw)
        self.kind_harness = kind
        self.alt_t = float(self.spec.extra["alt"])
        self._floor_z = 0.0
        self._rbh = {}
        self._art = _DroneArt(self)
        self._effort_cap = None
        self._base_masses = None
        self.last_cmd_names = {}

    # ---- dynamic-control handles (dropped whenever the harness restarts physics: it sets ready False)
    @property
    def ready(self):
        return self.__dict__.get("_ready", False)

    @ready.setter
    def ready(self, v):
        if not v:
            self.__dict__["_rbh"] = {}
        self.__dict__["_ready"] = bool(v)

    def _rb(self, name):
        dc = self.vehicle.get_dc_interface()
        h = self._rbh.get(name)
        if h is None or h == 0:
            h = dc.get_rigid_body(f"{self.prim_path}/{name}")
            if h:
                self._rbh[name] = h
        return dc, h

    def _dc_pose(self, name):
        dc, rb = self._rb(name)
        return dc.get_rigid_body_pose(rb)

    def _link_T(self):
        out = np.zeros((len(self.BODIES), 7))
        for i, n in enumerate(self.BODIES):
            T = self._dc_pose(n)
            out[i, :3] = T.p
            out[i, 3:] = T.r  # x y z qx qy qz qw
        return out

    # ---- commands
    def command(self, twist=None, **kw):
        if isinstance(twist, dict) or kw:
            d = dict(twist or {}, **kw)
            if d.get("alt") is not None and float(d["alt"]) > 0:
                self.alt_t = float(np.clip(float(d["alt"]), 0.2, 3.0))
            twist = _twist_from(d, self.cmd_req)
        if twist is None:
            twist = self.cmd_req
        t = np.zeros(4)
        t[: len(twist)] = np.asarray(twist, float)[:4]
        t[2] = self.backend.cmd[2] if self.backend is not None else 0.0  # vz belongs to the altitude loop
        c = RB.RobotAdapter.command(self, t)
        self.backend.cmd = c.copy()
        return c

    def step(self, dt):
        self.ready = True
        try:
            z = float(self._dc_pose("body").p[2]) - self._floor_z
            vz = float(np.clip(K_ALT * (self.alt_t - z), -VZ_MAX, VZ_MAX))
            self.backend.cmd = np.array([self.cmd[0], self.cmd[1], vz, self.cmd[3]], float)
            self.cmd = self.backend.cmd.copy()
        except Exception:  # noqa: BLE001
            pass

    def stop(self):
        RB.RobotAdapter.command(self, np.zeros(4))
        self.backend.cmd = np.zeros(4)

    def reset_setpoints(self):
        self.alt_t = float(self.spec.extra["alt"])
        self.stop()

    # ---- state
    def fallen(self, bs=None):
        bs = bs or self.base_state()
        return bs is None or bs["tilt"] > self.spec.fall_tilt or float(bs["pos"][2]) - self._floor_z < CRASH_Z

    def teleport(self, x, y, yaw, floor_z):
        self._rbh = {}
        if float(floor_z) < self._floor_z - 10.0:  # the harness' deep park (audit view): park far away instead
            x, y, floor_z = float(x) + 400.0, float(y) + 400.0, self._floor_z
        else:
            self._floor_z = float(floor_z)
        self._floor = float(floor_z)
        p = np.array([x, y, floor_z + self.alt_t])
        from omni.isaac.dynamic_control import _dynamic_control as _dc

        dc, rb = self._rb("body")
        q = RB.Rotation.from_euler("z", yaw).as_quat()
        dc.set_rigid_body_pose(rb, _dc.Transform(p.tolist(), q.tolist()))
        dc.set_rigid_body_linear_velocity(rb, [0.0, 0.0, 0.0])
        dc.set_rigid_body_angular_velocity(rb, [0.0, 0.0, 0.0])
        self.backend.reset_ref(p, yaw)
        self.stop()

    def descriptor(self, stage=None):
        d = RB.DroneAdapter.descriptor(self, stage)
        d.update(
            controller="Pegasus Simulator Iris geometric controller as a velocity tracker (robots.py) + altitude P loop"
        )
        return d


def make_iris5(kind, prim_path, position, yaw=0.0):
    """Iris5 + the Pegasus vehicle removed from the World's scene REGISTRY (gensim_v1.make_iris: its articulation view
    is invalidated after the first Replicator frame and World.reset() would crash on it)"""
    rob = Iris5(kind, prim_path, position, yaw)
    try:
        from omni.isaac.core.world import World

        W = World.instance()
        nm = rob.vehicle.name
        if W.scene.object_exists(nm):
            W.scene.remove_object(nm, registry_only=True)
            print(f"[xbody5] iris: removed '{nm}' from the scene registry", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[xbody5] iris registry removal failed: {e}", flush=True)
    return rob


make_iris5.SPEC = RB.SPECS["iris"]
