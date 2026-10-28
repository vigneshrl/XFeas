"""
navigation-locomotion (Fu et al., CVPR 2022) RELEASED planner on SPOT (spotp, ReLIC), run exactly like VAMOS:
the planner's velocity command (lin, ang) goes to Spot's stock ReLIC controller as the twist [lin, 0, 0, ang].

Same code path as baselines/navigation-locomotion/arl_adapter_v2/navloco_released.NavLocoReleased (released FMM
planner + main.py loop, map-free occupancy from depth, released stop condition, 10 s perception wait), with only the
robot-specific mounts changed:
  * depth camera: the released D435 model (640x480, fx 385) rendered at SPOT'S HEAD CAMERA mount (spec.cam: 0.52 m
    forward, pitched 0.45 rad down; the same mount every other method's head camera uses);
  * odometry (T265-style): Spot's ground-truth pose at that mount relative to the episode start, + velocity;
  * commands: capped at speed_frac * max_cmd (benchmark protocol, same cap as VAMOS / the stock planner).
What is NOT used (and why): the released A1 locomotion policy (A1-only), and the safety advisor (collision detector /
fall predictor), which the paper trained on A1 proprioception but never released. So on Spot no A1-trained model
runs at all: the only proprioceptive-type input the released planner consumes is odometry. A learned advisor would
see Spot proprioception, which is out of its A1 training distribution; it is absent here as in the A1 runs.
    PYTHONPATH=...:baselines/navigation-locomotion/arl_adapter_v2  --robot spotp --robot-factory spot_posture:make_spotp
        --method navloco_spot:NavLocoSpot   (NAVLOCO_PY=scripts/xembody/navloco_py.sh)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math
import numpy as np
from scipy.spatial.transform import Rotation
import navloco_released as NR


class NavLocoSpot(NR.NavLocoReleased):
    name = "navloco_spot"
    label = "navigation-locomotion (released planner) + Spot ReLIC stock controller"

    def setup(self, ctx):
        super().setup(ctx)
        self.spec = ctx["spec"]
        print(f"[navloco_spot] released planner on {ctx['kind']}, head mount {self.spec.cam}", flush=True)

    def reset(self, ep):
        super().reset(ep)
        sp, f = self.spec, ep["speed_frac"]
        self.cap = np.array([f * sp.max_cmd[0], f * sp.max_cmd[4]])

    def _poses(self, bs):
        fwd, up, pitch = self.spec.cam
        q = bs["quat"]
        Rb = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        Tb = NR._T(Rb, np.asarray(bs["pos"], float))
        yaw = float(bs["yaw"]); c, s = math.cos(yaw), math.sin(yaw)
        eye = np.asarray(bs["pos"], float) + np.array([fwd * c, fwd * s, up])
        Ryaw = Rotation.from_euler("z", yaw).as_matrix()
        Tt = NR._T(Ryaw, eye)                                                     # T265-style frame at the head mount
        Rlink = Rotation.from_euler("ZY", [yaw, -pitch]).as_matrix()              # camera link: x fwd (pitched), z up
        To = NR._T(Rlink @ NR.OPT_IN_LINK, eye)                                   # optical frame in world
        return Tb, Tt, To

    def act(self, obs):
        r = super().act(obs)
        tw = np.asarray(r["twist"], float).copy()
        tw[0] = float(np.clip(tw[0], -self.cap[0], self.cap[0])); tw[3] = float(np.clip(tw[3], -self.cap[1], self.cap[1]))
        r["twist"] = tw
        return r
