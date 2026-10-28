"""Isaac Sim Spot flat-terrain policy (robots.py kind 'spot', SpotFlatTerrainPolicy, as shipped) for the gen5 harness.
HELD OUT (GEN5_DESIGN 10a): probe only, never data. Thin adapter only (nothing about the policy changes):
  * _art = the policy's articulation (the gen3 harness reads joints / links / poses through `_art`);
  * command() also accepts a dict {vx, vy, wz} (the VRob proxy sends dicts in the base spec's names);
  * reset_setpoints() = zero twist (no posture commands exist).
    --base spotf  factory spot_flat5:make_spotf  (robot kind 'spot' internally; harness root /World/spotf)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import numpy as np
import robots as RB


class SpotFlat5(RB.RobotAdapter):
    def __init__(self, kind, prim_path, position, yaw=0.0):
        RB.RobotAdapter.__init__(self, "spot", prim_path, position, yaw)
        self.kind_harness = kind

    @property
    def _art(self):
        return self.robot

    def command(self, twist=None, **kw):
        if isinstance(twist, dict) or kw:
            d = dict(twist or {}, **kw)
            twist = [float(d.get("vx", 0.0)), float(d.get("vy", 0.0)), 0.0, float(d.get("wz", 0.0))]
        if twist is None:
            twist = self.cmd_req
        return RB.RobotAdapter.command(self, twist)

    def reset_setpoints(self):
        RB.RobotAdapter.command(self, np.zeros(4))


def make_spotf(kind, prim_path, position, yaw=0.0):
    return SpotFlat5(kind, prim_path, position, yaw)


make_spotf.SPEC = RB.SPECS["spot"]
