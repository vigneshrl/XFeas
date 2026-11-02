"""Narrow-passage arms for episode_runner.py (scenes N*, sidestep_scene.py). No planner, straight drive.

  stock_straight   (door_methods.StraightStock) -- the stock controller walks forward to the goal.
  SidestepOracle   ORACLE arm (not a method; uses the scene's gap position): walk forward to 0.6 m before the wall,
                   turn 90 deg with the controller's own yaw command, walk SIDEWAYS (vy) through the gap holding the
                   heading and centring on the gap, turn back once the body is past the wall, walk to the goal.
                   Tells whether "turn + sidestep" can pass the gap at all with the unchanged controller.

G1/AMO quirk (released deployment): |vx| < 0.1 puts the policy in stand mode (no stepping, no turning), so during the
turn and the sidestep vx is kept at magnitude 0.1, its sign chosen to centre the body on the gap (alternating each
second when centred). Turn direction: G1 tracks +vy much better than -vy (sidestep_probe.py: 0.42-0.54 vs 0.17-0.26
of the command), so the robot faces RIGHT of the lane (yaw = heading - 90 deg) and walks with +vy.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math
import numpy as np
import local_nav as LN
from door_methods import straight_act


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class SidestepOracle(LN.LocalNav):
    name = "sidestep_oracle"
    label = "ORACLE: scripted turn 90 deg + sidestep (vy)"
    D_TRIG, PAST = 0.6, 0.45

    def setup(self, ctx):
        super().setup(ctx)
        self.g1 = ctx["kind"].startswith("g1")

    def reset(self, ep):
        super().reset(ep)
        import sidestep_scene as NS

        self.sc = NS.N_SCENES[ep["scene"]]
        self.s0, self.s1 = NS.geometry(self.sc)["s_wall"]
        self.phase, self.t_phase = "approach", 0.0
        self.vy = 0.4 if self.g1 else 0.5
        self.vxm = 0.1 if self.g1 else 0.0
        self.D_TRIG = 0.6 if self.g1 else 1.1  # Spot's arm reaches 1.0 m ahead of its base

    def lane(self, x, y):
        ox, oy = self.sc["origin"]
        h = self.sc["heading"]
        dx, dy = x - ox, y - oy
        return dx * math.cos(h) + dy * math.sin(h), -dx * math.sin(h) + dy * math.cos(h)

    def act(self, obs):
        bs, t = obs["bs"], obs["t"]
        x, y, yaw = float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])
        s, e = self.lane(x, y)
        h = self.sc["heading"]
        g = self.sc["gap_e"]
        cap = self.spec.max_cmd[4]
        if self.phase == "approach":
            # aim at the gap entry point, then turn
            ax, ay = self.sc["origin"]
            sa = self.s0 - self.D_TRIG
            tx = ax + sa * math.cos(h) - g * math.sin(h)
            ty = ay + sa * math.sin(h) + g * math.cos(h)
            if s >= sa - 0.05:
                self.phase, self.t_phase = "turn", t
            else:
                err = wrap(math.atan2(ty - y, tx - x) - yaw)
                tw = np.array(
                    [self.v if abs(err) < 0.5 else 0.0, 0.0, 0.0, float(np.clip(2.0 * err, -self.wz_cap, self.wz_cap))]
                )
                return dict(twist=tw, info="oracle: approach")
        if self.phase == "turn":
            err = wrap(h - math.pi / 2 - yaw)
            if abs(err) < math.radians(4):
                self.phase, self.t_phase = "side", t
            else:
                return dict(
                    twist=np.array([self.vxm, 0.0, 0.0, float(np.clip(1.5 * err, -cap, cap))]),
                    info="oracle: turn 90 deg",
                )
        if self.phase == "side":
            if s >= self.s1 + self.PAST:
                self.phase, self.t_phase = "back", t
            else:
                err = wrap(h - math.pi / 2 - yaw)
                want = -1.5 * (g - e)  # body +x points to lane -e
                if self.vxm > 0:
                    vx = (
                        math.copysign(self.vxm, want)
                        if abs(g - e) > 0.03
                        else (self.vxm if int(t) % 2 == 0 else -self.vxm)
                    )
                else:
                    vx = float(np.clip(want, -0.2, 0.2))
                return dict(
                    twist=np.array([vx, self.vy, 0.0, float(np.clip(1.5 * err, -cap, cap))]),
                    info=f"oracle: sidestep vy {self.vy:.2f}",
                )
        if self.phase == "back":
            err = wrap(math.atan2(obs["goal_xy"][1] - y, obs["goal_xy"][0] - x) - yaw)
            if abs(err) < math.radians(8):
                self.phase = "go"
            else:
                return dict(
                    twist=np.array([self.vxm, 0.0, 0.0, float(np.clip(1.5 * err, -cap, cap))]), info="oracle: turn back"
                )
        r = straight_act(self, obs)
        r["info"] = "oracle: forward to goal"
        return r
