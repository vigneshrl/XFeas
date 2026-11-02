"""The stock path follower: steers the robot along the route to the goal. Steering is never a model decision, so
every method (stock, XFeas, baselines) steers the same way.

Details:
The stock path follower of gen5 (pure python). Steering is NEVER a model choice: every controller gets
the stock arm's steering:
  * controllers with a yaw_rate command: wz = clip(2.0 * heading_err - 1.5 * lateral_err, +-min(1, wz_max))   (gensim.drive)
    where heading_err / lateral_err are taken w.r.t. the path segment under the robot;
  * direction-class controllers (spec binding steer_by_mode == 'php_quantise', PHP): while the chosen class is a MOVING
    one, the class is the release quantiser of the body-frame direction to a 0.6 m lookahead point (php_methods.PhpStock).
Path: a polyline in the world plane (straight lanes: two points). PathFollower gives arclength progress, lateral error,
heading error and the lookahead direction.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math
import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def php_quantise(vx, vy, wz, dead=0.05):
    """php_ref.quantise (DepthDistillationPolicy._apply_velocity), verbatim"""
    if math.hypot(vx, vy) < dead and abs(wz) < dead:
        return "stand"
    if abs(vx) < dead and abs(vy) < dead:
        return "left_90" if wz > 0 else "right_90"
    a = math.degrees(math.atan2(vy, vx))
    if -22.5 <= a <= 22.5:
        return "forward"
    if 22.5 < a <= 67.5:
        return "left_45"
    if 67.5 < a <= 112.5:
        return "left_90"
    if -67.5 <= a < -22.5:
        return "right_45"
    if -112.5 <= a < -67.5:
        return "right_90"
    return "back"


class PathFollower:
    def __init__(self, pts, look=0.6):
        P = np.asarray(pts, float)[:, :2]
        keep = [0]
        for i in range(1, len(P)):
            if np.linalg.norm(P[i] - P[keep[-1]]) > 1e-6:
                keep.append(i)
        self.P = P[keep]
        seg = np.diff(self.P, axis=0)
        self.L = np.hypot(seg[:, 0], seg[:, 1])
        self.cum = np.r_[0.0, np.cumsum(self.L)]
        self.U = seg / self.L[:, None]
        self.look = look
        self.total = float(self.cum[-1])

    def project(self, xy):
        """-> (arclength s, signed lateral e (left +), segment index)"""
        q = np.asarray(xy, float)[:2]
        best = None
        for i in range(len(self.L)):
            t = float(np.clip((q - self.P[i]) @ self.U[i], 0.0, self.L[i]))
            c = self.P[i] + self.U[i] * t
            d = float(np.linalg.norm(q - c))
            if best is None or d < best[0] - 1e-9:
                n = np.array([-self.U[i][1], self.U[i][0]])
                best = (d, float(self.cum[i] + t), float((q - c) @ n), i)
        return best[1], best[2], best[3]

    def point(self, s):
        s = float(np.clip(s, 0.0, self.total))
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.L) - 1))
        return self.P[i] + self.U[i] * (s - self.cum[i])

    def heading(self, i):
        return math.atan2(self.U[i][1], self.U[i][0])

    def errors(self, xy, yaw):
        s, e, i = self.project(xy)
        herr = wrap(self.heading(i) - yaw)
        la = self.point(s + self.look)
        q = np.asarray(xy, float)[:2]
        lerr = wrap(math.atan2(la[1] - q[1], la[0] - q[0]) - yaw)
        return dict(s=s, e=e, seg=i, herr=herr, lerr=lerr)


def follower_cmd(spec, names, err, moving=True, wz_cap=1.0):
    """-> {spec name: value} steering command (wz or a direction class); err = PathFollower.errors()"""
    import ctrl_spec as CS

    out = {}
    c = CS.cmd(spec, "yaw_rate")
    if c is not None:
        cap = min(wz_cap, float(c["range"][1]))
        out[c["name"]] = float(np.clip(2.0 * err["herr"] - 1.5 * err["e"], -cap, cap))
        return out
    if spec.get("binding", {}).get("steer_by_mode") == "php_quantise" or (
        CS.has(spec, "mode") and any(m.get("steer") for m in spec.get("modes", []))
    ):
        if moving:
            mc = CS.cmd(spec, "mode")
            cls = php_quantise(math.cos(err["lerr"]), math.sin(err["lerr"]), 0.0)
            if cls == "back":
                cls = "left_90" if err["lerr"] > 0 else "right_90"
            if cls in mc["values"]:
                out[mc["name"]] = cls
    return out
