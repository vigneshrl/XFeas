"""The fall rule used to score episodes. Knows that crawl and kneel modes tilt the body on purpose.

Details:
Mode-aware fall rule for episode scoring (additive 2026-10-19; episode_runner uses it only with EP_FALL_RULE=mode_aware,
for EVERY arm of a run -- stock, rule, gen3 alike).

A crawl / kneel / lying mode tilts the body ~80 deg BY DESIGN, so tilt alone is not a fall there. Rule:
  * FLOOR HIT (always a fall, any mode): torso / head top < 0.12 m above the floor, or the pelvis < 0.10 m above the
    floor, or the torso upside down / on its back (torso z-axis . world z < -0.7, tilt > 134 deg). (v1 of this file used
    SONIC's own -0.2, which scores an elbow crawl with the torso pitched 102-103 deg head-down as a fall although
    nothing touched the floor -- corrected 2026-10-19 00:25 to the coordinator's definition: torso / pelvis hits the floor.)
  * UPRIGHT modes: the usual rule (tilt > fall_tilt or pelvis below min_z) -- but only once RECOVER_S = 3 s have passed
    since the controller was last commanded a low (crawl / kneel / squat / lying) mode, i.e. the robot must be back
    up within 3 s of switching back to walking;
  * low modes: only the floor-hit test.
Robots without a commanded-mode attribute (mode_now) get the unchanged rule (tilt / min_z).
Floor = the path surface under the robot on stairs/course scenes, else the adapter's floor.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import numpy as np

RECOVER_S = 3.0
TOP_MIN, PELVIS_MIN, UPSIDE = 0.12, 0.10, -0.7  # upside down = torso tilted > 134 deg (on its back)


def _low_ids():
    try:
        import g1_sonic as GS

        ids = set(GS.LOW_MODES)
        for k, v in GS.MODES.items():
            if any(w in str(k).lower() for w in ("kneel", "squat", "lying", "crawl")):
                ids.add(int(v))
        return ids
    except Exception:
        return set()


class FallJudge:
    def __init__(self, rob, spec):
        self.rob, self.spec = rob, spec
        self.has_mode = hasattr(rob, "mode_now")
        self.low_ids = _low_ids() if self.has_mode else set()
        self.t_low_end = -1e9
        self.n_low = 0
        self.t_low = 0.0
        self.max_tilt_low = 0.0
        self.min_top = 9.9
        self.reason, self.t_fall, self._t_prev, self._was_low = None, None, None, False
        self.low_spans = []

    def _floor(self, fz):
        return float(fz) if fz is not None else float(getattr(self.rob, "_floor_z", 0.0))

    def fallen(self, t, bs, fz=None, fall_tilt=None):
        if bs is None:
            self.reason, self.t_fall = "no_state", t
            return True
        ft = float(fall_tilt if fall_tilt is not None else self.spec.fall_tilt)
        floor = self._floor(fz)
        pel = float(bs["pos"][2]) - floor
        dt = 0.0 if self._t_prev is None else t - self._t_prev
        self._t_prev = t
        low = self.has_mode and int(getattr(self.rob, "mode_now", -1)) in self.low_ids
        if low:
            if not self._was_low:
                self.n_low += 1
                self.low_spans.append([round(t, 2), None])
            self.t_low += dt
            self.t_low_end = t
            self.max_tilt_low = max(self.max_tilt_low, float(bs["tilt"]))
        elif self._was_low and self.low_spans:
            self.low_spans[-1][1] = round(t, 2)
        self._was_low = low
        # floor hit
        top = None
        up = None
        if self.has_mode:
            try:
                top = float(self.rob.top_z()) - floor
                R, _ = self.rob._torso_pose()
                up = float(R[2, 2])
            except Exception:
                pass
        if top is not None:
            self.min_top = min(self.min_top, top)
        if (top is not None and top < TOP_MIN) or (up is not None and up < UPSIDE) or pel < PELVIS_MIN:
            self.reason, self.t_fall = (
                f"floor_hit (top {top if top is None else round(top, 3)}, torso_up {up if up is None else round(up, 2)}, pelvis {pel:.3f})",
                t,
            )
            return True
        if low:
            return False
        if self.has_mode and t - self.t_low_end < RECOVER_S:
            return False
        if float(bs["tilt"]) > ft or pel < float(self.spec.min_z):
            rec = self.has_mode and self.n_low > 0 and t - self.t_low_end < RECOVER_S + 0.05
            self.reason = (
                "no_recovery_3s_after_low_mode" if rec else "tilt_or_low_pelvis_upright"
            ) + f" (tilt {np.degrees(bs['tilt']):.0f} deg, pelvis {pel:.3f})"
            self.t_fall = t
            return True
        return False

    def info(self):
        return dict(
            reason=self.reason,
            t_fall=None if self.t_fall is None else round(self.t_fall, 2),
            n_low_mode=self.n_low,
            t_low_mode=round(self.t_low, 2),
            low_spans=self.low_spans,
            max_tilt_low_deg=round(float(np.degrees(self.max_tilt_low)), 1),
            min_top_m=round(self.min_top, 3),
            recover_s=RECOVER_S,
        )
