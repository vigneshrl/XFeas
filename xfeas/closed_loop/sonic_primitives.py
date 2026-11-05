"""Execution helpers for the GEAR-SONIC controller (enabled with XF_SONIC_PRIMS=1). The model / rules still choose the
move; these only change how SONIC's own commands carry it out (turn in place before a sidestep, hold heading and lateral
position while sidestepping or crawling). The SONIC policy itself is unchanged.

Details:
SONIC execution primitives. Pure python, no Isaac.

The model / skill rules still choose the MOVE (class latL / latR / low1 / low2 / slowP_low* / ...); these primitives only
change HOW the chosen move is executed with the off-the-shelf GEAR-SONIC controller's own commands (planner mode, speed,
body-frame vx / vy, yaw rate wz, body_height). The SONIC policy is untouched.

Diagnosis that motivated them (closed-loop logs c5_sonic_g5sr4/g5sr0/g5sm on HW1 / D1g / SC2A_O1):
  * the layer steers with wz capped at speed_frac * wz_max = 0.1 rad/s (local_nav.LocalNav.reset), so the 90 deg turn of
    a sidestep takes ~13 s and the travel starts while the body is still 25-30 deg off (straight_act turn-first gate
    0.5 rad), at walk 1.0 (clipped to vy 0.6) -> the body arrives 0.2-0.6 m off the gap centre and wedges on a wardrobe;
  * nothing holds the heading / lateral position during a sidestep or a crawl: SONIC's planner drifts in yaw (probe
    lat_L / lat_R: 0.17-0.30 rad/s) and sideways during hand crawl (SC2A_O1: 0.7 m lateral drift under the doorway).

Primitives (all with closed-loop heading hold wz = clip(K_YAW * err, +-WZ_MAX) and lateral centring on a lane line):
  side   ALIGN  : planner mode idle, zero velocity, turn in place to lane heading + face (+-90 deg) until |err| < ALIGN_TOL
                  for ALIGN_HOLD s (cap ALIGN_TMAX)
         SLIDE  : sidestep gait (SIDE_MODE, |v| = SIDE_V) along the lane, lateral velocity = K_LAT * (-e) (cap LAT_MAX),
                  heading held at lane heading + face
         BACK   : (when the layer leaves the sidestep) idle, zero velocity, turn back to the lane heading (|err| < BACK_TOL)
  crawl / crouch: the chosen low mode at its speed, heading = lane heading + atan(-e / LOOK) (pure pursuit on the lane line)
The lane line: straight scenes -> through the gap centre (perceived, see gap_lohi) or the start->goal line; course scenes
-> the path polyline the stock follower uses.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math, os
import numpy as np

K_YAW, WZ_MAX = 1.5, 0.6
K_LAT, LAT_MAX = 0.8, 0.15
LOOK = 0.8
ALIGN_TOL, ALIGN_HOLD, ALIGN_TMAX = math.radians(3.0), 0.4, 10.0
BACK_TOL, BACK_TMAX = math.radians(6.0), 8.0
SIDE_MODE = os.environ.get("XF_SP_SIDE_MODE", "slow_walk")
SIDE_V = float(os.environ.get("XF_SP_SIDE_V", "0.4"))
CRAWL_V = os.environ.get("XF_SP_CRAWL_V", "")
# CENTRE (after ALIGN, before SLIDE): SONIC does not track a small body-x component added to a sideways walk (HW1 oracle
# smoke: body vx +0.07..+0.15 commanded for 6 s, lateral error stayed 0.09-0.19 m), so the lane offset is removed first by a
# pure fore-aft slow_walk step (the body faces across the lane), then the slide starts
CEN_TOL, CEN_DONE, CEN_V, CEN_TMAX = float(os.environ.get("XF_SP_CEN_TOL", "0.025")), 0.012, 0.2, 6.0
CENTRE = os.environ.get("XF_SP_CENTRE", "0") == "1"  # default off: the validated HW1 recipe has no centre phase
# crawl / crouch heading hold: the pelvis yaw sways +-20-30 deg in SONIC's hand crawl (flat floor, sonic_prim_test hc04_h08),
# a stiff yaw loop on the raw yaw fights the gait (crawl 0.20-0.33 m/s with K 1.5 / cap 0.6 vs 0.36 m/s without), so the
# yaw is low-passed (time constant LOW_TAU) and the loop is gentle
LOW_TAU, LOW_K, LOW_WZ = (
    float(os.environ.get("XF_SP_LOW_TAU", "1.0")),
    float(os.environ.get("XF_SP_LOW_K", "1.0")),
    float(os.environ.get("XF_SP_LOW_WZ", "0.3")),
)  # '' = the class's own speed
# crouch_walk does not move below ~0.3 m/s (flat floor: commanded 0.2 -> 0.00 m/s, 0.4 -> 0.34 m/s, top 1.06-1.08 m)
# flat floor (sonic_prim_test cw04*, 3 trials each): crouch_walk 0.4 with wz exactly 0 -> top 1.06-1.075 m, 0.34 m/s, but
# 0.4-0.6 m lateral drift over 3.5 m; with ANY yaw-rate command (cap 0.1 / 0.2 / 0.3 / 0.6) -> top 1.18-1.22 m, 0.5-0.6 m/s.
# So the crouch is executed with zero yaw rate (no steering) and only over the obstacle span
CROUCH_NO_WZ = os.environ.get("XF_SP_CROUCH_NOWZ", "1") == "1"
CRAWL_NO_WZ = (
    os.environ.get("XF_SP_CRAWL_NOWZ", "0") == "1"
)  # test switch (flat: hand crawl with small wz 0.15-0.3 stalled 2/3)
CROUCH_VMIN = float(os.environ.get("XF_SP_CROUCH_VMIN", "0.4"))
LOW_MODES = ("hand_crawl", "handcrawl", "handcrawling", "elbow_crawl", "elbowcrawl", "elbowcrawling")
CROUCH_MODES = ("crouch", "crouch_walk", "crouchwalk")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _mkey(m):
    return str(m or "").lower().replace("_", "")


def kind_of(cmd):
    """role command -> 'side' | 'crawl' | 'crouch' | 'walk'"""
    cmd = cmd or {}
    if cmd.get("face"):
        return "side"
    m = _mkey(cmd.get("mode"))
    if m in [_mkey(x) for x in LOW_MODES]:
        return "crawl"
    if m in [_mkey(x) for x in CROUCH_MODES]:
        return "crouch"
    return "walk"


class Lane:
    """straight lane line: point C, unit direction d (travel), left normal n"""

    def __init__(self, C, d):
        self.C = np.asarray(C, float)[:2]
        d = np.asarray(d, float)[:2]
        self.d = d / max(1e-9, np.linalg.norm(d))
        self.n = np.array([-self.d[1], self.d[0]])
        self.h = math.atan2(self.d[1], self.d[0])

    def se(self, x, y):
        q = np.array([x, y]) - self.C
        return float(q @ self.d), float(q @ self.n)


def body_vel(vw, yaw):
    """world-plane velocity -> body frame (vx, vy)"""
    c, s = math.cos(yaw), math.sin(yaw)
    return c * vw[0] + s * vw[1], -s * vw[0] + c * vw[1]


class SonicPrim:
    def __init__(self):
        self.reset()

    def reset(self):
        self.kind, self.phase, self.t0, self.ok_t, self.face = "walk", None, None, None, 0.0
        self.log = []
        self.yf, self.tf = None, None  # low-passed yaw (crawl / crouch)

    def _ev(self, t, what):
        self.log.append(dict(t=round(t, 2), kind=self.kind, phase=what))

    def update(self, t, cmd):
        """the command in force (role dict) -> start / end primitives. Returns nothing."""
        k = kind_of(cmd)
        if k == "side":
            f = math.radians(float(cmd.get("face")))
            if self.kind != "side" or abs(wrap(f - self.face)) > 1e-3:
                self.kind, self.phase, self.t0, self.ok_t, self.face = "side", "align", t, None, f
                self._ev(t, "align")
            return
        if self.kind == "side":  # leaving a sidestep: turn back first
            if self.phase != "back":
                self.phase, self.t0, self.ok_t = "back", t, None
                self._ev(t, "back")
            return
        if k != self.kind:
            self.kind, self.phase, self.t0 = k, "run", t
            self._ev(t, "run")

    def active(self):
        return self.kind in ("side", "crawl", "crouch")

    def step(self, t, x, y, yaw, lane, cmd, e_ref=0.0):
        """-> (mode dict or None, twist [vx, vy, 0, wz] or None, label). None twist = primitive idle (layer default)."""
        s, e = lane.se(x, y)
        e = e - e_ref
        if self.kind == "side":
            if self.phase == "back":
                err = wrap(lane.h - yaw)
                if abs(err) < BACK_TOL or t - self.t0 > BACK_TMAX:
                    self._ev(t, "back done (%.1f deg)" % math.degrees(err))
                    self.kind, self.phase = kind_of(cmd), "run"
                    return None, None, "side: back done"
                return (
                    dict(mode="idle", speed=None),
                    np.array([0.0, 0.0, 0.0, float(np.clip(K_YAW * err, -WZ_MAX, WZ_MAX))]),
                    "side: turn back %.0f deg" % math.degrees(err),
                )
            tgt = wrap(lane.h + self.face)
            err = wrap(tgt - yaw)
            wz = float(np.clip(K_YAW * err, -WZ_MAX, WZ_MAX))
            if self.phase == "align":
                if abs(err) < ALIGN_TOL:
                    self.ok_t = self.ok_t if self.ok_t is not None else t
                else:
                    self.ok_t = None
                if (self.ok_t is not None and t - self.ok_t >= ALIGN_HOLD) or t - self.t0 > ALIGN_TMAX:
                    self.phase, self.t0 = ("centre" if CENTRE and abs(e) > CEN_TOL else "slide"), t
                    self._ev(t, "%s (err %.1f deg, e %.3f)" % (self.phase, math.degrees(err), e))
                else:
                    return (
                        dict(mode="idle", speed=None),
                        np.array([0.0, 0.0, 0.0, wz]),
                        "side: align %.0f deg" % math.degrees(err),
                    )
            if self.phase == "centre":
                if abs(e) < CEN_DONE or t - self.t0 > CEN_TMAX:
                    self.phase, self.t0 = "settle", t
                    self._ev(t, "settle (e %.3f)" % e)
                else:
                    vx, vy = body_vel(-math.copysign(CEN_V, e) * lane.n, yaw)
                    return dict(mode="slow_walk", speed=None), np.array([vx, vy, 0.0, wz]), "side: centre e %+.3f" % e
            if self.phase == "settle":  # 0.5 s idle between the fore-aft step and the slide
                if t - self.t0 < 0.5:
                    return dict(mode="idle", speed=None), np.array([0.0, 0.0, 0.0, wz]), "side: settle e %+.3f" % e
                self.phase, self.t0 = "slide", t
                self._ev(t, "slide (e %.3f)" % e)
            ve = float(np.clip(-K_LAT * e, -LAT_MAX, LAT_MAX))
            vw = SIDE_V * lane.d + ve * lane.n
            vx, vy = body_vel(vw, yaw)
            return (
                dict(mode=SIDE_MODE, speed=None),
                np.array([vx, vy, 0.0, wz]),
                "side: slide e %+.3f yaw err %+.1f" % (e, math.degrees(err)),
            )
        if self.kind in ("crawl", "crouch"):
            v = float(CRAWL_V) if CRAWL_V else float((cmd or {}).get("lin_vel_x") or 0.4)
            if self.kind == "crouch":
                v = max(v, CROUCH_VMIN)
            if self.yf is None or self.tf is None or t - self.tf > 0.5:
                self.yf = yaw
            else:
                a = min(1.0, (t - self.tf) / LOW_TAU)
                self.yf = wrap(self.yf + a * wrap(yaw - self.yf))
            self.tf = t
            tgt = wrap(lane.h + math.atan2(-e, LOOK))
            err = wrap(tgt - self.yf)
            wz = float(np.clip(LOW_K * err, -LOW_WZ, LOW_WZ))
            if (self.kind == "crouch" and CROUCH_NO_WZ) or (self.kind == "crawl" and CRAWL_NO_WZ):
                wz = 0.0  # any nonzero yaw rate lifts crouch_walk to ~1.20 m (see below)
            return None, np.array([v, 0.0, 0.0, wz]), f"{self.kind}: hold e {e:+.3f} yaw err {math.degrees(err):+.1f}"
        return None, None, "walk"


# ------------------------------------------------------------------ perception helper: gap centre (decision frame)
def gap_lohi(P, x_rng, h_low=(0.15, 0.5), h_wall=0.8, y_wall=1.2, y_ctr=0.5, min_pts=3):
    """as feas.gap_from_points over ONE slab x_rng, but -> (width, lo, hi) of the widest wall-to-wall free interval that
    overlaps the lane centre (|y| < y_ctr), or None"""
    if P is None or len(P) == 0:
        return None
    m = (
        (P[:, 0] >= x_rng[0])
        & (P[:, 0] <= x_rng[1])
        & (np.abs(P[:, 1]) < y_wall)
        & (P[:, 2] >= h_low[0])
        & (P[:, 2] <= 2.0)
    )
    S = P[m]
    if len(S) < min_pts:
        return None
    edges = np.arange(-y_wall, y_wall + 1e-9, 0.02)
    nb = len(edges) - 1
    low = S[S[:, 2] <= h_low[1]]
    blk = np.histogram(low[:, 1], edges)[0] >= 2
    if blk.sum() < 2:
        return None
    tall = np.histogram(S[S[:, 2] >= h_wall][:, 1], edges)[0] >= 1
    runs, i = [], 0
    while i < nb:
        if blk[i]:
            j = i
            while j + 1 < nb and blk[j + 1]:
                j += 1
            runs.append((i, j, bool(tall[max(0, i - 1) : min(nb, j + 2)].any())))
            i = j + 1
        else:
            i += 1
    best = None
    for (a0, a1, wa), (b0, b1, wb) in zip(runs[:-1], runs[1:]):
        lo, hi = edges[a1 + 1], edges[b0]
        if not (wa and wb) or hi - lo < 0.04:
            continue
        if hi > -y_ctr and lo < y_ctr and (best is None or hi - lo > best[0]):
            best = (float(hi - lo), float(lo), float(hi))
    return best
