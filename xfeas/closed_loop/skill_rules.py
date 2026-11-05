"""The skill rules used with the model (arms g5sr / g5sm). From the head-camera point memory they detect overhangs and
narrow gaps along the route and decide WHEN to switch skill (walk / crouch-crawl / sidestep) and how to stand back up.
Which command to use inside the allowed skill is picked by the feasibility model. Every threshold is a margin of the
robot's self-test measurements, not tuned on test scenes.

Details:
Skill rules for the gen5 'rules' arm.

  hazard map  while the robot is upright (camera looks ahead), the 4-frame head-camera point memory (the same points the
              model's feas v0.4 uses) is profiled along the lane: rows with an overhead lower than the standing top
              (+ STAND_MARGIN) are OVERHANG, 0.1 m slabs whose wall gap is narrower than the standing width (+ WIDTH_MARGIN)
              are NARROW. Detections are stored at path coordinate s + x and merged into intervals; nothing is updated while
              the robot is crouched / crawling / sidestepping (the head camera does not look ahead then).
  skill       overhang interval within [front - D_LOW, end + rel_low]       -> 'low'  (crouch / crawl classes; rel_low =
              half the crawling body depth + POS_MARGIN, end includes a doorway around the overhang)
              narrow interval   within [front - D_SIDE, end + REL_SIDE]     -> 'side' (sidestep left / right)
              otherwise                                                     -> 'walk' (stock / slow / fast)
              (overhang wins where both apply: crawling is the posture that fits under it)
  pick        argmax of the model's U = p - kappa*std - lambda*p_fall over the classes of that skill; the class chosen on
              entering a low / side interval is held until its release (no side switching inside a gap)
  stand-up    leaving 'low': stop in the low mode at zero speed (STOP_LOW s), then the walking command is sent with zero speed until the body top is back near standing height and level
              (or STAND_WAIT s), then walking resumes
Every constant is a geometry margin of the robot's probe fingerprint (stand_top, stand_width, stand_depth); none was tuned
on a test scene.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import numpy as np
import feas as FE
import clearmap as CLM

GROUPS = {
    "walk": ("stock", "slow", "fast"),
    "low": ("low1", "low2", "slowP_low1", "slowP_low2", "ground"),
    "side": ("latL", "latR"),
}
STAND_MARGIN, WIDTH_MARGIN = 0.05, 0.15
D_LOW, POS_MARGIN = 1.0, 0.3  # start crouching 1 m before the overhang front; stand when the whole crawling body
# (probe depth_at_min_top_walk, centred on the base) is POS_MARGIN past the hazard end
TILT_UP = 0.2  # stand-up is done only when the body is level again (gen4 rules: 0.15)
D_SIDE = 2.0  # sidestep turn needs room (10-31 sweep: d2.0 starts succeed, later ones stall)
X_RNG = (0.3, 4.5)
MERGE, MIN_HITS = 0.6, 3
STAND_WAIT, STAND_MIN, SETTLE = 5.0, 1.0, 1.5  # stand up in place (idle, zero speed), then walk slow for SETTLE s
# ablation (env XF_RULES_NOMODEL=1): the same rules, no model -- one fixed default command per skill (the controller's
# nominal walk, its own 'crouch' mode, sidestep to the right); the model is still asked, only for the log
import os

NOMODEL = os.environ.get("XF_RULES_NOMODEL", "0") == "1"
FIXED = {"walk": ("stock",), "low": ("low2", "low1", "ground"), "side": ("latR", "latL")}
# 11-03 (user): the MODEL picks the move, the rules only time it (env XF_RULES_MODELSKILL=1, arm g5sm): near an obstacle
# (any overhang / narrow interval, from D_ACT before it until past it) the model's best class over the WHOLE menu is executed
# and a crawl / sidestep is held until past; away from obstacles only walking gaits (model picks the speed); leaving a crawl
# = the staged stand-up below
MODELSKILL = os.environ.get("XF_RULES_MODELSKILL", "0") == "1"
D_ACT = 2.0
WALKS = ("stock", "slow", "fast", "careful", "stealth")
STANDUP = {"mode": "idle", "lin_vel_x": 0.0, "lin_vel_y": 0.0}
STOP_LOW = 3.0  # leaving 'low': first stop IN the low mode (zero speed) for STOP_LOW s, then stand up
# (idle), then walk -- the sequence verified on a flat floor 11-01 (sonic_seq_test.py);
# switching straight from a moving crawl to walk / idle fell ~50 % in the gen5b data
# 11-03 (additive, default off): env XF_COURSE_SAFE=1 (arm g5smC) -- general timing fixes for multi-obstacle courses, found
# on the full school course (the model still picks every move; no constant comes from a course position):
#  C1 re-observe on rising: the head camera cannot look ahead while low, so the stand-up itself is the re-observation. While
#     standing up (and in the low stop before it) an obstacle whose action window starts within MIN_WALK of the robot (window
#     front = obstacle front - D_ACT - MIN_WALK) re-opens the window: the model picks over the whole menu again (stand-up
#     aborted if it picks a low / lateral move). MIN_WALK = the crawling body depth of the probe (depth_at_min_top_walk): a
#     stand-up that cannot be followed by at least one crawl-body length of walking is not worth its transitions.
#  C2 felt overhead: a stand-up whose body top stalls (< STALL_DH over STALL_DT, after STAND_MIN) below the rules' own
#     'upright' height (stand_top - 0.15) is blocked by something the low camera could not see -> an overhang interval of the standing body
#     depth is added at the robot (MIN_HITS hits) and the window logic hands the move back to the model, restricted to the
#     classes whose probed body top (probe ops top_p50 of the class's mode; unprobed modes = stand_top) is at least
#     STAND_MARGIN below the felt clearance (the stalled top; the same margin the overhang profile uses). The model picks
#     among those. (Without the margin, SC2A run crs_g5smCB1 t1: the model picked crouch, top 1.084 under a felt 1.108,
#     and rocked in place under the roof.)
#  C3 cautious walk after a stand-up: the floor around a stand-up was not seen from an upright view while the robot was on
#     it -> for one crawl-body length after the stand-up only the menu's slowest walking gaits are allowed (model picks).
#  C4 walking-gait hold: a walking gait is kept GAIT_HOLD s before the model may switch to another one (no 0.25 s chatter).
#  C5 crouch-staged stand-up: after the low stop, a crawl is first turned into the controller's crouch (a low class whose
#     mode is not a crawl) at zero speed for CROUCH_T s, then idle. Off-course test (scripts/gen5/sonic_oil_stand_test.py,
#     flat floor, dry feet + hands on a mu 0.05 slab): hand_crawl -> idle FALLS, hand_crawl -> crouch_walk 0 -> idle and
#     -> squat -> idle stand; all three stand on uniformly dry and uniformly slippery floors.
COURSE = os.environ.get("XF_COURSE_SAFE", "0") == "1"
GAIT_HOLD, STALL_DT, STALL_DH = 1.0, 0.5, 0.02
# C2 is OFF unless XF_COURSE_C2=1 (11-03): as first run on SC2A its stall window kept only samples <= STALL_DT old, i.e. 2
# decisions 0.26 s apart, so it could never fire; the evaluated g5smC configuration is therefore C1 + C3 + C4 + C5. The
# window is fixed below (samples <= 1.5 STALL_DT old) but C2 has NOT been tested on a course.
C2_ON = os.environ.get("XF_COURSE_C2", "0") == "1"
CROUCH_T = 3.0


def profile(P, stand_top, stand_width):
    """decision-frame points (x fwd, y left, h above floor) -> (overhang xs, narrow xs) along the lane"""
    if P is None or len(P) == 0:
        return [], []
    cm = CLM.grid(P, 0.0)
    ground, free = cm[0], cm[2]
    xs, ys = FE._xs(cm.shape[1]), FE._ys(cm.shape[2])
    fl, ref = FE.floor_connected(ground, xs, return_ref=True)
    ok = np.isfinite(free) & (free < 9.0) & fl & (free >= FE.H_OVER_MIN_V04) & ~FE.wall_cells_local(P, ref)
    ok[:, np.abs(ys) > 0.30] = False
    rows = np.where((xs >= X_RNG[0]) & (xs <= X_RNG[1]))[0]
    oh = [float(xs[r]) for r in rows if (ok[r] & (free[r] < stand_top + STAND_MARGIN)).any()]
    if oh:  # extent: the floor under a slab is hidden from a standing camera, so the slab is
        loose = (
            np.isfinite(free) & (free < stand_top + STAND_MARGIN) & (free >= FE.H_OVER_MIN_V04)
        )  # followed through every
        loose[:, np.abs(ys) > 0.30] = False  # contiguous overhead row
        x_end, gap = max(oh), 0.0
        for r in rows:
            if xs[r] <= x_end:
                continue
            if loose[r].any():
                x_end, gap = float(xs[r]), 0.0
            else:
                gap += xs[1] - xs[0]
                if gap > 0.25:
                    break
        oh = oh + [x for x in xs[rows] if max(oh) < x <= x_end]
    nar = []
    for x0 in np.arange(X_RNG[0], X_RNG[1], 0.1):
        w, _ = FE.gap_from_points(P, x_rng=(x0, x0 + 0.1))
        if w < stand_width + WIDTH_MARGIN:
            nar.append(float(x0 + 0.05))
    return oh, nar


class SkillRules:
    def __init__(self, fp):
        self.stand_top, self.stand_width, self.stand_depth = (
            float(fp["stand_top"]),
            float(fp["stand_width"]),
            float(fp["stand_depth"]),
        )
        self.rel_side = 0.5 * self.stand_depth + 0.3
        self.rel_low = 0.5 * float(fp.get("depth_at_min_top_walk", 2 * self.stand_depth)) + POS_MARGIN
        # 11-01 TEST-ONLY (arm g5srL, school course: oil + tunnel right after the doorway, not seen while crawling): stay low
        # XF_RULES_STAYLOW m longer before standing up. Not part of the method; 0 = off (default)
        self.rel_low += float(os.environ.get("XF_RULES_STAYLOW", "0") or 0)
        self.min_walk = float(fp.get("depth_at_min_top_walk", 2 * self.stand_depth))  # XF_COURSE_SAFE: crawl body depth
        self.mode_top = {}  # C2: probed body top per controller mode (probe ops, not fallen)
        if COURSE:
            try:
                import json

                for o in json.load(open(os.environ["XFEAS_GEN3_PROBE"])).get("ops", []):
                    m, tp = (o.get("cmd") or {}).get("mode"), o.get("top_p50")
                    if m and tp is not None and np.isfinite(tp) and not o.get("fell"):
                        self.mode_top[m] = min(self.mode_top.get(m, 9.0), float(tp))
            except Exception as e:
                print("[skill_rules] XF_COURSE_SAFE: no probe ops for C2 (", repr(e), ")", flush=True)
        self.reset()

    def reset(self):
        self.iv = {"low": [], "side": []}  # [s0, s1, hits]
        self.skill, self.hold, self.su_t, self.settle_t, self.low_cmd = "walk", None, None, None, None
        self.events = []
        self.gait, self.gait_t, self.su_s, self.tops, self.felt = None, None, None, [], False  # XF_COURSE_SAFE state
        self.hold_end, self._last_win, self.felt_iv = None, None, []

    def _add(self, kind, ss):
        for s in ss:
            for I in self.iv[kind]:
                if I[0] - MERGE <= s <= I[1] + MERGE:
                    I[0], I[1], I[2] = min(I[0], s), max(I[1], s), I[2] + 1
                    break
            else:
                self.iv[kind].append([s, s, 1])
        self.iv[kind].sort()

    def observe(self, P, s, upright):
        if not upright:
            return None
        oh, nar = profile(P, self.stand_top, self.stand_width)
        self._add("low", [s + x for x in oh])
        self._add("side", [s + x for x in nar])
        return dict(oh=(min(oh), max(oh)) if oh else None, nar=(min(nar), max(nar)) if nar else None)

    def _active(self, kind, s, d_in, rel):
        for I in self.iv[kind]:
            end = I[1]
            if kind == "low":  # a doorway (narrow) around the overhang must be cleared too
                end = max(
                    [end]
                    + [
                        J[1]
                        for J in self.iv["side"]
                        if J[2] >= MIN_HITS and J[0] - MERGE <= I[1] and I[0] <= J[1] + MERGE
                    ]
                )
            if I[2] >= MIN_HITS and I[0] - d_in <= s <= end + rel:
                if kind == "side" and any(
                    J[2] >= MIN_HITS and J[0] - MERGE <= I[1] and I[0] <= J[1] + MERGE for J in self.iv["low"]
                ):
                    continue  # a narrow gap under an overhang is crawled (the 'low' skill)
                return I
        return None

    def _window(self, s, rel, d_act=D_ACT):
        """the merged obstacle interval (overhang and narrow together) with s in [front - d_act, end + rel], else None"""
        I = sorted([list(J[:2]) for k in ("low", "side") for J in self.iv[k] if J[2] >= MIN_HITS])
        M = []
        for a, b in I:
            if M and a <= M[-1][1] + MERGE:
                M[-1][1] = max(M[-1][1], b)
            else:
                M.append([a, b])
        return next((m for m in M if m[0] - d_act <= s <= m[1] + rel), None)

    def _want_model(self, s, by, table):
        rel = self.rel_low if self.skill == "low" else self.rel_side if self.skill == "side" else 0.0
        if (
            COURSE and self.skill == "low" and self.hold_end is not None
        ):  # C1: a low move re-picked during a stand-up is held
            if s <= self.hold_end + rel:  # until past the obstacle that caused it (only that one)
                return "low"
            self.hold_end = None
        d_act = (
            D_ACT + self.min_walk if (COURSE and self.su_t is not None) else D_ACT
        )  # C1 (stopping low / standing up: no
        self._last_win = self._window(s, rel, d_act)  # stand-up for < MIN_WALK of walking)
        if self._last_win is None:
            return "walk"
        if self.skill in ("low", "side"):
            return self.skill  # hold the model's crawl / sidestep until past the obstacle
        best = max(self._fit(table, s), key=lambda r: r["U"])["cls"]
        return next((k for k in ("low", "side") if best in GROUPS[k]), "walk")

    def _fit(self, rows, s):
        """C2: rows whose probed body top fits under a felt clearance active at s (all rows when none / none fit)"""
        hs = [h for a, b, h in self.felt_iv if a - MERGE <= s <= b + self.rel_low]
        if not COURSE or not hs:
            return rows
        ok = [
            r for r in rows if self.mode_top.get(str(r["cmd"].get("mode", "")), self.stand_top) < min(hs) - STAND_MARGIN
        ]
        return ok or rows

    def step(self, t, s, table, top, tilt=0.0):
        """-> (row to send, speed override or None, why)"""
        by = {r["cls"]: r for r in table}
        if MODELSKILL:
            want = self._want_model(s, by, table)
        elif self._active("low", s, D_LOW, self.rel_low) is not None:
            want = "low"
        elif self._active("side", s, D_SIDE, self.rel_side) is not None:
            want = "side"
        else:
            want = "walk"
        if COURSE and self.su_t is not None and want != self.skill:
            want = self._course_abort(t, s, want)
        if want != self.skill:
            self.events.append(dict(t=round(t, 2), s=round(s, 2), frm=self.skill, to=want))
            if self.skill == "low" and want == "walk":
                self.su_t = t
                self.low_cmd = dict(by[self.hold]["cmd"]) if self.hold in by else None
            self.skill, self.hold = want, None
            self.gait = None
        if self.su_t is not None and self.low_cmd is not None and t - self.su_t < STOP_LOW:  # 1. stop, still low
            return (
                dict(
                    cls="stoplow", cmd=dict(self.low_cmd, lin_vel_x=0.0, lin_vel_y=0.0), U=float("nan"), p=float("nan")
                ),
                0.0,
                "stop low (zero speed)",
            )
        if COURSE and self.su_t is not None and self.low_cmd is not None and t - self.su_t < STOP_LOW + CROUCH_T:  # C5
            cc = self._crouch_cmd(by)
            if cc is not None:
                return (
                    dict(cls="standup", cmd=cc, U=float("nan"), p=float("nan")),
                    0.0,
                    "stand-up stage: crouch (zero speed)",
                )
        if self.su_t is not None and self.low_cmd is not None:  # 2. stand-up phase starts after the stop
            self.su_t, self.low_cmd, self.tops, self.felt = t, None, [], False
        if COURSE and C2_ON and self.su_t is not None and top is not None and self._felt_overhead(t, s, top):  # C2
            want = self._course_abort(t, s, self._want_model(s, by, table))
            if want != self.skill:
                self.events.append(dict(t=round(t, 2), s=round(s, 2), frm=self.skill, to=want))
                self.skill, self.hold, self.gait = want, None, None
        if self.su_t is not None:  # stand-up in place: idle until the body is up again
            up = top is not None and top >= self.stand_top - 0.15 and tilt < TILT_UP
            if (up and t - self.su_t >= STAND_MIN) or t - self.su_t > STAND_WAIT:
                self.events.append(dict(t=round(t, 2), event="stand-up done", upright=bool(up)))
                self.su_t, self.settle_t, self.su_s = None, t, s
            else:
                return (
                    dict(cls="standup", cmd=dict(STANDUP), U=float("nan"), p=float("nan")),
                    0.0,
                    "stand-up (idle, in place)",
                )
        cls = WALKS if (MODELSKILL and want == "walk") else GROUPS[want]
        if COURSE and want == "walk":
            self.settle_t = None  # C3 replaces the timed settle
            if self.su_s is not None and s - self.su_s < self.min_walk:
                ws = [by[c] for c in cls if c in by]
                if ws:
                    vmin = min(float(r["cmd"].get("lin_vel_x", 0.0)) for r in ws)
                    cls = tuple(r["cls"] for r in ws if float(r["cmd"].get("lin_vel_x", 0.0)) <= vmin + 1e-6)
            else:
                self.su_s = None
        if want == "walk" and self.settle_t is not None:
            if t - self.settle_t < SETTLE:
                cls = ("stock", "slow")
            else:
                self.settle_t = None
        rows = [by[c] for c in cls if c in by]
        if not rows:  # the controller has no command of this skill -> nominal
            return table[0], None, f"{want}: no class, nominal"
        if want in ("low", "side"):
            if self.hold is None or self.hold not in by:
                self.hold = (
                    next(by[c] for c in FIXED[want] if c in by)
                    if NOMODEL and any(c in by for c in FIXED[want])
                    else max(self._fit(rows, s), key=lambda r: r["U"])
                )["cls"]
            row, why = by[self.hold], f"{want} hold {self.hold}"
        else:
            row, why = (
                (by["stock"], "walk nominal (no model)")
                if NOMODEL and "stock" in by
                else (max(rows, key=lambda r: r["U"]), "walk pick")
            )
            if COURSE and not NOMODEL:  # C4 walking-gait hold
                if self.gait in by and self.gait in cls and self.gait_t is not None and t - self.gait_t < GAIT_HOLD:
                    row, why = by[self.gait], "walk pick (gait hold)"
                elif row["cls"] != self.gait:
                    self.gait, self.gait_t = row["cls"], t
                if self.su_s is not None:
                    why += " (cautious after stand-up)"
        return row, None, why

    # ---------------------------------------------------------------- XF_COURSE_SAFE helpers
    def _course_abort(self, t, s, want):
        """C1 while stopping low / standing up: the model re-picks; a low pick aborts the stand-up (resume low), a lateral
        pick waits for the stand-up to finish (a lateral move is a walking mode: no crawl -> walk switch)"""
        if want == "low":
            self.events.append(dict(t=round(t, 2), s=round(s, 2), event="stand-up aborted: model picks low again"))
            self.su_t, self.low_cmd, self.su_s = None, None, None
            self.hold_end = None if self._last_win is None else float(self._last_win[1])
            return want
        return self.skill

    def _crouch_cmd(self, by):
        """C5: the menu's crouch (low class, mode not a crawl) at zero speed, only when leaving a crawl"""
        crawl = lambda c: "crawl" in str(c.get("mode", "")).lower()
        if self.low_cmd is None or not crawl(self.low_cmd):
            return None
        for c in GROUPS["low"]:
            if c in by and "mode" in by[c]["cmd"] and not crawl(by[c]["cmd"]):
                return dict(by[c]["cmd"], lin_vel_x=0.0, lin_vel_y=0.0)
        return None

    def _felt_overhead(self, t, s, top):
        """C2: body top stalled below standing height during the stand-up -> overhead the camera could not see"""
        if self.low_cmd is not None or self.felt:
            return False
        self.tops = [(a, b) for a, b in self.tops if t - a <= 1.5 * STALL_DT + 1e-6] + [(t, top)]
        if (
            t - self.su_t < STAND_MIN
            or top >= self.stand_top - 0.15
            or self.tops[-1][0] - self.tops[0][0] < 0.8 * STALL_DT
        ):
            return False
        if max(b for _, b in self.tops) - min(b for _, b in self.tops) >= STALL_DH:
            return False
        self.felt = True
        h = 0.5 * self.stand_depth
        self.iv["low"].append([s - h, s + h, MIN_HITS])
        self.iv["low"].sort()
        self.felt_iv.append((s - h, s + h, float(top)))
        self.events.append(dict(t=round(t, 2), s=round(s, 2), event=f"felt overhead (top {top:.2f} stalled)"))
        return True

    def summary(self):
        return {k: [[round(a, 2), round(b, 2), n] for a, b, n in v] for k, v in self.iv.items()}
