#!/usr/bin/env python
"""
EXPLORATION COURSE -- one continuous L-shaped scene the robot traverses segment by segment,
with the feasibility model making every decision (collect_course.py / demo_course.py).

    leg 1 (+x)                                             corner      leg 2 (-y)
    start | OIL patch | dry | RUBBLE | dry | floor(mu2) -> STOP line | WALL
                                                       turn RIGHT here, then down leg 2:
                                                       floor(mu2) | GAP (w) | pocket | end WALL
                                                       ...then REVERSE back out through the gap

Segments (each a decision point with its own skill + continuous parameters):
    S1 cross_oil     forward v across the oil patch          (skill go, chunk +v)
    S2 cross_rubble  forward v across the rubble patch       (skill go, chunk +v)
    S3 stop          approach at v, stop at the painted line (skill stop, chunk +v then 0)
    S4 turn_right    yaw rate wz (<0) with creep v, 90 deg   (skill turn_in_place, chunk (v, wz))
    S5 enter_gap     forward v through the gap into the pocket (skill pass_gap)
    S6 reverse_out   REVERSE at v back out through the gap   (skill back_off, chunk -v)
  + back_off at S1/S5 (reverse away from the hazard) so forward-vs-reverse is a real choice.

Gap widths start at 0.8 m so Carter (0.63 m) and Jackal (0.43 m) can make it; the narrow end
still separates bodies (ANYmal 0.61 + drift, Carter at 0.8).

Everything is parameterised by a seed (sample_course) -> identical course for every robot
process -> region ids match across robots. Execution works in each segment's LOCAL frame
(origin + heading), so the same code runs leg 1 (+x) and leg 2 (-y), forward or backward.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math
import numpy as np

HW = 1.5  # corridor half width
X_OIL = (1.0, 3.0)
X_RUB = (4.0, 6.0)
X_CORNER = 6.5  # corner floor (mu2) starts here
X_WALL = 10.5  # leg-1 end wall face (4 m of corner floor: stop run-up + turn)
X_OPEN = (7.9, 10.5)  # opening on the right (-y) side into leg 2
XC = 0.5 * (X_OPEN[0] + X_OPEN[1])  # leg-2 centreline x (9.2)
Y_GAP = (-3.5, -3.0)  # gap blocks span in y
Y_END = -6.5  # pocket end wall face
LEG1 = dict(o=(0.0, 0.0), h=0.0)
LEG2 = dict(o=(XC, 0.0), h=-math.pi / 2)
SEGMENTS = ("cross_oil", "cross_rubble", "stop", "turn_right", "enter_gap", "reverse_out")
ABORT_M = 1.0  # back-off distance. Was 2 m: from the gap entrance that reversed straight into
# the leg-1 far wall, so gap back-offs 'failed' for geometry, not the robot.
SKILL_ID = {"go": 0, "stop_at_line": 1, "back_off": 2, "pass_gap": 4, "turn_in_place": 5, "refuse": 7}


def sample_course(seed):
    from terrain_classes import FLAT

    rng = np.random.default_rng(10_000 + seed)

    def surface():
        name, (lo, hi), col, jit = FLAT[int(rng.integers(len(FLAT)))]
        return dict(
            surface=name,
            mu=float(rng.uniform(lo, hi)),
            col=np.clip(np.array(col) + rng.normal(0, jit, 3), 0.01, 0.9).tolist(),
            rough=0.08 if name in ("oil", "wet", "contaminated") else 0.75,
        )

    C = dict(
        seed=seed,
        oil=surface(),
        corner=surface(),
        rubble=dict(
            h_max=float(rng.uniform(0.02, 0.09)), n=int(rng.integers(14, 34)), mu=0.7, seed=int(rng.integers(1 << 30))
        ),
        line_m=float(rng.uniform(0.7, 1.1)),
        gap_m=float(rng.uniform(0.8, 1.3)),
    )
    # oil segment should be SLIPPERY most of the time -- that is its purpose
    if rng.random() < 0.7:
        from terrain_classes import flat_by_name

        name = ["oil", "contaminated", "wet"][int(rng.integers(3))]
        _, (lo, hi), col, jit = flat_by_name(name)
        C["oil"] = dict(
            surface=name,
            mu=float(rng.uniform(lo, hi)),
            col=np.clip(np.array(col) + rng.normal(0, jit, 3), 0.01, 0.9).tolist(),
            rough=0.08,
        )
    return C


def build(stage, C, root="/World/Course"):
    """Author the course. Floors/walls/blocks/rubble collide; the stop line is visual only.
    Stable names for Replicator: *_dry* floors and *_wall*/_blk* get random colours; the oil
    and corner surfaces keep their friction-class look."""
    from pxr import UsdGeom
    from arena import _box

    UsdGeom.Xform.Define(stage, root)
    T, DRY, DRYC = 0.05, 0.9, (0.40, 0.39, 0.37)
    fl = lambda nm, x0, x1, y0, y1, mu, col, rough=0.8: _box(
        stage, f"{root}/{nm}", ((x0 + x1) / 2, (y0 + y1) / 2, -T), ((x1 - x0) / 2, (y1 - y0) / 2, T), mu, col, rough
    )
    # leg 1 floors
    fl("dry0", -2.0, X_OIL[0], -HW, HW, DRY, DRYC)
    fl("oil", X_OIL[0], X_OIL[1], -HW, HW, C["oil"]["mu"], C["oil"]["col"], C["oil"]["rough"])
    fl("dry1", X_OIL[1], X_RUB[0], -HW, HW, DRY, DRYC)
    fl("dry2", X_RUB[0], X_RUB[1], -HW, HW, 0.7, DRYC)  # rubble base
    fl("dry3", X_RUB[1], X_CORNER, -HW, HW, DRY, DRYC)
    fl("corner", X_CORNER, X_WALL, -HW, HW, C["corner"]["mu"], C["corner"]["col"], C["corner"]["rough"])
    # leg 2 floors (corner surface continues to the gap; pocket is dry)
    fl("corner2", X_OPEN[0], X_OPEN[1], Y_GAP[1], -HW, C["corner"]["mu"], C["corner"]["col"], C["corner"]["rough"])
    fl("dry4", X_OPEN[0], X_OPEN[1], Y_END, Y_GAP[1], DRY, DRYC)
    # rubble chunks
    from terrain_classes import rough_by_name, author_material, author_physics

    rr = np.random.default_rng(C["rubble"]["seed"])
    _, _, srange, _, base, _ = rough_by_name("rubble")
    for k in range(C["rubble"]["n"]):
        sx, sy = rr.uniform(*srange, 2)
        sz = rr.uniform(0.3, 1.0) * C["rubble"]["h_max"] / 2
        cx, cy = rr.uniform(X_RUB[0] + 0.15, X_RUB[1] - 0.15), rr.uniform(-HW + 0.2, HW - 0.2)
        p = _box(
            stage,
            f"{root}/rub{k:02d}",
            (cx, cy, sz),
            (sx, sy, sz),
            C["rubble"]["mu"],
            np.clip(np.array(base) * rr.uniform(0.8, 1.2), 0.02, 0.5),
        )
        from pxr import UsdGeom as _G, Gf

        _G.Xformable(p).MakeMatrixXform().Set(
            Gf.Matrix4d().SetScale(Gf.Vec3d(sx, sy, sz))
            * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), float(rr.uniform(0, 180))))
            * Gf.Matrix4d().SetTranslate(Gf.Vec3d(cx, cy, sz))
        )
    # stop line (visual)
    _box(stage, f"{root}/line", (X_WALL - C["line_m"], 0.0, 0.001), (0.03, HW, 0.001), None, (0.85, 0.75, 0.05), 0.6)
    # walls: leg-1 end wall, leg-1 sides (right side open at the corner), leg-2 sides + end
    WH, WT, WC = 0.6, 0.1, (0.55, 0.25, 0.18)
    wall = lambda nm, c, h: _box(stage, f"{root}/{nm}", c, h, DRY, WC)
    wall("wall_end1", (X_WALL + WT, 0.0, WH), (WT, HW + 2 * WT, WH))
    wall("wall_l1", ((-2.0 + X_WALL) / 2, HW + WT, WH), ((X_WALL + 2.0) / 2, WT, WH))
    wall("wall_r1", ((-2.0 + X_OPEN[0]) / 2, -HW - WT, WH), ((X_OPEN[0] + 2.0) / 2, WT, WH))
    wall("wall_l2", (X_OPEN[1] + WT, (-HW + Y_END) / 2, WH), (WT, (-HW - Y_END) / 2, WH))
    wall("wall_r2", (X_OPEN[0] - WT, (-HW + Y_END) / 2, WH), (WT, (-HW - Y_END) / 2, WH))
    wall("wall_end2", (XC, Y_END - WT, WH), ((X_OPEN[1] - X_OPEN[0]) / 2 + 2 * WT, WT, WH))
    # gap blocks (taller, so the drone can't just hop them)
    w = C["gap_m"]
    half = (X_OPEN[1] - X_OPEN[0]) / 2
    side = (half - w / 2) / 2
    for s, nm in ((-1, "a"), (1, "b")):
        _box(
            stage,
            f"{root}/blk{nm}",
            (XC + s * (w / 2 + side), (Y_GAP[0] + Y_GAP[1]) / 2, 0.8),
            (side, (Y_GAP[1] - Y_GAP[0]) / 2, 0.8),
            DRY,
            (0.30, 0.33, 0.40),
        )
    _box(stage, f"{root}/net", (XC / 2, -3.0, -2.0), (20.0, 20.0, 0.05), DRY, (0.08, 0.08, 0.08))


# ------------------------------------------------------------------ segment definitions
def segment(C, name, spec):
    """Start pose (x, y, yaw), local frame, goal, and the continuous action ranges."""
    vmin, vmax, vrev, wzmax = spec.v_min, spec.max_cmd[0], spec.max_cmd[1], spec.max_cmd[4]
    if name == "cross_oil":
        return dict(start=(-0.6, 0.0, 0.0), frame=LEG1, goal=X_OIL[1] + 0.4, skill="go", v=(vmin, vmax))
    if name == "cross_rubble":
        return dict(start=(3.0, 0.0, 0.0), frame=LEG1, goal=X_RUB[1] + 0.3, skill="go", v=(vmin, vmax))
    if name == "stop":
        return dict(start=(6.2, 0.0, 0.0), frame=LEG1, goal=X_WALL - C["line_m"], skill="stop_at_line", v=(vmin, vmax))
    if name == "turn_right":
        return dict(
            start=(XC, 0.0, 0.0),
            frame=LEG1,
            goal=-math.pi / 2,
            skill="turn_in_place",
            v=(0.0, min(0.3, vmax)),
            wz=(0.3, min(1.2, wzmax)),
            stand=True,
        )
    if name == "enter_gap":
        return dict(start=(XC, -0.4, -math.pi / 2), frame=LEG2, goal=-Y_GAP[0] + 0.9, skill="pass_gap", v=(vmin, vmax))
    if name == "reverse_out":
        return dict(
            start=(XC, -5.2, -math.pi / 2),
            frame=LEG2,
            goal=-Y_GAP[1] - 0.8,
            skill="back_off",
            v=(min(vmin, vrev), vrev),
            stand=True,
        )
    raise KeyError(name)


def sample_action(seg, rng, p_backoff=0.15):
    a = dict(skill=seg["skill"], v=round(float(rng.uniform(*seg["v"])), 3), wz=0.0)
    if seg["skill"] == "turn_in_place":
        a["wz"] = -round(float(rng.uniform(*seg["wz"])), 3)  # RIGHT turn
    elif seg["skill"] in ("go", "pass_gap") and rng.random() < p_backoff:
        a = dict(skill="back_off", v=round(float(rng.uniform(*seg["v"])), 3), wz=0.0, abort=True)
    return a


def encode(action, K=10):
    v, wz, sk = float(action.get("v", 0.0)), float(action.get("wz", 0.0)), action["skill"]
    ch = np.zeros((K, 4), np.float32)
    if sk == "stop_at_line":
        ch[:7, 0] = v
    elif sk == "turn_in_place":
        ch[:, 0] = v
        ch[:, 3] = wz
    elif sk == "back_off":
        ch[:, 0] = -v
    elif sk != "refuse":
        ch[:, 0] = v
    return ch, SKILL_ID[sk]


# ------------------------------------------------------------------ local frame + execution
def local(bs, frame):
    """(s along the segment heading, lateral offset, yaw error) of the base."""
    ox, oy = frame["o"]
    h = frame["h"]
    u = np.array([math.cos(h), math.sin(h)])
    n = np.array([-u[1], u[0]])
    d = bs["pos"][:2] - np.array([ox, oy])
    return float(d @ u), float(d @ n), (bs["yaw"] - h + math.pi) % (2 * math.pi) - math.pi


def keep(rob, bs, frame, v):
    """Lane keeping in the segment frame, forward (v>0) or reverse (v<0). Reversing flips
    how yaw moves the lateral error, hence the sign on the lateral term."""
    _, lat, ye = local(bs, frame)
    rob.command([v, 0, 0, float(np.clip(-1.5 * ye - math.copysign(1.2, v) * lat, -0.6, 0.6))])


def execute(world, rob, C, seg, action, HZ, front, on_step=None):
    every = max(1, HZ // 50)
    sk, v, wz = action["skill"], float(action.get("v", 0.0)), float(action.get("wz", 0.0))
    fr = seg["frame"]
    bs = rob.base_state()
    s0, _, ye0 = local(bs, fr)
    yaw0 = bs["yaw"]
    if sk == "refuse":
        rob.stop()
        for i in range(int(1.5 * HZ)):
            world.step(render=False)
            if on_step:
                on_step(i, i / HZ)
        return dict(
            outcome="refused",
            success=False,
            fell=False,
            collision=False,
            stall=False,
            progress=0.0,
            max_tilt=float(rob.base_state()["tilt"]),
        )
    if sk == "turn_in_place":
        tmax = abs(math.pi / 2 / wz) + 3.0
    elif sk == "back_off" and action.get("abort"):
        tmax = ABORT_M / max(v, 0.05) + 3.0  # abort: back away ABORT_M
    else:
        tmax = (abs(seg["goal"] - s0) + 2.0) / max(v, 0.08) + 5.0
    maxtilt, still, min_clear, stopped_at, res = 0.0, 0, 1e9, None, None
    for i in range(int(tmax * HZ)):
        world.step(render=False)
        t = i / HZ
        if on_step:
            on_step(i, t)
        if i % every:
            continue
        bs = rob.base_state()
        if bs is None or rob.fallen(bs):
            res = dict(outcome="fell", fell=True)
            break
        maxtilt = max(maxtilt, bs["tilt"])
        s, lat, ye = local(bs, fr)
        if sk == "stop_at_line":
            clear = X_WALL - (bs["pos"][0] + front)
            min_clear = min(min_clear, clear)
            if stopped_at is None:
                keep(rob, bs, fr, v)
                if clear <= C["line_m"]:
                    rob.stop()
                    stopped_at = t
            elif t - stopped_at > 2.5:
                ok = min_clear > 0.05
                res = dict(outcome="stopped" if ok else "hit_wall", collision=not ok, min_clear=round(min_clear, 3))
                break
        elif sk == "turn_in_place":
            rob.command([v, 0, 0, wz])
            turned = (bs["yaw"] - yaw0 + math.pi) % (2 * math.pi) - math.pi
            if turned <= -math.pi / 2 + 0.05:
                rob.stop()
                res = dict(outcome="turned", yaw_turned=round(turned, 3))
                break
        elif sk == "back_off":
            keep(rob, bs, fr, -v)
            if action.get("abort"):
                if s0 - s > ABORT_M:
                    res = dict(outcome="backed_off")
                    break
            elif s < seg["goal"]:  # back through the gap (s decreasing)
                res = dict(outcome="reversed_out")
                break
        else:
            keep(rob, bs, fr, v)
            if s > seg["goal"]:
                res = dict(outcome="crossed")
                break
        if stopped_at is None and sk != "turn_in_place":
            still = still + 1 if abs(bs["v_b"][0]) < 0.03 else 0
            if still > 150:
                res = dict(outcome="stuck", stall=True, collision=sk in ("pass_gap", "back_off"))
                break
    if res is None:
        res = dict(outcome="under_turn" if sk == "turn_in_place" else "timeout", stall=True)
    bs = rob.base_state()
    res.setdefault("fell", False)
    res.setdefault("collision", False)
    res.setdefault("stall", False)
    res["success"] = res["outcome"] in ("stopped", "turned", "crossed", "backed_off", "reversed_out")
    if bs is not None:
        s, _, _ = local(bs, fr)
        span = max(abs(seg["goal"] - s0), 0.5) if sk not in ("turn_in_place",) else 1.0
        res["progress"] = round(float(np.clip((s - s0) / span * (-1 if sk == "back_off" else 1), -1, 1.5)), 3)
    else:
        res["progress"] = 0.0
    res["max_tilt"] = round(maxtilt, 3)
    rob.stop()
    return res


def realign(world, rob, seg, HZ, on_step=None, tmax=6.0):
    """Between segments: steer back onto the segment frame's centreline, facing along it,
    before sensing. Training episodes start centred (teleport + small jitter); a robot that came
    off the rubble against a wall is otherwise out of distribution for the next decision.
    Skipped for segments that start at rest (turn, reverse)."""
    if seg.get("stand"):
        return
    fr = seg["frame"]
    every = max(1, HZ // 50)
    v = max(0.3, min(0.45, rob.limits()["vx_fwd"]))
    for i in range(int(tmax * HZ)):
        if i % every == 0:
            bs = rob.base_state()
            if bs is None:
                break
            _, lat, ye = local(bs, fr)
            if abs(lat) < 0.12 and abs(ye) < 0.08:
                break
            w = float(np.clip(-2.0 * ye - 1.6 * lat, -0.9, 0.9))
            rob.command([v * max(0.3, 1 - abs(ye)), 0, 0, w])
        world.step(render=False)
        if on_step:
            on_step(i, i / HZ)
    rob.stop()
