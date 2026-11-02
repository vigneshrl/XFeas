#!/usr/bin/env python
"""
FLAT HAZARD ARENA -- shared by collect_xe.py (training data) and demo_xe.py (model-driven
demo), so the demo can never drift from the geometry/materials the model was trained on.

Lanes run along +x at y = i * LANE_DY. Every lane: dry run-up [-2, RUN], hazard zone
[RUN, RUN+ZONE], dry outlet. Four hazard kinds (the Experiment-0 survivors):

  brake  zone = a surface CLASS (terrain_classes.FLAT: dry/dusty/wet/contaminated/oil, mu drawn
         from the class range). Wall at the zone end, a painted STOP LINE `line_m` before it
         (visual only). The robot must stop at the line; braking distance ~ v^2 / 2 mu g.
  turn   zone = a surface class; the robot turns on it. Wider lane.
  gap    two blocks leave an opening of width `gap_m`.
  curb   a step of height `curb_m` spanning the zone.

Appearance of the ZONE is tied to its friction class (that is the visual cue the model must
learn); everything else (run-up colour, walls, blocks, lights, distractors) is randomised by
Replicator in collect_xe.py so it carries no signal.

Lane parameters are drawn from a seeded RNG, so every robot's collector (one process per robot,
same --arena-seed) builds IDENTICAL places -> region_id matches across robots, which is what
robot ranking needs.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math
import numpy as np

RUN, ZONE = 3.5, 3.0
LANE_DY = 6.0
DRY_MU = 0.9
HAZARDS = ("brake", "turn", "gap", "curb")


def sample_lanes(seed, per_hazard=4, hazards=HAZARDS):
    """Deterministic lane specs. Friction classes are drawn so both easy and hard surfaces
    appear; geometric parameters span both-pass, robot-dependent and both-fail ranges (from
    the Experiment-0 probes)."""
    from terrain_classes import FLAT

    rng = np.random.default_rng(seed)
    lanes = []
    for hz in hazards:
        for k in range(per_hazard):
            L = dict(hazard=hz, idx=len(lanes))
            if hz in ("brake", "turn"):
                name, (lo, hi), col, jit = FLAT[int(rng.integers(len(FLAT)))]
                L.update(
                    surface=name,
                    mu=float(rng.uniform(lo, hi)),
                    col=np.clip(np.array(col) + rng.normal(0, jit, 3), 0.01, 0.9).tolist(),
                    rough=0.08 if name in ("oil", "wet", "contaminated") else 0.75,
                )
                if hz == "brake":
                    L["line_m"] = float(rng.uniform(0.3, 1.2))
                L["half_w"] = 1.5 if hz == "brake" else 2.6
            elif hz == "gap":
                L.update(gap_m=float(rng.uniform(0.4, 1.3)), half_w=1.5)
            else:
                L.update(curb_m=float(rng.uniform(0.02, 0.12)), half_w=1.5)
            L["y"] = L["idx"] * LANE_DY
            L["value"] = L.get("mu", L.get("gap_m", L.get("curb_m")))
            lanes.append(L)
    return lanes


def _box(stage, path, c, h, mu=None, col=None, rough=0.8):
    from pxr import UsdGeom, Gf
    from terrain_classes import author_physics, author_material
    import real_assets as _RA

    _RA.forbid_primitive("arena.py synthetic box")  # real-asset rule
    prim = UsdGeom.Cube.Define(stage, path).GetPrim()
    UsdGeom.Xformable(prim).MakeMatrixXform().Set(
        Gf.Matrix4d().SetScale(Gf.Vec3d(*h)) * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*c))
    )
    if mu is not None:
        author_physics(stage, prim, mu, path + "_pm")
    if col is not None:
        author_material(stage, prim, np.asarray(col, float), rough)
    return prim


def build(stage, lanes, root="/World/Arena"):
    """Author all lanes. Collision on floors/walls/blocks/steps; the stop line is visual only.
    Prim names are stable (Replicator randomisers match on them): *_run *_out *_wall *_blk*."""
    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, root)
    T = 0.05
    for L in lanes:
        i, y, hw = L["idx"], L["y"], L["half_w"]
        p = f"{root}/L{i:02d}"
        _box(stage, p + "_run", (-2.0 + (RUN + 2.0) / 2, y, -T), ((RUN + 2.0) / 2, hw, T), DRY_MU, (0.40, 0.39, 0.37))
        _box(stage, p + "_out", (RUN + ZONE + 2.0, y, -T), (2.0, hw, T), DRY_MU, (0.40, 0.39, 0.37))
        hz = L["hazard"]
        if hz in ("brake", "turn"):
            _box(stage, p + "_zone", (RUN + ZONE / 2, y, -T), (ZONE / 2, hw, T), L["mu"], L["col"], L["rough"])
            if hz == "brake":
                wx = RUN + ZONE
                _box(stage, p + "_wall", (wx + 0.1, y, 0.6), (0.1, hw, 0.6), DRY_MU, (0.55, 0.25, 0.18))
                _box(stage, p + "_line", (wx - L["line_m"], y, 0.001), (0.03, hw, 0.001), None, (0.85, 0.75, 0.05), 0.6)
        elif hz == "gap":
            _box(stage, p + "_zone", (RUN + ZONE / 2, y, -T), (ZONE / 2, hw, T), DRY_MU, (0.40, 0.39, 0.37))
            g = L["gap_m"]
            side = (hw - g / 2) / 2
            for s, nm in ((1, "a"), (-1, "b")):
                _box(
                    stage,
                    f"{p}_blk{nm}",
                    (RUN + 0.5, y + s * (g / 2 + side), 0.8),
                    (0.25, side, 0.8),
                    DRY_MU,
                    (0.30, 0.33, 0.40),
                )
        else:
            h = L["curb_m"]
            _box(stage, p + "_zone", (RUN + ZONE / 2, y, -T), (ZONE / 2, hw, T), DRY_MU, (0.40, 0.39, 0.37))
            _box(stage, p + "_step", (RUN + ZONE / 2, y, h / 2), (ZONE / 2, hw, h / 2), DRY_MU, (0.48, 0.45, 0.38))
    n = len(lanes)
    _box(
        stage, f"{root}/net", (5.0, (n - 1) * LANE_DY / 2, -2.0), (40.0, n * LANE_DY, 0.05), DRY_MU, (0.08, 0.08, 0.08)
    )


def goal_x(L):
    """x the robot's base must pass for 'crossed' (gap/curb)."""
    return RUN + 1.5 if L["hazard"] == "gap" else RUN + ZONE - 0.5


def wall_x():
    return RUN + ZONE
