"""
XFeas-Gen situation SPECS (pure python, seedable, no Isaac). A situation is fully defined by
    (family, params, location = lane + s_front + lateral, lighting, seed, robot)
and stored as one JSON object, so every situation is reproducible and auditable. gen/gensim.py builds it in Isaac.

Parameter ranges are PHYSICAL and robot-agnostic -- from trivial to impossible for every robot we have
(Spot/ReLIC standing top 1.01 m, lowest-crouch top ~0.86 m; G1/AMO standing top 1.27 m, forward-lean crouch 0.66 m;
Spot climbs ~0.08 m stock, ~0.17 m at best; widths Spot 0.55 m, G1 ~0.45 m). No per-robot band sampling: the oracle
search (gen/run_gen.py) finds where the interesting band is.

REAL ASSETS ONLY (2026-10-10, standing rule): every hazard is a duplicate of a real school asset (gensim builds them
with overhang_lib copies and real_assets.RealPool). The former primitive 'prim' overhang class is replaced by the real
classes 'partition' (cloakroom partition), 'bench' (gym benches) and a larger 'plank' share; step/curb/ledge/steps and
gap walls are built from real wooden planks and real furniture (lockers, bookcases, shelves). Specs keep their
parameters; a legacy spec with cls 'prim' is converted to a real plank of the same depth by gensim. The colour/'mat'
keys are still sampled (so seeds reproduce the same geometry draws) but are ignored by the real builders.
Parameter coverage with real assets (what the builders can reach; the BUILT value is measured by rays and stored):
  overhang underside u: any (objects are raised); depth: object-determined (planks 0.43-0.58, benches 0.27-0.49,
      partition 1.0, doors, windows, shelves ...), no free 0.10-0.25 m slab
  step heights: any 0.03-0.30 m (plank stack k x 0.040 x s, s 0.85-1.15, bottom layer <= 2 cm into the floor);
      'steps' risers snap to k x 0.040 x s (gaps 0.046-0.068 and 0.092-0.102 m -> nearest, |err| <= 0.011 m);
      curb depth 0.43-0.58 m (one plank deep) instead of 0.25 m; plank piles are 2.55-3.45 m wide
  gap: width any (furniture side faces placed exactly); wall thickness L snaps to furniture depths x 0.88-1.12
      (0.20-0.26, 0.33-0.51, 0.64-0.81, 0.83-1.09, 1.26-1.63 m), height 1.67-2.0 m
Families
  overhang  : one object over the lane (school-asset copy laid flat): underside u 0.40-1.90 m,
              depth 0.10-1.20 m, slant +-10 deg (60 %), yaw +-30 deg (70 %), roll +-5 deg (40 %), scale 0.85-1.2,
              classes as overhang_lib (the held-out 'cabinet' type and the floor-standing table are not used)
  step      : ledge (platform), curb (0.25 m bar, step up + down), mats (1-6 stacked school gym mats, copies),
              steps (2-4 box risers); front-edge height 0.03-0.30 m (mats 0.075-0.24), approach angle +-0.5 rad (50 %)
  gap       : a cross wall 1.6 m high with ONE gap: width 0.25-1.30 m, wall length (thickness along the lane)
              0.15-1.20 m, gap centre +-0.10 m
  null      : nothing on the lane (55 %), or a decoy that does not touch the lane: an object high above (u 1.95-2.4 m)
              or beside it (|lateral| 1.2-2.0 m), a box/mat beside the lane
  compound  : overhang (u 0.60-1.40, depth 0.20-0.60) then a curb (0.04-0.16 m) 0.6-1.4 m behind it  (H2-type)
Lighting: >= 60 % night with the robot headlight (night_white / night_yellow, 30 % of those pitch dark), the rest
original / day / dusk (as the v11 collectors).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, math
import numpy as np

FAMILIES = ("overhang", "step", "gap", "null", "compound")
OH_CLS_W_LEGACY = dict(prim=0.34, door=0.20, plank=0.12, grating=0.06, shelf=0.08, lamp=0.07, window=0.06, debris=0.07)
# NO FLOATING OBJECTS (2026-10-10): every raised object must bridge two real support columns standing outside the lane
# (|l| >= 0.65 m), so it has to reach |l| >= 0.80 m on both sides: lamps (1.0 m) and washbasins are dropped, slant is
# forced to 0 (a slanted board would slide off flat supports), 'beside' decoys lie on the floor.
OH_CLS_W = dict(partition=0.12, bench=0.10, door=0.27, plank=0.31, grating=0.06, shelf=0.08, window=0.06)
try:  # reproducing old suites only
    import os as _os

    if _os.environ.get("XFEAS_LEGACY_PRIMITIVES", "0") == "1":
        OH_CLS_W = OH_CLS_W_LEGACY
except Exception:
    pass
MAT_FIRST = ["SM_gym_props_12", "SM_gym_props_15", "SM_gym_props_13", "SM_gym_props_16"]
MAT_ALL = MAT_FIRST + ["SM_gym_props_14", "SM_gym_props_17"]


def r3(x):
    return round(float(x), 3)


def sample_lighting(rng, night_frac=0.70):
    if rng.random() < night_frac:
        return dict(mode="night_white" if rng.random() < 0.5 else "night_yellow", dark=bool(rng.random() < 0.30))
    r = rng.random()
    return dict(mode="original" if r < 0.6 else ("day" if r < 0.8 else "dusk"), dark=False)


def _overhang_obj(rng, u_lo, u_hi, d_lo, d_hi):
    cls = str(rng.choice(list(OH_CLS_W), p=np.array(list(OH_CLS_W.values())) / sum(OH_CLS_W.values())))
    o = dict(
        cls=cls,
        u=r3(rng.uniform(u_lo, u_hi)),
        yaw_rel=r3(rng.uniform(-0.52, 0.52)) if rng.random() < 0.7 else 0.0,
        slant=r3(rng.uniform(-0.175, 0.175)) if rng.random() < 0.6 else 0.0,
        roll=r3(rng.uniform(-0.09, 0.09)) if rng.random() < 0.4 else 0.0,
    )
    if cls == "prim":
        o.update(
            depth=r3(rng.uniform(d_lo, d_hi)),
            width=r3(rng.uniform(1.0, 3.5)),
            thick=r3(rng.uniform(0.03, 0.4)),
            color=[r3(c) for c in rng.uniform(0.05, 0.95, 3)],
            mat=int(rng.integers(1 << 20)) if rng.random() < 0.5 else None,
        )
    else:
        o.update(
            copy_idx=int(rng.integers(1 << 20)),
            across=bool(rng.random() < 0.75),
            scale=r3(rng.uniform(0.85, 1.2 if OH_CLS_W is OH_CLS_W_LEGACY else 1.15)),  # real rule: <= +-15 %
            depth_max=r3(rng.uniform(d_lo, d_hi)),
        )
    return o


def sample_params(rng, family, lane_len):
    """-> (params, s_front). All lengths in the lane frame (s along the lane from its start, l lateral)."""
    if family == "overhang":
        s0 = rng.uniform(2.6, 3.2)
        return dict(obj=_overhang_obj(rng, 0.40, 1.90, 0.10, 1.20), lat=r3(rng.uniform(-0.15, 0.15))), s0
    if family == "compound":
        s0 = rng.uniform(2.4, 2.7)
        o = _overhang_obj(rng, 0.60, 1.40, 0.20, 0.60)
        return dict(
            obj=o,
            lat=r3(rng.uniform(-0.10, 0.10)),
            curb_h=r3(rng.uniform(0.04, 0.16)),
            curb_gap=r3(rng.uniform(0.6, 1.4)),
        ), s0
    if family == "step":
        s0 = rng.uniform(2.6, 3.2)
        kind = str(rng.choice(["ledge", "curb", "mats", "steps"], p=[0.30, 0.25, 0.25, 0.20]))
        rot = r3(rng.uniform(-0.5, 0.5)) if rng.random() < 0.5 else 0.0
        p = dict(
            kind=kind,
            rot=rot,
            color=[r3(c) for c in rng.uniform(0.1, 0.9, 3)],
            mat=int(rng.integers(1 << 20)) if rng.random() < 0.6 else None,
        )
        if kind in ("ledge", "curb"):
            p.update(h=r3(rng.uniform(0.03, 0.30)), width=r3(rng.uniform(1.2, 2.4)), lc=r3(rng.uniform(-0.15, 0.15)))
        elif kind == "steps":
            p.update(h=r3(rng.uniform(0.04, 0.20)), n=int(rng.integers(2, 5)), tread=r3(rng.uniform(0.25, 0.45)))
        else:
            k = int(rng.integers(1, 7))
            srcs = []
            cnt = {}
            for i in range(k):
                s_ = str(rng.choice(MAT_ALL[4:])) if rng.random() < 0.2 else str(rng.choice(MAT_FIRST))
                while cnt.get(s_, 0) >= 3:
                    s_ = str(rng.choice(MAT_ALL))
                cnt[s_] = cnt.get(s_, 0) + 1
                srcs.append(s_)
            p.update(
                layers=k,
                srcs=srcs,
                short=bool(rng.random() < 0.3),
                lat=r3(rng.uniform(-0.2, 0.2)),
                jit=[[r3(rng.uniform(-0.04, 0.04)), r3(rng.uniform(-0.05, 0.05))] for _ in range(k)],
            )
        return p, s0
    if family == "gap":
        s0 = rng.uniform(2.6, 3.2)
        return dict(
            w=r3(rng.uniform(0.25, 1.30)),
            L=r3(rng.uniform(0.15, 1.20)),
            e=r3(rng.uniform(-0.10, 0.10)),
            color=[r3(c) for c in rng.uniform(0.1, 0.9, 3)],
            mat=int(rng.integers(1 << 20)) if rng.random() < 0.7 else None,
        ), s0
    if family == "null":
        s0 = rng.uniform(2.6, 3.2)
        r = rng.random()
        if r < 0.55:
            return dict(decoy="none"), s0
        if r < 0.75:
            o = _overhang_obj(rng, 1.95, 2.40, 0.10, 1.20)
            return dict(decoy="high", obj=o, lat=0.0), s0
        if r < 0.90:
            o = _overhang_obj(rng, 0.40, 1.20, 0.10, 1.20)
            return dict(decoy="beside_obj", obj=o, lat=r3(rng.choice([-1, 1]) * rng.uniform(1.6, 2.2))), s0
        return dict(
            decoy="beside_box",
            h=r3(rng.uniform(0.07, 0.30)),
            lc=r3(rng.choice([-1, 1]) * rng.uniform(1.4, 2.0)),
            color=[r3(c) for c in rng.uniform(0.1, 0.9, 3)],
        ), s0
    raise KeyError(family)


def make_situation(seed, family, lane, robot, split, night_frac=0.70):
    """one situation spec from an integer seed (the seed alone + lane + family reproduce it)"""
    rng = np.random.default_rng([int(seed), FAMILIES.index(family), sum(map(ord, robot))])
    L = float(lane.get("len", 5.6))
    params, s0 = sample_params(rng, family, L)
    return dict(
        sid=f"{split[:2]}_{family[:3]}_{robot}_{seed}",
        seed=int(seed),
        family=family,
        robot=robot,
        split=split,
        params=params,
        s_front=r3(s0),
        lane={k: lane[k] for k in ("id", "level", "start", "yaw", "floor", "zone", "split")} | dict(len=L),
        lighting=sample_lighting(rng, night_frac),
        d_dec=r3(rng.uniform(0.6, 1.6) if robot.startswith("spot") else rng.uniform(0.4, 1.2)),
        lat0=r3(rng.uniform(-0.06, 0.06)),
        yaw0=r3(rng.normal(0, 0.03)),
    )


def suite(seeds, families, lanes, robot, split, rng_seed):
    """deterministic list of situations: seed i -> family families[i % n] (balanced), lane drawn by a seeded rng"""
    rng = np.random.default_rng(rng_seed)
    out = []
    for i, sd in enumerate(seeds):
        fam = families[i % len(families)]
        lane = lanes[int(rng.integers(len(lanes)))]
        out.append(make_situation(sd, fam, lane, robot, split))
    return out


if __name__ == "__main__":
    import os, sys

    ROOT = xfeas.paths.ROOT
    L = json.load(open(os.path.join(ROOT, "data", "gen", "lanes.json")))["lanes"]
    tr = [l for l in L if l["split"] == "train"]
    S = suite(range(10), FAMILIES, tr, "spotp", "val", 1)
    for s in S:
        print(json.dumps(s)[:300])
