"""
gen_v1 situation SPECS for the non-school TRAINING environments (pure python, seedable; built by envsim.EnvSim).
Hazards are built ONLY from each environment's OWN assets (envsim.CATALOG: the env's prop USD files, referenced with
their own meshes + materials); every raised object rests on real support columns (stacks of the env's own
boxes / pallets / stands) outside the lane; nothing floats.

Families (params are physical, robot-agnostic: trivial -> impossible for all 4 robots; the oracle finds the band):
  env_overhang  a flat slab of the env (warehouse rack shelf / bracket beam, office door leaf) across the lane on two
                support stacks: underside u 0.55-1.65 m, yaw +-0.3 rad (60 %), lateral +-0.1   -> bucket setpoint
  env_gap       two rows of the env's tall furniture / box stacks with ONE gap: width 0.30-1.30 m  -> bucket setpoint
  env_step      a row of the env's low objects across the lane (pallets, boxes, rack shelf on the floor, ...),
                optionally 2 stacked                                                               -> bucket scene
  env_clutter   3-9 small env objects scattered over 1.6 m of the lane                             -> bucket scene
  env_curb      (Rivermark) the env's OWN curb on a curb lane (nothing built), step up or down    -> bucket scene
  terrain       one Isaac Lab terrain tile (scripts/terrain catalog: 8 rough + 6 slippery types, difficulty
                0.15-1.0), crossed over 5 m                                                         -> rough / slippery
  null          nothing (65 %) or an env object beside the lane (|l| 1.25-1.8 m)                  -> null
Lighting: families.sample_lighting (night >= 60 %: 70 % night_white / night_yellow, 30 % of those dark).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
import families as FA

r3 = FA.r3
WAREHOUSE_FAMS = (["env_overhang", "env_gap", "env_step", "env_clutter", "null"], [0.24, 0.18, 0.20, 0.18, 0.20])
DEFAULT_FAMS = dict(
    office=WAREHOUSE_FAMS,
    warehouse_shelves=WAREHOUSE_FAMS,
    full_warehouse=WAREHOUSE_FAMS,
    warehouse=WAREHOUSE_FAMS,
    rivermark=(["env_curb", "null"], [0.6, 0.4]),
    simple_room=(["null"], [1.0]),
    dt_warehouse=(["null"], [1.0]),
    terrain=(["terrain"], [1.0]),
)
SLAB = dict(office=dict(door=1.0), warehouse=dict(beam=1.0))
STEP = dict(
    office=dict(box=0.40, boxbig=0.35, boxport=0.25), warehouse=dict(pallet=0.45, cardbox_c=0.30, rackshield=0.25)
)
CLUT = dict(
    office=["box", "boxa", "boxbig", "boxport", "binder", "papertray", "book", "bottle", "chair"],
    warehouse=["cardbox_d", "cardbox_c", "bottle", "bracket", "cardbox_d2"],
)
# climbable steps (2026-10-11): rows / small stacks of the env's own LOW props, front edge targeted at 0.07-0.17 m
# (Spot's climb headroom; 85 %) or 0.18-0.32 m (infeasible control; 15 %); kind / stack / scale solved in envsim
CLIMB = dict(
    office=["box", "boxa", "binder", "book", "boxbig"],
    warehouse=["crate_e", "crate_c", "crate_nb", "cardbox_d1", "cardbox_d2", "pallet", "cardbox_c"],
)
CLIMB_FLAT = ("binder", "book")  # laid flat (thinnest axis vertical): binders 0.08 m, books ~0.1 m per layer
TERRAIN_TYPES = None  # optional restriction (make_suite_v1 --types)
GAP_RANGE = None  # optional gap-width band for 80 % of env_gap (make_suite_v1 --gap-w), e.g. G1 sidestep regime
WALL = dict(office=["cupboard", "filecab"], warehouse=["cardbox_a"])


def pack(env):
    return "office" if env == "office" else "warehouse"


INDOOR_ONLY = ("office", "warehouse_shelves", "full_warehouse", "warehouse", "dt_warehouse")


def lane_ok(family, lane, env=None):
    if env in INDOOR_ONLY and not lane.get("indoor", True):  # office.usd sits in a city backdrop: stay inside
        return False
    if family == "env_curb":
        return lane.get("kind") == "curb"
    if lane.get("kind") == "curb":
        return False
    if family in ("env_overhang", "env_gap", "env_compound"):
        return lane.get("clr_haz", 0) >= 1.25
    return True


def _pick(rng, d):
    k = list(d)
    p = np.array([d[x] for x in k], float)
    return str(rng.choice(k, p=p / p.sum()))


def sample_params(rng, family, env, lane, i_in_field=0):
    pk = pack(env)
    s0 = rng.uniform(2.6, 3.2)
    if family == "env_overhang":
        return dict(
            slab=_pick(rng, SLAB[pk]),
            u=r3(rng.uniform(0.55, 1.65)),
            yaw_rel=r3(rng.uniform(-0.3, 0.3)) if rng.random() < 0.6 else 0.0,
            lat=r3(rng.uniform(-0.1, 0.1)),
            scale=r3(rng.uniform(0.9, 1.1)),
            sup_seed=int(rng.integers(1 << 20)),
        ), s0
    if family == "env_climb":
        h = rng.uniform(0.07, 0.17) if rng.random() < 0.85 else rng.uniform(0.18, 0.32)
        plat = bool(rng.random() < 0.6)  # 60 %: a platform to climb ONTO (goal on top), else a row to cross
        return dict(
            h=r3(h),
            rot=r3(rng.uniform(-0.4, 0.4)) if (rng.random() < 0.4 and not plat) else 0.0,
            lat=r3(rng.uniform(-0.15, 0.15)),
            kseed=int(rng.integers(1 << 20)),
            platform=plat,
            pdepth=r3(rng.uniform(1.3, 2.0)),
        ), s0
    if family == "env_compound":
        s0 = rng.uniform(2.0, 2.3)
        return dict(
            slab=_pick(rng, SLAB[pk]),
            u=r3(rng.uniform(0.60, 1.40)),
            yaw_rel=r3(rng.uniform(-0.2, 0.2)) if rng.random() < 0.5 else 0.0,
            lat=r3(rng.uniform(-0.08, 0.08)),
            scale=r3(rng.uniform(0.9, 1.1)),
            sup_seed=int(rng.integers(1 << 20)),
            gap=r3(rng.uniform(0.5, 1.0)),
            h=r3(rng.uniform(0.06, 0.17)),
            kseed=int(rng.integers(1 << 20)),
        ), s0
    if family == "env_gap":
        if GAP_RANGE is not None and rng.random() < 0.8:
            return dict(
                w=r3(rng.uniform(*GAP_RANGE)),
                e=r3(rng.uniform(-0.10, 0.10)),
                wall=str(rng.choice(WALL[pk])),
                depth2=bool(rng.random() < 0.35),
                wseed=int(rng.integers(1 << 20)),
            ), s0
        if rng.random() < 0.7:  # G1 / Spot width band (G1 ~0.45 m, Spot 0.55 m)
            return dict(
                w=r3(rng.uniform(0.30, 0.85)),
                e=r3(rng.uniform(-0.10, 0.10)),
                wall=str(rng.choice(WALL[pk])),
                depth2=bool(rng.random() < 0.35),
                wseed=int(rng.integers(1 << 20)),
            ), s0
        return dict(
            w=r3(rng.uniform(0.30, 1.30)),
            e=r3(rng.uniform(-0.10, 0.10)),
            wall=str(rng.choice(WALL[pk])),
            depth2=bool(rng.random() < 0.35),
            wseed=int(rng.integers(1 << 20)),
        ), s0
    if family == "env_step":
        k = _pick(rng, STEP[pk])
        stack = 2 if (k in ("pallet", "box", "cardbox_c") and rng.random() < 0.35) else 1
        return dict(
            kind=k,
            stack=stack,
            rot=r3(rng.uniform(-0.4, 0.4)) if rng.random() < 0.4 else 0.0,
            scale=r3(rng.uniform(0.9, 1.1)),
            lat=r3(rng.uniform(-0.15, 0.15)),
        ), s0
    if family == "env_clutter":
        n = int(rng.integers(3, 10))
        items = [
            dict(
                kind=str(rng.choice(CLUT[pk])),
                ds=r3(rng.uniform(0.0, 1.6)),
                l=r3(rng.uniform(-0.7, 0.7)),
                yaw=r3(rng.uniform(-math.pi, math.pi)),
                scale=r3(rng.uniform(0.9, 1.1)),
            )
            for _ in range(n)
        ]
        if not any(abs(it["l"]) < 0.3 for it in items):  # at least one object on the lane centre band
            items[0]["l"] = r3(rng.uniform(-0.25, 0.25))
        return dict(items=items), s0
    if family == "env_curb":
        return dict(rise=lane["rise"], s_riser=lane["s_riser"]), float(lane["s_riser"])
    if family == "terrain":
        import catalog as CT

        types = TERRAIN_TYPES or (CT.ROUGH + CT.SLIPPERY)
        t = types[int(rng.integers(len(types)))] if i_in_field is None else types[i_in_field % len(types)]
        tile = CT.sample_tile(rng, t, float(rng.uniform(0.15, 1.0)))
        for D in tile["dressing"]:  # branch-and-restore needs a static world (loose pebbles are not
            if D.get("kind") == "rocks":  # restored by the state reset) -> rocks static (documented)
                D["dynamic"] = False
        return dict(type=t, slippery=tile["bucket"] == "slippery", difficulty=tile["difficulty"], tile=tile), 2.5
    if family == "null":
        if rng.random() < 0.65 or lane.get("clr_haz", 0) < 1.9:
            return dict(decoy="none"), s0
        return dict(
            decoy="beside",
            kind=str(rng.choice(CLUT[pk] + WALL[pk])),
            l=r3(rng.choice([-1, 1]) * rng.uniform(1.25, 1.8)),
            yaw=r3(rng.uniform(-math.pi, math.pi)),
        ), s0
    raise KeyError(family)


def make_situation(seed, family, lane, robot, env, i_in_field=None):
    rng = np.random.default_rng([int(seed), sum(map(ord, family)), sum(map(ord, robot)), sum(map(ord, env))])
    params, s0 = sample_params(rng, family, env, lane, i_in_field)
    L = float(lane.get("len", 5.6))
    lane_d = {k: lane[k] for k in ("id", "level", "start", "yaw", "floor", "zone", "split") if k in lane} | dict(len=L)
    for k in ("kind", "clr_haz", "rise", "s_riser"):
        if k in lane:
            lane_d[k] = lane[k]
    return dict(
        sid=f"{env[:2]}_{family.replace('env_', '')[:3]}_{robot}_{seed}",
        seed=int(seed),
        family=family,
        robot=robot,
        split=env,
        env=env,
        params=params,
        s_front=r3(s0),
        lane=lane_d,
        lighting=FA.sample_lighting(rng, 0.70),
        d_dec=r3(
            (rng.uniform(1.0, 2.0) if robot.startswith("spot") else rng.uniform(0.8, 1.6))
            if family in ("env_overhang", "env_compound")
            # overhang in the head camera's view
            else (rng.uniform(0.6, 1.6) if robot.startswith("spot") else rng.uniform(0.4, 1.2))
        ),
        lat0=r3(rng.uniform(-0.06, 0.06)),
        yaw0=r3(rng.normal(0, 0.03)),
    )


def terrain_lane(i_in_field, seed_zone):
    """terrain tile k of a field sits at y = 14 k (catalog.field_spec spacing); the lane starts 2.5 m before the tile on
    the flat ground and runs 9 m along +x. Validation: every 5th field (by seed) is a 'val' field."""
    zone = f"terrain:field{seed_zone}"
    split = "val" if seed_zone % 5 == 4 else "train"
    return dict(
        id=f"ter_f{seed_zone}_t{i_in_field:02d}",
        level=0,
        start=[-2.5, 14.0 * i_in_field],
        yaw=0.0,
        floor=0.0,
        zone=zone,
        split=split,
        len=9.0,
        kind="terrain",
    )
