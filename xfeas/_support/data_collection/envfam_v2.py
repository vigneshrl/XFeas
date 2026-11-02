"""
gen_v2 situation SPECS (pure python, seedable). analysis/GEN_V2_SPEC.md section 7. Builds on envfam.py (Isaac envs) and
families.py (school); gen_v1 builders are reused (envsim.EnvSim.build / gensim.GenSim.build), new families map onto them:

  env_lowclutter   (new, bucket scene)  1-4 LOW env objects (0.10-0.33 m) in a compact cluster ON the lane, lanes with
                   lateral room (clr_haz >= 1.1 m) so that a route offset (detour) can pass  -> built by the env_clutter builder
  school lowclutter (new)  a short real plank-stack block (0.5-1.0 m wide, 0.12-0.35 m high) on the lane, off-centre
                   -> built by the school step/curb builder (narrow width)
  env_gap          widths stratified on the robot's depth-width band (G1 0.28-0.49 m: sidestep) 70 %
  terrain          50 % slippery tiles (slip label), lane starts 5 m before the tile
  null twins       for 30 % of overhang situations, a NULL situation on the same lane / position / lighting / decision
                   distance (breaks "this room has overhangs")
Decision distance d_dec: 30 % U(1.0, 2.0), 70 % U(2.0, 3.0) m before the front edge; s_front = d_dec + run-up.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
import envfam as EF
import families as FA

r3 = FA.r3
LOWCLUT = dict(
    office=["box", "boxa", "boxbig", "boxport", "book", "bottle"],
    warehouse=["cardbox_d", "cardbox_c", "cardbox_d2", "bottle", "cardbox_c2"],
)
G1_BAND = (0.28, 0.49)
SPOT_BAND = (0.40, 0.62)
RUNUP = dict(spotp=(1.3, 2.0), g1c=(1.1, 1.6))
FAMS_V2 = dict(
    office=(
        ["env_overhang", "env_climb", "env_compound", "env_gap", "env_lowclutter", "env_clutter", "env_step", "null"],
        [0.20, 0.13, 0.08, 0.12, 0.20, 0.05, 0.04, 0.18],
    ),
    warehouse_shelves=(
        ["env_overhang", "env_climb", "env_compound", "env_gap", "env_lowclutter", "env_clutter", "env_step", "null"],
        [0.20, 0.13, 0.08, 0.12, 0.20, 0.05, 0.04, 0.18],
    ),
    full_warehouse=(
        ["env_overhang", "env_climb", "env_compound", "env_gap", "env_lowclutter", "env_clutter", "env_step", "null"],
        [0.20, 0.13, 0.08, 0.12, 0.20, 0.05, 0.04, 0.18],
    ),
    rivermark=(["env_curb", "null"], [0.65, 0.35]),
    simple_room=(["null"], [1.0]),
    dt_warehouse=(["null"], [1.0]),
    # school lowclutter dropped 2026-10-14 (smoke): the real plank-pile builder makes every 'curb' 2.55-3.45 m wide (full
    # corridor), so no route offset exists; the school has no small low real asset in the RealPool besides books
    school=(["overhang", "step", "gap", "compound", "null"], [0.32, 0.24, 0.12, 0.12, 0.20]),
    terrain=(["terrain"], [1.0]),
)


def sample_d_dec(rng):
    return r3(rng.uniform(1.0, 2.0) if rng.random() < 0.3 else rng.uniform(2.0, 3.0))


def lane_ok_v2(family, lane, env):
    if family == "env_lowclutter":
        return EF.lane_ok("env_clutter", lane, env) and lane.get("clr_haz", 0) >= 1.1
    if family == "null_wide":
        return EF.lane_ok("env_overhang", lane, env)
    return EF.lane_ok(family, lane, env)


def _lowclutter_params(rng, env):
    pk = EF.pack(env)
    n = int(rng.integers(1, 5))
    c = rng.uniform(-0.25, 0.25)
    items = []
    for i in range(n):
        items.append(
            dict(
                kind=str(rng.choice(LOWCLUT[pk])),
                ds=r3(rng.uniform(0.0, 0.8)),
                l=r3(c + rng.uniform(-0.2, 0.2)),
                yaw=r3(rng.uniform(-math.pi, math.pi)),
                scale=r3(rng.uniform(0.9, 1.1)),
            )
        )
    return dict(items=items, cluster_c=r3(c))


def make_situation_v2(seed, family, lane, robot, env, twin_of=None):
    """-> spec dict (gen_v1 schema + v2 keys). family may be a v2 family name."""
    rng = np.random.default_rng([int(seed), sum(map(ord, family)), sum(map(ord, robot)), sum(map(ord, env)), 2])
    d_dec = sample_d_dec(rng)
    ru = rng.uniform(*RUNUP.get(robot, (1.3, 2.0)))
    L = float(lane.get("len", 8.0))
    if env == "school":
        base_f = {"lowclutter": "step"}.get(family, family)
        x = FA.make_situation(int(seed), base_f, lane, robot, "school")
        x["sid"] = f"sc_{family[:3]}_{robot}_{seed}"
        if family == "lowclutter":
            x["params"].update(
                kind="curb",
                h=r3(rng.uniform(0.12, 0.35)),
                width=r3(rng.uniform(0.5, 1.0)),
                lc=r3(rng.choice([-1, 1]) * rng.uniform(0.05, 0.25)),
                rot=r3(rng.uniform(-0.3, 0.3)) if rng.random() < 0.4 else 0.0,
            )
        if family == "gap":
            band = G1_BAND if robot.startswith("g1") else SPOT_BAND
            if rng.random() < 0.7:
                x["params"]["w"] = r3(rng.uniform(*band))
        if family == "step" and rng.random() < 0.85:  # climb band
            P = x["params"]
            if P["kind"] in ("ledge", "curb", "steps"):
                P["h"] = r3(rng.uniform(0.07, 0.17))
    elif family == "terrain":
        raise ValueError("terrain situations: use make_terrain_v2")
    else:
        base_f = {"env_lowclutter": "env_clutter"}.get(family, family)
        x = EF.make_situation(int(seed), base_f if base_f != "env_lowclutter" else "env_clutter", lane, robot, env)
        x["sid"] = f"{env[:2]}_{family.replace('env_', '')[:3]}_{robot}_{seed}"
        if family == "env_lowclutter":
            x["params"] = _lowclutter_params(rng, env)
            x["family_build"] = "env_clutter"
        if family == "env_gap":
            band = G1_BAND if robot.startswith("g1") else SPOT_BAND
            if rng.random() < 0.7:
                x["params"]["w"] = r3(rng.uniform(*band))
    x["family"] = family
    x["d_dec"] = d_dec
    if family == "env_curb":
        x["s_front"] = r3(float(lane["s_riser"]))
        x["d_dec"] = r3(min(d_dec, float(lane["s_riser"]) - ru))
    else:
        cap = min(5.0 if family not in ("null",) else 6.0, L - 3.0)  # room for the hazard + goal on the lane
        x["s_front"] = r3(min(d_dec + ru, cap))
        x["d_dec"] = r3(min(d_dec, x["s_front"] - RUNUP.get(robot, (1.3, 2.0))[0]))
    x["lighting"] = FA.sample_lighting(rng, 0.70)
    x["gen"] = "v2"
    if twin_of is not None:
        x["twin_of"] = twin_of
    return x


def null_twin(sit, seed):
    """a NULL situation at exactly the overhang situation's lane / position / lighting / decision distance"""
    t = dict(sit)
    t["params"] = dict(decoy="none")
    t["family"] = "null"
    t["twin_of"] = sit["sid"]
    t["seed"] = int(seed)
    t["sid"] = sit["sid"].split("_")[0] + f"_nul_{sit['robot']}_{seed}"
    t.pop("family_build", None)
    return t


def make_terrain_v2(seed, robot, k, field):
    lane = EF.terrain_lane(k, field)
    lane = dict(lane, start=[-5.0, lane["start"][1]], len=11.5, id=lane["id"] + "v2")
    x = EF.make_situation(int(seed), "terrain", lane, robot, "terrain", i_in_field=k)
    rng = np.random.default_rng([int(seed), 77])
    x["s_front"] = 5.0
    x["d_dec"] = sample_d_dec(rng)
    x["gen"] = "v2"
    return x


def bucket_of(sit):
    fam = sit["family"]
    if fam in ("env_lowclutter", "lowclutter"):
        return "scene"
    import gensim_v1_buckets as GB

    return GB.bucket_of(sit)


def subfamily(sit):
    fam = sit["family"]
    if fam in ("env_lowclutter", "lowclutter"):
        return "lowclutter_detour"
    if fam in ("env_gap", "gap"):
        w = float(sit["params"].get("w", 1.0))
        band = G1_BAND if sit["robot"].startswith("g1") else SPOT_BAND
        return "gap_band" if band[0] <= w <= band[1] else "gap_other"
    import gensim_v1_buckets as GB

    s = GB.subfamily(sit)
    if fam == "null" and sit.get("twin_of"):
        return "null_twin"
    return s
