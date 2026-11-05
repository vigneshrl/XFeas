"""The list of controllers XFeas knows: their IDs (g1c = AMO, g1s = SONIC, spotp = ReLIC Spot, ...), which robot each
drives, how to construct each one in simulation, where their spec files live, and which are held out.

Details:
gen_v5 controller registry (pure python)..

Extends the gen_v4 registry (scripts/gen4/ctrl4.py, not modified) with
  * g1p   PHP (released depth student, scripts/php/g1_php:make_g1p) as a G1 TRAINING controller;
  * spotf the Isaac Sim Spot flat-terrain policy (robots.py kind 'spot', wrapped by spot_flat5:make_spotf) -- HELD OUT
          (probe only, never data; the second zero-shot test,.
SONIC (g1s) was held out until 2026-10-30; since then it is a G1 TRAINING controller. Spec files: data/gen_v5/specs first (copies of gen_v4 + the new ones), then gen_v4.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os, sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
import ctrl_spec as CS

D5 = os.path.join(ROOT, "data", "gen_v5")
SPEC_D5, WRAP_D5 = os.path.join(D5, "specs"), os.path.join(D5, "wrappers")
SPEC_D4, WRAP_D4 = os.path.join(ROOT, "data", "gen_v4", "specs"), os.path.join(ROOT, "data", "gen_v4", "wrappers")
CS.SPEC_DIRS = (SPEC_D5, SPEC_D4, os.path.join(THIS, "..", "_support", "model", "specs")) + tuple(CS.SPEC_DIRS)
CS.ROLES = tuple(CS.ROLES)

FACT = dict(
    g1c="sidestep_g1:make_g1c",
    g1g="g1_gr00t:make_g1g",
    g1l="il_g1_robot:make_il_g1",
    g1u="urma_robot:make_urma_g1",
    g1p="php5:make_g1p5",
    g1s="g1_sonic:make_g1s",
    spotp="spot_posture:make_spotp",
    spotf="spot_flat5:make_spotf",
    jackal="xbody5:make_jackal5",
    iris="xbody5:make_iris5",
)
G1_BASES = ("g1c", "g1g", "g1l", "g1u", "g1p", "g1s")  # g1s added 2026-10-30 (was held out)
SPOT_BASES = ("spotp",)
XBODY_BASES = ("jackal", "iris")  # gen5 2026-10-25: Clearpath Jackal (wheeled), Pegasus Iris (aerial); xbody5.py
BASES = G1_BASES + SPOT_BASES + XBODY_BASES  # collected (training) bases
HELD_OUT = ("spotf",)  # probe only, NEVER data / fitting (g1s removed 2026-10-30)
ALL_BASES = BASES + HELD_OUT
CS.HELD_OUT = HELD_OUT
ROBOT_OF = {b: ("g1" if b.startswith("g1") else b if b in XBODY_BASES else "spot") for b in ALL_BASES}
PROBES_V4 = os.path.join(ROOT, "data", "probes_v4")
PROBES_V5 = os.path.join(ROOT, "data", "probes_v5")


def base_of(ctrl_id):
    c = str(ctrl_id)
    if c.startswith(("v_", "lk_", "mm_")):
        return c.split("_")[1]
    return c


def robot_of(ctrl_id):
    b = base_of(ctrl_id)
    return ROBOT_OF.get(b, "g1" if b.startswith("g1") else b if b in XBODY_BASES else "spot")


def is_held_out(ctrl_id):
    return base_of(ctrl_id) in HELD_OUT or ctrl_id in HELD_OUT


def spec_path(cid):
    for d in (SPEC_D5, SPEC_D4):
        p = os.path.join(d, f"{cid}.json")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(cid)


def wrapper_path(cid):
    for d in (WRAP_D5, WRAP_D4):
        p = os.path.join(d, f"{cid}.json")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(cid)


def load_ctrl(cid):
    """-> (spec, wrapper cfg or None for a real controller)"""
    import json

    s = CS.load(spec_path(cid))
    if not s.get("virtual"):
        return s, None
    w = json.load(open(wrapper_path(cid)))
    return s, {k: v for k, v in w.items() if k not in ("ctrl_id", "base", "NOT_A_MODEL_INPUT")}


def env_for(base):
    return {"IL_G1_ASSET": "full"} if base == "g1l" else ({"SONIC_PLANNER_ASYNC": "0"} if base == "g1s" else {})
