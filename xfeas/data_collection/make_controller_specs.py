#!/usr/bin/env python
"""Writes the controller spec files and the hidden settings of the 'look-alike' controller variants (same name and
spec, different behaviour), which force the model to trust the self-test rather than names.

Details:
gen_v5 controllers (plain python, seeded; extends scripts/gen4/make_ctrls_v4.py, which is not modified).
  data/gen_v5/specs/<ctrl_id>.json     model-visible spec (xfeas.ctrl_spec/1)
  data/gen_v5/wrappers/<ctrl_id>.json  hidden wrapper parameters (NOT a model input)

  * copies of every gen_v4 spec / wrapper (bases + v_*_NN), unchanged
  * g1p   PHP spec (direction classes of the release; 'back' excluded -- it walks forward); TRAINING base
  * spotf Isaac Spot flat-terrain policy spec (robots.py SPECS['spot'] ranges, no posture commands); HELD OUT
  * LOOK-ALIKES lk_<base>_NN: the spec is BYTE-IDENTICAL to the base spec except ctrl_id / virtual / binding (same command
    names, ranges, nominal), but a hidden behaviour differs (wrapper 'l5' effects, virtual5.py):
        tall_crouch    height / pitch commands barely act (h_eff = h_nom - k_h (h_nom - h), p_eff = k_p p)
        switch_glitch  switching INTO the fast command (vx >= 0.85 max) while already commanded >= 0.5 max -> 0.5-0.9 s burst
                       of max roll / lateral / yaw commands alternating every 0.2 s (falls or strongly tilts);
                       variant trigger 'posture_change': any posture / mode change while commanded vx >= 0.5 max
        lat_deadzone   |vy| shrinks by a hidden dead-zone dz (0.6-0.9 of max): vy_eff = sign(vy) max(0, |vy| - dz)
        height_lag     height / pitch follow with a first-order lag tau 1.0-2.5 s after a 0.2-0.5 s delay
  * MANY-MODE wrappers mm_<base>_NN (g1c, g1g): 10-20 named style modes (names s00..), each mode = hidden (height, pitch,
    speed gear 0.3-1.6, lateral gain / dead-zone, roll offset); some modes glitch on entry at speed. mm_<b>_04 is a
    look-alike of mm_<b>_01 (identical spec, different hidden table).
SONIC (g1s) / spotf: specs only (held out).
    python3 xfeas/data_collection/make_controller_specs.py
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import copy, glob, json, os, shutil, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
import controller_registry as K5
import ctrl_spec as CS
import virtual as VC

SEED = 20261022
os.makedirs(K5.SPEC_D5, exist_ok=True)
os.makedirs(K5.WRAP_D5, exist_ok=True)


def put(vs, w, base):
    CS.validate(vs)
    json.dump(vs, open(os.path.join(K5.SPEC_D5, f"{vs['ctrl_id']}.json"), "w"), indent=1)
    if w is not None:
        json.dump(
            dict(w, ctrl_id=vs["ctrl_id"], base=base, NOT_A_MODEL_INPUT=True),
            open(os.path.join(K5.WRAP_D5, f"{vs['ctrl_id']}.json"), "w"),
            indent=1,
        )


def php_spec():
    dirs = ["stand", "forward", "left_45", "left_90", "right_45", "right_90"]
    modes = [
        dict(
            name=n,
            id=i,
            group="direction",
            moving=n != "stand",
            static=n == "stand",
            speed_range=None,
            uses_height=False,
            height_range=None,
            steer=n not in ("stand", "forward"),
            doc=dict(
                stand="stand still",
                forward="walk forward",
                left_45="arc left ~45 deg/s",
                left_90="turn left ~90 deg/s",
                right_45="arc right ~45 deg/s",
                right_90="turn right ~90 deg/s",
            )[n],
        )
        for i, n in enumerate(dirs)
    ]
    return {
        "schema": "xfeas.ctrl_spec/1",
        "ctrl_id": "g1p",
        "base": "g1p",
        "virtual": False,
        "robot": "unitree_g1",
        "rate_hz": 50,
        "controller": "PHP depth student (released, amazon-far/php_parkour @c7c9ce3, D435i student)",
        "nominal": {"speed": 0.73, "mode": "forward"},
        "commands": [dict(name="direction", role="mode", type="discrete", values=dirs, nominal="stand")],
        "modes": modes,
        "binding": {
            "adapter": "g1_php:make_g1p",
            "call": "dict",
            "names": {"direction": "dir"},
            "mode_values": "name",
            "resend_every_step": True,
            "steer_by_mode": "php_quantise",
        },
        "doc_notes": [
            "release direction classes (one-hot); 'back' (class 11) walks forward and is excluded",
            "no speed / lateral / height / pitch / gait input; steering = direction class (path follower)",
            "nominal.mode = the stock moving class (forward, steered by the follower)",
        ],
        "sources": ["xfeas/sim/controllers/php/g1_php.py", "analysis/PHP_COURSE.md", "analysis/BASELINES_R1_PHP.md"],
    }


def spotf_spec():
    return {
        "schema": "xfeas.ctrl_spec/1",
        "ctrl_id": "spotf",
        "base": "spotf",
        "virtual": False,
        "robot": "bd_spot",
        "rate_hz": 50,
        "controller": "Isaac Sim 5.1 SpotFlatTerrainPolicy (isaacsim.robot.policy.examples, as shipped)",
        "nominal": {"speed": 0.9},
        "commands": [
            dict(name="vx", role="lin_vel_x", type="continuous", unit="m/s", range=[-1.0, 1.5], nominal=0.0),
            dict(name="vy", role="lin_vel_y", type="continuous", unit="m/s", range=[-0.8, 0.8], nominal=0.0),
            dict(name="wz", role="yaw_rate", type="continuous", unit="rad/s", range=[-1.5, 1.5], nominal=0.0),
        ],
        "modes": [],
        "binding": {"adapter": "spot_flat5:make_spotf", "call": "dict", "names": {"vx": "vx", "vy": "vy", "wz": "wz"}},
        "doc_notes": [
            "ranges = robots.py SPECS['spot'].max_cmd (trained command ranges); nominal speed = 0.6 x 1.5 (harness rule)",
            "HELD OUT: probe only, never data",
        ],
        "sources": ["xfeas/sim/robots.py"],
    }


def lookalike_spec(base, cid):
    s = copy.deepcopy(base)
    s.update(
        ctrl_id=cid,
        virtual=True,
        base=base["ctrl_id"],
        controller=f"virtual wrapper {cid} of {base['ctrl_id']}",
        doc_notes=[],
        sources=[],
    )
    s["binding"] = dict(adapter=base["binding"]["adapter"], call="virtual", base=base["ctrl_id"])
    return s


def l5_effect(kind, base, rng):
    has_h = CS.has(base, "pelvis_height")
    if kind == "tall_crouch":
        return dict(kind=kind, k_h=round(float(rng.uniform(0.08, 0.25)), 3), k_p=round(float(rng.uniform(0.0, 0.3)), 3))
    if kind == "fastfall":
        return dict(
            kind="switch_glitch",
            trigger="vx_up",
            vx_hi_f=0.85,
            vx_prev_f=0.5,
            T=round(float(rng.uniform(0.5, 0.9)), 3),
            period=0.2,
            amp=round(float(rng.uniform(0.85, 1.0)), 3),
        )
    if kind == "switchtilt":
        return dict(
            kind="switch_glitch",
            trigger="posture_change",
            vx_prev_f=0.5,
            T=round(float(rng.uniform(0.5, 0.9)), 3),
            period=0.2,
            amp=round(float(rng.uniform(0.85, 1.0)), 3),
        )
    if kind == "lat_deadzone":
        return dict(kind=kind, dz_f=round(float(rng.uniform(0.6, 0.9)), 3))
    if kind == "height_lag":
        return dict(kind=kind, tau=round(float(rng.uniform(1.0, 2.5)), 3), delay=round(float(rng.uniform(0.2, 0.5)), 3))
    raise KeyError(kind)


LK_PLAN = dict(
    g1c=["tall_crouch", "fastfall", "lat_deadzone", "height_lag", "switchtilt"],
    g1g=["tall_crouch", "fastfall", "lat_deadzone", "height_lag", "switchtilt"],
    spotp=["tall_crouch", "fastfall", "lat_deadzone", "height_lag", "switchtilt"],
    g1l=["fastfall"],
    g1u=["fastfall", "lat_deadzone"],
)


def many_mode(base, k, rng, names=None, table=None):
    """K style modes over a base with height (+ pitch) commands. Returns (vspec, wcfg)."""
    hlo, hhi = CS.rng(base, "pelvis_height")
    hnom = CS.cmd(base, "pelvis_height")["nominal"]
    phi = CS.rng(base, "torso_pitch")[1] if CS.has(base, "torso_pitch") else 0.0
    rlim = CS.rng(base, "torso_roll")[1] if CS.has(base, "torso_roll") else 0.0
    vmax = CS.rng(base, "lin_vel_x")[1]
    if table is None:
        K = int(rng.integers(10, 21))
        names = [f"s{i:02d}" for i in range(K)]
        rows = [dict(h=hnom, p=0.0, gear=1.0, lat_k=1.0, lat_db=0.0, roll=0.0, glitch=False)]  # the nominal walk
        for i in range(1, K):
            r = rng.random()
            if r < 0.30:  # crouch / crawl styles: low, slow
                h = float(rng.uniform(hlo, hlo + 0.6 * (hnom - hlo)))
                p = float(rng.uniform(0.2 * phi, phi))
                g = float(rng.uniform(0.3, 0.8))
            elif r < 0.50:  # run / high-energy styles: tall, fast (some glitch on entry at speed)
                h = float(rng.uniform(hlo + 0.85 * (hnom - hlo), hnom))
                p = float(rng.uniform(0.0, 0.15 * phi))
                g = float(rng.uniform(1.2, 1.6))
            elif r < 0.75:  # mid styles
                h = float(rng.uniform(hlo + 0.5 * (hnom - hlo), hnom))
                p = float(rng.uniform(0.0, 0.6 * phi))
                g = float(rng.uniform(0.6, 1.2))
            else:  # odd styles: leaning / rolled / sideways-capable
                h = float(rng.uniform(hlo + 0.4 * (hnom - hlo), hnom))
                p = float(rng.uniform(0.0, 0.8 * phi))
                g = float(rng.uniform(0.4, 1.3))
            rows.append(
                dict(
                    h=round(h, 3),
                    p=round(p, 3),
                    gear=round(g, 3),
                    lat_k=round(float(rng.choice([0.0, rng.uniform(0.3, 1.3)], p=[0.3, 0.7])), 3)
                    if rng.random() < 0.9
                    else 1.0,
                    lat_db=round(float(rng.choice([0.0, rng.uniform(0.05, 0.3)])), 3),
                    roll=round(float(rng.uniform(-0.3, 0.3) * rlim), 3) if rng.random() < 0.35 else 0.0,
                    glitch=bool(g >= 1.2 and rng.random() < 0.5),
                )
            )
        perm = rng.permutation(K)
        table = {names[perm[i]]: rows[i] for i in range(K)}
        nominal = names[perm[0]]
    else:
        nominal = table["__nominal__"]
        table = {k_: v for k_, v in table.items() if k_ != "__nominal__"}
    cid = f"mm_{base['ctrl_id']}_{k:02d}"
    s = copy.deepcopy(base)
    s.update(
        ctrl_id=cid,
        virtual=True,
        base=base["ctrl_id"],
        controller=f"virtual many-mode wrapper {cid} of {base['ctrl_id']}",
        doc_notes=[],
        sources=[],
    )
    cmds = [dict(c) for c in s["commands"] if c["role"] not in ("pelvis_height", "torso_pitch")]
    modes = []
    for i, n in enumerate(sorted(table)):
        g = table[n]["gear"]
        sr = None
        if g < 0.8:
            sr = [round(0.15 * vmax, 3), round(0.6 * vmax, 3)]
        elif g >= 1.2:
            sr = [round(0.6 * vmax, 3), round(vmax, 3)]
        modes.append(
            dict(
                name=n,
                id=i,
                group="style",
                moving=True,
                static=False,
                speed_range=sr,
                uses_height=False,
                height_range=None,
                doc="",
            )
        )
    cmds.append(dict(name="style", role="mode", type="discrete", values=sorted(table), nominal=nominal))
    s["commands"] = cmds
    s["modes"] = modes
    s["binding"] = dict(adapter=base["binding"]["adapter"], call="virtual", base=base["ctrl_id"])
    w = VC.identity_cfg()
    w.update(
        primary="many_mode",
        quant={n: dict(h=v["h"], p=v["p"]) for n, v in table.items()},
        quant_nominal=nominal,
        quant_vx={n: v["gear"] for n, v in table.items()},
        quant_lat={n: dict(k=v["lat_k"], db=v["lat_db"]) for n, v in table.items()},
    )
    gl = [n for n, v in table.items() if v["glitch"]]
    w["l5"] = [dict(kind="mode_roll", roll={n: v["roll"] for n, v in table.items() if v["roll"]})]
    if gl:
        w["l5"].append(
            dict(kind="switch_glitch", trigger="mode_in", modes=gl, vx_prev_f=0.5, T=0.7, period=0.2, amp=0.95)
        )
    return s, w, dict(table, __nominal__=nominal)


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(K5.SPEC_D4, "*.json"))):
        shutil.copy(f, os.path.join(K5.SPEC_D5, os.path.basename(f)))
    for f in sorted(glob.glob(os.path.join(K5.WRAP_D4, "*.json"))):
        shutil.copy(f, os.path.join(K5.WRAP_D5, os.path.basename(f)))
    put(php_spec(), None, "g1p")
    rows.append(("g1p", "real (PHP)"))
    put(spotf_spec(), None, "spotf")
    rows.append(("spotf", "real (Isaac Spot flat, HELD OUT)"))
    for b, plan in LK_PLAN.items():
        base = CS.load(os.path.join(K5.SPEC_D4, f"{b}.json"))
        for k, kind in enumerate(plan, 1):
            rng = np.random.default_rng([SEED, k, sum(map(ord, b))])
            cid = f"lk_{b}_{k:02d}"
            w = VC.identity_cfg()
            w["primary"] = f"lookalike_{kind}"
            w["l5"] = [l5_effect(kind, base, rng)]
            vs = lookalike_spec(base, cid)
            pub = lambda x: {
                kk: vv
                for kk, vv in x.items()
                if kk not in ("ctrl_id", "virtual", "base", "controller", "binding", "doc_notes", "sources")
            }
            assert json.dumps(pub(vs), sort_keys=True) == json.dumps(pub(base), sort_keys=True), cid
            put(vs, w, b)
            rows.append((cid, w["primary"]))
    for b in ("g1c", "g1g"):
        base = CS.load(os.path.join(K5.SPEC_D4, f"{b}.json"))
        tab1 = None
        for k in (1, 2, 3):
            rng = np.random.default_rng([SEED, 500 + k, sum(map(ord, b))])
            vs, w, tab = many_mode(base, k, rng)
            if k == 1:
                tab1 = tab
            put(vs, w, b)
            rows.append((vs["ctrl_id"], f"many_mode K={len(vs['modes'])}"))
        # mm_<b>_04: same spec as mm_<b>_01 (names, nominal), hidden table re-assigned: the mode rows are permuted among the
        # names and the fastest non-nominal mode glitches on entry
        rng = np.random.default_rng([SEED, 504, sum(map(ord, b))])
        nom = tab1["__nominal__"]
        names = sorted(k_ for k_ in tab1 if k_ != "__nominal__")
        others = [n for n in names if n != nom]
        perm = list(rng.permutation(others))
        t4 = {nom: dict(tab1[nom])}
        for a_, b_ in zip(others, perm):
            t4[a_] = dict(tab1[b_])
        fast = max(others, key=lambda n: t4[n]["gear"])
        t4[fast]["glitch"] = True
        _vs, w, _ = many_mode(base, 4, rng, table=dict(t4, __nominal__=nom))
        s1 = json.load(open(os.path.join(K5.SPEC_D5, f"mm_{b}_01.json")))
        vs = copy.deepcopy(s1)
        vs.update(ctrl_id=f"mm_{b}_04", controller=f"virtual many-mode wrapper mm_{b}_04 of {b}")
        pub = lambda x: {kk: vv for kk, vv in x.items() if kk not in ("ctrl_id", "controller")}
        assert json.dumps(pub(vs), sort_keys=True) == json.dumps(pub(s1), sort_keys=True), "mm_04 spec must equal mm_01"
        w["primary"] = "many_mode_lookalike_of_01"
        put(vs, w, b)
        rows.append((vs["ctrl_id"], "many_mode look-alike of _01"))
    print(len(rows), "new gen_v5 controllers (+ gen_v4 copies)")
    for r in rows:
        print(" ", *r)


# ====================================================================== gen5 wheeled / aerial bodies (2026-10-25, additive)
# One off-the-shelf controller each (GEN5_DESIGN 10a): the Jackal's shipped diff-drive controller and Pegasus' Iris
# geometric controller as a velocity tracker (xfeas/sim/robots.py; adapters xfeas/sim/controllers/xbody5.py). Ranges =
# robots.py SPECS max_cmd (the ranges each controller accepts); nominal speed 0.8 m/s (indoor cruise; the 0.6 x vx_max harness rule = 1.2 m/s would use up the warm-up on the short school lanes).
# Iris altitude: role pelvis_height (the posture-like height command of the spec format), name 'alt', metres above the
# floor; nominal = the stock cruise altitude 1.5 m = top of the range (like a humanoid's upright height): flying lower is
# the 'crouch' (under overhangs / desks), the stock altitude flies over low clutter.
#     python3 xfeas/data_collection/make_controller_specs.py --xbody          (writes ONLY jackal / iris specs + wrappers; nothing else touched)
def jackal_spec():
    return {
        "schema": "xfeas.ctrl_spec/1",
        "ctrl_id": "jackal",
        "base": "jackal",
        "virtual": False,
        "robot": "clearpath_jackal",
        "rate_hz": 50,
        "controller": "Clearpath Jackal skid-steer base: Isaac Sim DifferentialController + yaw-rate integral (robots.py, as shipped)",
        "nominal": {"speed": 0.8},
        "commands": [
            dict(name="vx", role="lin_vel_x", type="continuous", unit="m/s", range=[-1.0, 2.0], nominal=0.0),
            dict(name="wz", role="yaw_rate", type="continuous", unit="rad/s", range=[-2.0, 2.0], nominal=0.0),
        ],
        "modes": [],
        "binding": {"adapter": "xbody5:make_jackal5", "call": "dict", "names": {"vx": "vx", "wz": "wz"}},
        "doc_notes": [
            "ranges = robots.py SPECS['jackal'].max_cmd (2.0 fwd / 1.0 rev / 2.0 rad/s); nominal speed 0.8 m/s (indoor cruise; 0.4 x vx_max)",
            "wheeled skid-steer: no lateral, no posture; cannot climb curbs / steps above its wheel radius",
        ],
        "sources": ["xfeas/sim/robots.py", "xfeas/sim/controllers/xbody5.py"],
    }


def iris_spec():
    return {
        "schema": "xfeas.ctrl_spec/1",
        "ctrl_id": "iris",
        "base": "iris",
        "virtual": False,
        "robot": "pegasus_iris",
        "rate_hz": 50,
        "controller": "Pegasus Simulator Iris: geometric controller as a velocity tracker (robots.py) + altitude setpoint",
        "nominal": {"speed": 0.8},
        "commands": [
            dict(name="vx", role="lin_vel_x", type="continuous", unit="m/s", range=[-1.0, 2.0], nominal=0.0),
            dict(name="vy", role="lin_vel_y", type="continuous", unit="m/s", range=[-1.0, 1.0], nominal=0.0),
            dict(name="wz", role="yaw_rate", type="continuous", unit="rad/s", range=[-1.5, 1.5], nominal=0.0),
            dict(
                name="alt",
                role="pelvis_height",
                type="continuous",
                unit="m",
                range=[0.35, 1.5],
                nominal=1.5,
                doc="flight altitude of the body above the lane floor (spawn level)",
            ),
        ],
        "modes": [],
        "binding": {
            "adapter": "xbody5:make_iris5",
            "call": "dict",
            "names": {"vx": "vx", "vy": "vy", "wz": "wz", "alt": "alt"},
        },
        "doc_notes": [
            "vx / wz = robots.py SPECS['iris'].max_cmd (2.0 m/s, 1.5 rad/s); vy limited to 1.0 m/s (sidestep around hazards)",
            "alt = height setpoint above the lane floor at the spawn (0.35-1.5 m); stock cruise altitude 1.5 m; nominal speed 0.8 m/s",
        ],
        "sources": ["xfeas/sim/robots.py", "xfeas/sim/controllers/xbody5.py"],
    }


LK_PLAN_XB = dict(jackal=["fastfall"], iris=["tall_crouch", "fastfall", "lat_deadzone", "height_lag", "switchtilt"])
NV_XB = dict(jackal=7, iris=9)  # v_<base>_01..07 (sample_wrapper), iris + 08, 09 (sample_wrapper_modal: altitude modes)


def main_xbody():
    rows = []
    for sf in (jackal_spec, iris_spec):
        base = sf()
        b = base["ctrl_id"]
        put(base, None, b)
        rows.append((b, "real"))
        for k in range(1, NV_XB[b] + 1):
            vs, w = VC.sample_wrapper(base, k, SEED) if k <= 7 else VC.sample_wrapper_modal(base, k, SEED)
            put(vs, w, b)
            rows.append((vs["ctrl_id"], w["primary"]))
        for k, kind in enumerate(LK_PLAN_XB[b], 1):
            rng = np.random.default_rng([SEED, k, sum(map(ord, b))])
            cid = f"lk_{b}_{k:02d}"
            w = VC.identity_cfg()
            w["primary"] = f"lookalike_{kind}"
            w["l5"] = [l5_effect(kind, base, rng)]
            if b == "jackal" and w["l5"][0].get("trigger") == "vx_up":
                w["l5"][0]["vx_prev_f"] = (
                    0.45  # vx_max 2.0: the probe's fast speed (0.95 m/s) must be able to trigger it
                )
            vs = lookalike_spec(base, cid)
            pub = lambda x: {
                kk: vv
                for kk, vv in x.items()
                if kk not in ("ctrl_id", "virtual", "base", "controller", "binding", "doc_notes", "sources")
            }
            assert json.dumps(pub(vs), sort_keys=True) == json.dumps(pub(base), sort_keys=True), cid
            put(vs, w, b)
            rows.append((cid, w["primary"]))
    print(len(rows), "gen_v5 wheeled / aerial controllers")
    for r in rows:
        print(" ", *r)


if __name__ == "__main__":
    main_xbody() if "--xbody" in sys.argv else main()
