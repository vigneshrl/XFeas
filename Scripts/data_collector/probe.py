"""
Self-test PROBE: plan (from the spec only) + summary (ops table, caps, fingerprint). analysis/GEN3_SPEC.md section 4.
Pure python; the Isaac execution of the plan is gensim3.V3Mixin.run_probe.

plan_segments(spec)          -> ordered list of segment dicts (kind, cmd, v, vy, wz, dur_min, dur_max, ...)
summarize(spec, tests, meta) -> dict(ops, caps, fingerprint, fp_keys)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import ctrl_spec as CS

BUDGET_S = 30.0
FP_KEYS = ["stand_top", "stand_width", "stand_depth", "vx_ach_lo", "vx_ach_mid", "vx_ach_hi", "vx_cmd_lo", "vx_cmd_mid", "vx_cmd_hi",
           "lag_tau", "vy_ach_L", "vy_ach_R", "lateral_needs_vx", "wz_ach", "turn_needs_vx", "min_top_walk", "v_at_min_top_walk",
           "width_at_min_top_walk", "depth_at_min_top_walk", "min_top_any", "min_width_walk", "min_depth_walk", "max_pitch_walk",
           "n_ops_walk", "n_top_levels", "posture_discrete", "has_height_cmd", "has_pitch_cmd", "has_lateral_cmd", "tilt_max_probe",
           "n_falls_probe", "top_span_walk"]
WALK_V = 0.05          # an op "moves" when its achieved forward speed exceeds this


def _name(spec, role):
    c = CS.cmd(spec, role)
    return None if c is None else c["name"]


def posture_points(spec):
    """list of (label, posture cmd dict, v_cmd) for the posture / mode sweep"""
    pts = []
    v_nom = float(spec.get("nominal", {}).get("speed", 0.3))
    vxr = CS.rng(spec, "lin_vel_x")
    if CS.has(spec, "mode"):
        mc = CS.cmd(spec, "mode")
        hn = _name(spec, "pelvis_height")
        for m in CS.moving_modes_round_robin(spec):
            if m["name"] == mc["nominal"]:
                continue                          # the nominal mode is measured by the stand / speed segments
            sr = m.get("speed_range")
            # the doc speed band's mid-point, never above 1 m/s unless the band starts above it (safety on a short lane)
            v = float(np.clip(0.5 * (sr[0] + sr[1]), sr[0], max(sr[0], 1.0))) if sr else v_nom
            cmd = {mc["name"]: m["name"]}
            if hn and m.get("uses_height") and m.get("height_range"):
                cmd[hn] = float(m["height_range"][1])         # docs: the set's height defaults to its top value on entry
            pts.append((f"mode:{m['name']}", cmd, float(v)))
        return pts
    hc = CS.cmd(spec, "pelvis_height")
    if hc is None:
        return pts
    lo, hi = hc["range"]; nom = float(hc["nominal"])
    hs = [lo, lo + (nom - lo) / 3.0, lo + 2.0 * (nom - lo) / 3.0]
    pc = CS.cmd(spec, "torso_pitch")
    ps = [0.0] if pc is None else [0.0, 0.5 * pc["range"][1], pc["range"][1]]
    for h in hs:
        for p in ps:
            cmd = {hc["name"]: round(float(h), 3)}
            if pc is not None:
                cmd[pc["name"]] = round(float(p), 3)
            pts.append((f"h{h:.2f}_p{p:.2f}", cmd, v_nom))
    # order for a budget cut: start at the deepest point, then farthest-point order in normalised (h, p) space, so a cut
    # removes the densest part of the grid and never a whole region
    def key(x):
        v = list(x[1].values())
        return np.array([(v[0] - lo) / max(nom - lo, 1e-6), (v[1] / max(pc["range"][1], 1e-6)) if len(v) > 1 else 0.0])
    rest = list(pts); out = [min(rest, key=lambda x: (key(x)[0], -key(x)[1]))]; rest.remove(out[0])
    while rest:
        nxt = max(rest, key=lambda x: min(float(np.linalg.norm(key(x) - key(y))) for y in out))
        out.append(nxt); rest.remove(nxt)
    return out


def plan_segments(spec):
    S = []
    vxr = CS.rng(spec, "lin_vel_x")
    S.append(dict(kind="stand", label="stand", cmd={}, v=0.0, vy=0.0, wz=0.0, dur_min=1.0, dur_max=1.0))
    if CS.has(spec, "lin_vel_y"):
        lo, hi = CS.rng(spec, "lin_vel_y")
        vy = min(0.8 * max(abs(lo), abs(hi)), 0.5)
        S.append(dict(kind="lateral", label="lat_L", cmd={}, v=0.0, vy=+vy, wz=0.0, dur_min=1.2, dur_max=1.2, retry_vx=0.1))
        S.append(dict(kind="lateral", label="lat_R", cmd={}, v=0.0, vy=-vy, wz=0.0, dur_min=1.2, dur_max=1.2, retry_vx=0.1))
    hi = min(vxr[1], 1.0) if vxr is not None else 0.5
    rvx = round(max(0.1, 0.35 * hi), 3)          # forward speed for the turn / lateral retries = the speed sweep's low command
    for s_ in S:
        if "retry_vx" in s_:
            s_["retry_vx"] = rvx
    for f, lab in ((0.35, "lo"), (0.65, "mid"), (0.95, "hi")):
        S.append(dict(kind="speed", label=f"speed_{lab}", cmd={}, v=round(f * hi, 3), vy=0.0, wz=None, dur_min=1.5, dur_max=1.5))
    wr = CS.rng(spec, "yaw_rate")
    S.append(dict(kind="turn", label="turn", cmd={}, v=0.0, vy=0.0, wz=0.8 * (wr[1] if wr else 1.0), dur_min=1.2, dur_max=1.2,
                  retry_vx=rvx, then_reverse=True))
    for lab, cmd, v in posture_points(spec):
        S.append(dict(kind="posture" if not lab.startswith("mode:") else "mode", label=lab, cmd=cmd, v=v, vy=0.0, wz=None,
                      dur_min=1.2 if lab.startswith("mode:") else 1.4, dur_max=2.0))
    return S


def _f(x, d=np.nan):
    return float(x) if x is not None and np.isfinite(x) else d


def summarize(spec, tests, meta=None):
    """tests: list of executed segments with 'ach' dicts (gensim3.run_probe). -> ops, caps, fingerprint"""
    ops = []
    stand = next((t for t in tests if t["kind"] == "stand" and not t["ach"].get("fell")), None)
    for t in tests:
        if t["kind"] not in ("stand", "speed", "posture", "mode", "lateral") or t.get("skipped"):
            continue
        a = t["ach"]
        vcmd = float(t.get("v_used", t["v"]))
        ratio = a["vx"] / vcmd if vcmd > 1e-3 else np.nan
        ops.append(dict(op_id=len(ops), kind=t["kind"], label=t["label"], cmd=t["cmd"], vx_cmd=vcmd, vy_cmd=float(t.get("vy_used", t["vy"])),
                        moving=bool(a["vx"] > WALK_V) if t["kind"] != "lateral" else bool(abs(a["vy"]) > WALK_V),
                        vx_ach=round(a["vx"], 3), ratio=None if not np.isfinite(ratio) else round(float(ratio), 3), vy_ach=round(a["vy"], 3),
                        top_min=round(a["top_min"], 3), top_p50=round(a["top_p50"], 3), width=round(a["width"], 3), depth=round(a["depth"], 3),
                        pitch_ach=round(a["pitch_t"], 3), tilt_max=round(a["tilt_max"], 3), fell=bool(a["fell"]), settled=bool(a.get("settled", True))))
    walk = [o for o in ops if o["kind"] in ("speed", "posture", "mode") and o["moving"] and not o["fell"]]
    sp = {t["label"]: t for t in tests if t["kind"] == "speed"}
    lat = {t["label"]: t for t in tests if t["kind"] == "lateral"}
    tr = next((t for t in tests if t["kind"] == "turn"), None)
    caps = dict(
        turn_in_place=bool(tr is not None and tr.get("ach") and not tr.get("needs_vx") and abs(tr["ach"]["wz"]) > 0.3 * abs(tr["wz"])),
        turn_needs_vx=bool(tr is not None and tr.get("needs_vx", False)),
        turn_ok=bool(tr is not None and tr.get("ach") and abs(tr["ach"]["wz"]) > 0.3 * abs(tr["wz"])),
        wz_ach=abs(_f(tr["ach"]["wz"] if tr else None, 0.0)),
        lateral_from_stand=bool(any(abs(t["ach"]["vy"]) > 0.3 * abs(t["vy"]) and not t.get("needs_vx") for t in lat.values())),
        lateral_needs_vx=bool(any(t.get("needs_vx") for t in lat.values())),
        vy_max_L=_f(lat["lat_L"]["ach"]["vy"] if "lat_L" in lat else None, 0.0),
        vy_max_R=_f(lat["lat_R"]["ach"]["vy"] if "lat_R" in lat else None, 0.0),
        vx_max_ach=max([o["vx_ach"] for o in walk], default=0.0), vx_min_ach=min([o["vx_ach"] for o in walk], default=0.0),
        turn_vx=float(tr.get("v_used", 0.0)) if tr is not None and tr.get("needs_vx") else 0.0,
        lat_vx=float(max([t.get("v_used", 0.0) for t in lat.values() if t.get("needs_vx")], default=0.0)),
        lag_tau=_f(np.nanmedian([t["ach"].get("lag63", np.nan) for t in sp.values()]) if sp else None, np.nan))
    caps["lateral_ok"] = bool(len(lat) == 2 and all(abs(t["ach"]["vy"]) > max(0.06, 0.3 * abs(t["vy"])) and not t["ach"].get("fell")
                                                 for t in lat.values()))
    st = stand["ach"] if stand else dict(top_p50=np.nan, width=np.nan, depth=np.nan)
    mtw = min(walk, key=lambda o: o["top_min"]) if walk else None
    tops = sorted(o["top_min"] for o in walk)
    lev = 0; last = -9.0
    for x in tops:
        if x - last > 0.03:
            lev += 1; last = x
    g = lambda k: sp.get(k, {}).get("ach", {}).get("vx", np.nan)
    gc = lambda k: sp.get(k, {}).get("v", np.nan)
    fp = dict(stand_top=st["top_p50"], stand_width=st["width"], stand_depth=st["depth"],
              vx_ach_lo=g("speed_lo"), vx_ach_mid=g("speed_mid"), vx_ach_hi=g("speed_hi"),
              vx_cmd_lo=gc("speed_lo"), vx_cmd_mid=gc("speed_mid"), vx_cmd_hi=gc("speed_hi"), lag_tau=caps["lag_tau"],
              vy_ach_L=caps["vy_max_L"], vy_ach_R=caps["vy_max_R"], lateral_needs_vx=float(caps["lateral_needs_vx"]),
              wz_ach=caps["wz_ach"], turn_needs_vx=float(caps["turn_needs_vx"]),
              min_top_walk=mtw["top_min"] if mtw else st["top_p50"], v_at_min_top_walk=mtw["vx_ach"] if mtw else 0.0,
              width_at_min_top_walk=mtw["width"] if mtw else st["width"], depth_at_min_top_walk=mtw["depth"] if mtw else st["depth"],
              min_top_any=min([o["top_min"] for o in ops if not o["fell"]], default=np.nan),
              min_width_walk=min([o["width"] for o in walk], default=np.nan), min_depth_walk=min([o["depth"] for o in walk], default=np.nan),
              max_pitch_walk=max([o["pitch_ach"] for o in walk], default=0.0), n_ops_walk=float(len(walk)), n_top_levels=float(lev),
              posture_discrete=float(CS.has(spec, "mode")), has_height_cmd=float(CS.has(spec, "pelvis_height")),
              has_pitch_cmd=float(CS.has(spec, "torso_pitch")), has_lateral_cmd=float(CS.has(spec, "lin_vel_y")),
              tilt_max_probe=max([t["ach"]["tilt_max"] for t in tests if t.get("ach")], default=0.0),
              n_falls_probe=float(sum(bool(t["ach"].get("fell")) for t in tests if t.get("ach"))),
              top_span_walk=(st["top_p50"] - mtw["top_min"]) if mtw else 0.0)
    vec = np.array([_f(fp[k], 0.0) for k in FP_KEYS], np.float32)
    return dict(ops=ops, caps=caps, fingerprint=[round(float(x), 4) for x in vec], fp_keys=FP_KEYS, fp=fp)


if __name__ == "__main__":
    for k in ("g1c", "g1g", "g1l", "g1u", "g1s"):
        s = CS.load(k)
        P = plan_segments(s)
        print(k, len(P), round(sum(p["dur_min"] for p in P), 1), [p["label"] for p in P])
