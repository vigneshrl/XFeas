#!/usr/bin/env python
"""
gen4 SIDE-BY-SIDE data in gen5 form (2026-10-26; analysis/GEN4_VS_GEN5_STATUS.md 6, option 1).

gen4 replayed each situation (one decision snapshot, 1.1-2.9 m before the hazard) with ~7 options. Most options are ONE
controller command held from the snapshot (macro 'constant'; 74 % of runs); the command actually sent is logged at 10 Hz in
the canonical roles (cmd_trace3, rows start at the decision). This module turns every such run into a gen5 CANDIDATE:
  command  = per-role median of cmd_trace3 over the first 2 s (yaw_rate dropped: steering stays with the path follower)
  describe = describe5.Describer on the gen4 controller's own spec + probes_v4 (no transitions file -> t_missing = 1)
  labels   = the run's outcome: success (goal reached), fell, collision (hazard body contact), stall, progress (m)
so one ITEM = one situation x controller: the gen4 context (dataset_gen4.build_context on the snapshot obs, UNCHANGED) + K
candidates (K x 35) each with REAL outcome labels. Multi-step options (detour / two_phase / turn_side_turn) are left out
(they are not one held command). No driver output is used anywhere: only what was sent to the controller.

    PY=/p/cral/vignesh/envs/gsam2/bin/python
    $PY scripts/gen5/dataset5g4.py build --out data/gen_train/gen5g4/manifest.jsonl
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import canon5                                                # gen5 canonical joints (before any gen4 import)
for _p in (os.path.join(THIS, "..", "gen4"), os.path.join(THIS, "..", "gen_train"), os.path.join(THIS, "..", "xembody")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import dataset_gen4 as DG
import candidates5 as CA
import describe5 as DS
import ctrl_spec as CS

ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
GEN4_MANIFEST = os.path.join(ROOT, "data", "gen_train", "gen4", "manifest.jsonl")
CMD_ROLES = ("lin_vel_x", "lin_vel_y", "yaw_rate", "pelvis_height", "torso_pitch", "torso_roll", "mode", "speed")   # cmd_trace3 columns
KEEP_MACROS = ("constant",)
LABELS = ("success", "fell", "collision", "stall", "progress")
HOLD_ROWS = 20                                              # 2 s at 10 Hz


def held_command(z, spec):
    """the controller command held from the decision (canonical roles present in the spec), or None"""
    c = np.asarray(z["cmd_trace3"], np.float64)[:HOLD_ROWS]
    if len(c) == 0:
        return None
    cmd = {}
    for j, role in enumerate(CMD_ROLES):
        if role == "yaw_rate" or not CS.has(spec, role):
            continue
        col = c[:, j]
        col = col[np.isfinite(col)]
        if not len(col):
            continue
        v = float(np.median(col))
        if role == "mode":
            vals = CS.cmd(spec, "mode")["values"]
            i = int(round(v))
            if not 0 <= i < len(vals):
                return None
            cmd["mode"] = vals[i]
        else:
            cmd[role] = round(v, 4)
    return cmd or None


def build(gen4_manifest=GEN4_MANIFEST):
    rows = [json.loads(l) for l in open(gen4_manifest)]
    out, drop = [], defaultdict(int)
    desc_cache = {}
    for r in rows:
        spec = json.load(open(r["spec"])) if isinstance(r["spec"], str) else r["spec"]
        probe = json.load(open(r["probe"]))
        key = (r["ctrl_id"], r["probe"])
        if key not in desc_cache:
            desc_cache[key] = DS.Describer(spec, probe, None)
        D = desc_cache[key]
        nom = CA.nominal_cmd(spec)
        cands, labs, meta = [], [], []
        for ep in r["eps"]:
            z = np.load(ep, allow_pickle=False)
            macro = json.loads(str(z["exec"]))["macro"]
            kind = str(z["kind"])
            if macro not in KEEP_MACROS:
                drop[f"macro_{macro}"] += 1; continue
            if int(z["dup_of"]) >= 0:
                drop["dup"] += 1; continue
            cmd = held_command(z, spec)
            if cmd is None:
                drop["no_cmd"] += 1; continue
            c = dict(kind="NOMINAL" if kind == "stock" else "grid", cmd=cmd)
            c["cls"] = CA.transition_class(spec, cmd)
            cands.append(c)
            labs.append([float(z["success"]), float(z["fell"]), float(z["collision"]), float(z["stall"]), float(z["progress"])])
            meta.append(dict(ep=ep, kind=kind, cmd=cmd))
        if len(cands) < 2:
            drop["lt2_options"] += 1; continue
        X = D.describe(cands, nom, float(spec.get("nominal", {}).get("speed", 0.3)))
        out.append(dict(key=r["key"], obs=r["obs"], ctrl_id=r["ctrl_id"], base=r["base"], robot=r["robot"], env=r["env"], split=r["split"],
                        family=r["family"], subfamily=r.get("subfamily", ""), light=r.get("light", ""), spec=r["spec"], probe=r["probe"],
                        X=np.round(X, 5).tolist(), y=labs, opts=meta, stock=[m["kind"] == "stock" for m in meta]))
    return out, dict(drop)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("build")
    b.add_argument("--gen4-manifest", default=GEN4_MANIFEST)
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "build":
        rows, drop = build(a.gen4_manifest)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        Y = np.concatenate([np.asarray(r["y"]) for r in rows])
        print(f"[gen5g4] {len(rows)} situations, {len(Y)} candidates (median {np.median([len(r['y']) for r in rows]):.0f} / situation); "
              f"success {Y[:, 0].mean():.2f}; situations with >= 1 success {np.mean([max(x[0] for x in r['y']) for r in rows]):.2f}, "
              f"stock succeeds {np.mean([any(y[0] and s for y, s in zip(r['y'], r['stock'])) for r in rows]):.2f}; dropped {drop}")


if __name__ == "__main__":
    main()
