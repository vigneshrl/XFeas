#!/usr/bin/env python
"""data/probes_v5/MANIFEST.json: every controller with a gen5 probe pair (office_v4.json + transitions.json), sha1s,
source of the probe_v4 part (copied from data/probes_v4 or run by gen5), transition summary, held-out flag."""
import glob, hashlib, json, os, sys

THIS = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, THIS)
import ctrl5 as K5

out = dict(protocol="probe_v4 + transitions_v5 (scripts/gen5/probe_transitions5.py)", root="data/probes_v5", probes={})
for d in sorted(glob.glob(os.path.join(K5.PROBES_V5, "*"))):
    cid = os.path.basename(d)
    fp, ft = os.path.join(d, "office_v4.json"), os.path.join(d, "transitions.json")
    if not (os.path.exists(fp) and os.path.exists(ft)):
        continue
    P, T = json.load(open(fp)), json.load(open(ft))
    sha = lambda f: hashlib.sha1(open(f, "rb").read()).hexdigest()
    src4 = os.path.join(K5.PROBES_V4, cid, "office_v4.json")
    nfell = sum(1 for c in T["classes"].values() for k in ("to", "back") for r in c[k].values() if r and r.get("fell"))
    ngl = sum(1 for c in T["classes"].values() for r in c["to"].values() if r and r.get("n_glitch"))
    out["probes"][cid] = dict(base=K5.base_of(cid), held_out=K5.is_held_out(cid), probe=os.path.relpath(fp, K5.ROOT), probe_sha1=sha(fp),
                              probe_src="probes_v4_copy" if os.path.exists(src4) and sha(src4) == sha(fp) else "gen5_run",
                              transitions=os.path.relpath(ft, K5.ROOT), transitions_sha1=sha(ft), n_classes=len(T["classes"]),
                              speeds=T["speeds"], t_robot=T["t_robot"], t_wall=T.get("t_wall"), n_fell_records=nfell, n_glitch_to=ngl,
                              fingerprint=P.get("fingerprint"), fp_keys=P.get("fp_keys"))
json.dump(out, open(os.path.join(K5.PROBES_V5, "MANIFEST.json"), "w"), indent=1)
print(f"[manifest5] {len(out['probes'])} controllers -> data/probes_v5/MANIFEST.json "
      f"(held out: {[c for c, v in out['probes'].items() if v['held_out']]})")
