#!/usr/bin/env python
"""
Sidecar for the closed-loop gen1 arms (gen_methods.py): select_gen.Selector + an EXPLICIT utility file, same framing
as select_gen.py --serve / select_xe.py --serve (4-byte length + npz; reply npz with 'choice' json).

Why a separate entry point: select_gen.Selector reads utility['chosen'] only; utility_posture.json is a FLAT dict
(c, lam, kappa, w, pdev) without 'chosen', so Selector would silently fall back to the success-only utility. Here the
utility is loaded from either layout and checked (fail loudly if neither).

    $XFEAS_PY closed_loop/gen_serve.py --ckpt data/xe_train/gen1/xfeas_gen_ens.pt \
        --utility data/xe_train/gen1/utility_posture.json
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import argparse, json, os, sys

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import select_gen as SG


def load_utility(path):
    U = json.load(open(path))
    util = U.get("chosen") if isinstance(U.get("chosen"), dict) else (U if all(k in U for k in ("c", "lam", "kappa", "w")) else None)
    if util is None:
        raise SystemExit(f"[gen_serve] {path}: no utility found (neither 'chosen' nor flat c/lam/kappa/w)")
    util = {k: util[k] for k in ("c", "lam", "kappa", "w", "pdev") if k in util}
    return util


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--utility", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    sel = SG.Selector(a.ckpt, a.device, utility=a.utility)
    sel.util = load_utility(a.utility)
    sel.util_status = f"explicit:{os.path.basename(a.utility)}"
    print(f"[gen_serve] {os.path.basename(a.ckpt)} utility {os.path.basename(a.utility)} = {json.dumps(sel.util)}", file=sys.stderr, flush=True)
    SG.serve(sel)
