#!/usr/bin/env python
"""Writes the extra suites for the overhang scenes in the other environments and for the scenes with nothing to avoid.

Details:
gen5b env_overhang / null suites. Runs xfeas/_support/data_collection/make_suite5.py UNCHANGED in logic (exec of its source, same
lanes, builders, lighting top-up to night >= 0.62, controller pools) with three gen5b changes:
  * output data/gen_v5b/suites/<env>_<base>_<tag>.json (not data/gen_v5/suites), seed offset --seed-off (new sids);
  * env_overhang underside drawn per ASSIGNED controller like crawl v2: with p = --p-crawl, u = t_lo(ctrl) + U(-0.12, +0.18)
    clipped to [0.40, stand_top - 0.06] (t_lo = lowest MOVING probe top; so most need a crouch, ~half fit the lowest posture);
    otherwise the gen5 env_overhang draw (50 % p1 crawl band U(0.50, 1.00) G1 / U(0.45, 0.80) Spot, else the gen_v4 band);
    every such situation stores params.crawl_v2 = {ctrl, t_lo, stand, rel, passable_est} (Spot: below its standing top);
  * --crawl-pool restricts the controller pool to crawl-capable controllers (stand - t_lo >= 0.15), as crawl v2.
Why: every school training lane is within 3 m of the school-course test SC2A (see make_suites.py), so school crawl v2
situations cannot be used; the low-obstacle data comes from office / warehouse_shelves / full_warehouse.
    python3 xfeas/data_collection/make_env_suites.py --env office --base g1c --families env_overhang,null --weights 0.8,0.2 --n 40         --seed-off 70000 --tag b1 --crawl-pool
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os, sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
argv = list(sys.argv[1:])
p_crawl = 0.7
if "--p-crawl" in argv:
    i = argv.index("--p-crawl")
    p_crawl = float(argv[i + 1])
    del argv[i : i + 2]
SRC = os.path.join(ROOT, "xfeas", "_support", "data_collection", "make_suite5.py")
src = open(SRC).read()
rep = [
    ('OUTD = os.path.join(D5, "suites")', 'OUTD = os.path.join(ROOT, "data", "gen_v5b", "suites")'),
    (
        '        if x["family5"] == "crawl" and not a.crawl_legacy:\n            lo, st = ctrl_low_top(c[0])',
        '        _eo = x["family5"] == "env_overhang" and np.random.default_rng([int(x["seed"]), 13, 4]).random() < P_CRAWL5B\n'
        '        if (x["family5"] == "crawl" or _eo) and not a.crawl_legacy:\n            lo, st = ctrl_low_top(c[0])',
    ),
    (
        '            x["params"]["obj"]["u"] = r3(u)',
        '            (x["params"]["obj"] if "obj" in x["params"] else x["params"])["u"] = r3(u)\n'
        '            if _eo:\n                x["params"]["crawl"] = True; x["aggr"] = True',
    ),
]
for a_, b_ in rep:
    assert src.count(a_) == 1, a_
    src = src.replace(a_, b_)
sys.argv = [SRC] + argv
g = dict(__name__="__main__", __file__=SRC, P_CRAWL5B=p_crawl)
exec(compile(src, SRC + " (gen5b)", "exec"), g)
