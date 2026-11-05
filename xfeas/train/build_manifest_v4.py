#!/usr/bin/env python
"""Builds the training manifest for the retrain with SONIC (v4) from a raw branch-dataset manifest, and runs a data-quality
check. Exit code 1 means the data failed the check and training should not start.

Details:
gen5b manifest v4 for the SONIC retrain. Input: a raw `branch_dataset.py build` manifest. Output: the v4 manifest + a
data-QA report; exit code 1 when the QA gate fails (the overnight chain then does not train).
  1. the v3 filter: SONIC s1 situations (sid _(5303|5423)dddd__) lose latL / latR (old pure-lateral sidestep, before 'face')
  2. stand-up relabel: at SONIC B
     moments (posture held, group 'B:*') the class 'stock' must mean the staged stand-up. Rows with a 'stockS' branch drop
     the direct-switch 'stock' branch and rename stockS -> stock (same command, so the same candidate features); rows without
     one (s1-s3, before the staged option existed) drop their direct-switch 'stock' branch. Rows left with < 2 branches go.
QA gate (SONIC): >= 100 stockS branches; staged success at 'end' / 'post*' moments >= 0.8 and above the direct switch there.
  python3 xfeas/train/build_manifest_v4.py RAW OUT
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, re, sys
from collections import Counter

S1 = re.compile(r"_(5303|5423)\d{4}__")


def main(raw, out):
    rows = [json.loads(l) for l in open(raw) if l.strip()]
    kept, n_drop, c = [], Counter(), Counter()
    for r in rows:
        keep = list(range(len(r["cls"])))
        if r["base"] == "g1s" and S1.search(r["sid"]):
            keep = [j for j in keep if r["cls"][j] not in ("latL", "latR")]
        if r["base"] == "g1s" and r["group"].startswith("B:"):
            lab_ok = r["label"] == "end" or r["label"].startswith("post")
            for j in keep:
                if r["cls"][j] in ("stock", "stockS"):
                    c[(r["cls"][j], lab_ok, r["y_success"][j] > 0.5)] += 1
            if "stockS" in [r["cls"][j] for j in keep]:
                keep = [j for j in keep if r["cls"][j] != "stock"]
                n_drop["direct_stock_replaced"] += 1
            else:
                n_drop["direct_stock_dropped"] += sum(r["cls"][j] == "stock" for j in keep)
                keep = [j for j in keep if r["cls"][j] != "stock"]
        if len(keep) < 2:
            n_drop["lt2_branches"] += 1
            continue
        r = dict(
            r,
            branches=[r["branches"][j] for j in keep],
            y_success=[r["y_success"][j] for j in keep],
            cls=["stock" if r["cls"][j] == "stockS" else r["cls"][j] for j in keep],
        )
        kept.append(r)
    with open(out, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    rate = lambda k, ok: c[(k, ok, True)] / max(1, c[(k, ok, True)] + c[(k, ok, False)])
    n_s = sum(v for (k, _, _), v in c.items() if k == "stockS")
    rep = dict(
        rows_in=len(rows),
        rows_out=len(kept),
        branches_out=sum(len(r["branches"]) for r in kept),
        drops=dict(n_drop),
        stockS_n=n_s,
        stockS_succ_clear=round(rate("stockS", True), 3),
        stock_direct_succ_clear=round(rate("stock", True), 3),
        stockS_succ_under=round(rate("stockS", False), 3),
        stock_direct_succ_under=round(rate("stock", False), 3),
        g1s_rows=sum(r["base"] == "g1s" for r in kept),
        bases=dict(Counter(r["base"] for r in kept)),
    )
    fails = []
    if n_s < 100:
        fails.append(f"stockS branches {n_s} < 100")
    if rep["stockS_succ_clear"] < 0.8:
        fails.append(f"staged success past the obstacle {rep['stockS_succ_clear']} < 0.8")
    if rep["stockS_succ_clear"] <= rep["stock_direct_succ_clear"]:
        fails.append("staged not above the direct switch past the obstacle")
    rep["qa"] = "PASS" if not fails else "FAIL: " + "; ".join(fails)
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(out + ".qa.json", "w"), indent=1)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
