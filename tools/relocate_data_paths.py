"""Re-point the absolute paths stored in the XFeas data files at YOUR data folder (run once after downloading the data).

Manifests and indexes (*.jsonl / *.json) store absolute paths of the machine that wrote them, e.g.
    "br": "/old/machine/Scripts/data/gen_v5b/full_warehouse/g1c/br/....npz"
This rewrites every '<old prefix>/data/' to '<XFeas>/data/' (or --data) in place. The old prefix is detected from the files
(the most common '.../data/' prefix) unless --old is given. Dry run by default; --apply writes.

    python tools/relocate_data_paths.py            # show what would change
    python tools/relocate_data_paths.py --apply
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import argparse
import collections
import os
import re

PAT = re.compile(r'"(/[^"\n]*?)/data/')


def files(root):
    for r, _, fs in os.walk(root, followlinks=True):
        for f in fs:
            if f.endswith((".jsonl", ".json")):
                yield os.path.join(r, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=xfeas.paths.DATA)
    ap.add_argument("--old", default="", help="old prefix, i.e. the part before /data/ (auto-detected when empty)")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    data = os.path.abspath(a.data)
    new = os.path.dirname(data)
    old = a.old
    if not old:
        c = collections.Counter()
        for f in files(data):
            with open(f, errors="ignore") as fh:
                c.update(PAT.findall(fh.read(1 << 20)))
        c.pop(new, None)
        if not c:
            print("[relocate] nothing to do: no foreign absolute data paths found")
            return
        old = c.most_common(1)[0][0]
    print(f"[relocate] {old}/data/ -> {new}/data/  ({'APPLY' if a.apply else 'dry run'})")
    n_f = n_s = 0
    for f in files(data):
        with open(f, errors="ignore") as fh:
            t = fh.read()
        k = t.count(old + "/data/")
        if not k:
            continue
        n_f += 1
        n_s += k
        if a.apply:
            tmp = f + ".tmp"
            with open(tmp, "w") as fh:
                fh.write(t.replace(old + "/data/", new + "/data/"))
            os.replace(tmp, f)
    print(f"[relocate] {n_s} paths in {n_f} files" + ("" if a.apply else "  (re-run with --apply)"))


if __name__ == "__main__":
    main()
