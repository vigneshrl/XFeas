"""Point scenes/assets/school.usd at your copy of the Abandoned School asset pack.

school.usd is the baked school scene: it REFERENCES the pack's USDs, it does not contain them. The repo copy expects the pack at
third_party/scene_assets/Collected_Abandoned_school_showcase (a symlink is fine). If yours lives elsewhere, rewrite the
references once:

    $ISAACSIM_PYTHON scenes/relink_school_usd.py --pack /path/to/Collected_Abandoned_school_showcase

Every reference / payload whose path contains 'Collected_Abandoned_school_showcase' is re-pointed to --pack (relative to
school.usd when --relative, else absolute). Other references (Isaac Sim's public S3 assets) are left alone. Needs only pxr.
"""
import argparse
import os

from pxr import Sdf

PACK = "Collected_Abandoned_school_showcase"
HERE = os.path.dirname(os.path.abspath(__file__))


def relink(usd, pack, relative):
    layer = Sdf.Layer.FindOrOpen(usd)
    base = os.path.dirname(os.path.abspath(usd))
    tgt = os.path.relpath(pack, base) if relative else os.path.abspath(pack)
    n = 0

    def fix(path):
        nonlocal n
        if PACK not in path:
            return path
        n += 1
        return os.path.join(tgt, path.split(PACK, 1)[1].lstrip("/"))

    def walk(prim):
        for spec in prim.nameChildren:
            for lst, cls in ((spec.referenceList, Sdf.Reference), (spec.payloadList, Sdf.Payload)):
                for op in ("prependedItems", "appendedItems", "explicitItems"):
                    items = list(getattr(lst, op))
                    if any(PACK in it.assetPath for it in items):
                        setattr(lst, op, [cls(fix(it.assetPath), it.primPath, it.layerOffset) for it in items])
            walk(spec)

    walk(layer.pseudoRoot)
    layer.subLayerPaths[:] = [fix(p) for p in layer.subLayerPaths]
    layer.Save()
    print(f"[relink] {usd}: {n} paths -> {tgt}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", default=os.path.join(HERE, "assets", "school.usd"))
    ap.add_argument("--pack", default=os.path.join(HERE, "..", "third_party", "scene_assets", PACK))
    ap.add_argument("--absolute", action="store_true", help="write an absolute path (default: relative to school.usd)")
    a = ap.parse_args()
    relink(a.usd, a.pack, not a.absolute)
