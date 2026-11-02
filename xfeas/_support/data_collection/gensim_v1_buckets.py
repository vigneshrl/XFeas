"""bucket / subfamily mapping of gen_v1 situations (no Isaac imports; same as gensim_v1)"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)


def bucket_of(sit):
    """the 4 user buckets (+ null) from a situation spec"""
    fam = sit["family"]
    if fam == "null":
        return "null"
    if fam in ("overhang", "compound", "gap", "step", "env_overhang", "env_gap", "env_climb", "env_compound"):
        return "setpoint"
    if fam == "terrain":
        return "slippery" if sit["params"].get("slippery") else "rough"
    if fam.startswith("env_"):
        return "scene"
    return "other"


def subfamily(sit):
    fam, P = sit["family"], sit["params"]
    if fam == "overhang":
        return "overhang_crouch"
    if fam == "compound":
        return "compound_oh_curb"
    if fam == "gap":
        return "narrow_gap"
    if fam == "step":
        return f"climb_{P.get('kind')}"
    if fam == "null":
        return f"null_{P.get('decoy', 'none')}"
    if fam == "terrain":
        return f"terrain_{P.get('type')}"
    return fam
