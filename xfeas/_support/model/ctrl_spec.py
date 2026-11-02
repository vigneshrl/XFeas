"""
xfeas.ctrl_spec/1 loader + helpers (analysis/GEN3_SPEC.md section 1). Pure python.

A spec describes a controller's command INTERFACE only (names, roles, types, ranges, nominal values, mode list with what the
docs say). It never contains a measured response. `binding` is harness plumbing (which adapter, which adapter key per name).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import copy, hashlib, json, os

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
SPEC_DIRS = (os.path.join(THIS, "specs"), os.path.join(ROOT, "data", "gen_v3", "specs"))
ROLES = (
    "lin_vel_x",
    "lin_vel_y",
    "yaw_rate",
    "pelvis_height",
    "torso_pitch",
    "torso_roll",
    "mode",
    "speed",
    "torso_yaw",
)
HIST_ROLES = ("lin_vel_x", "lin_vel_y", "yaw_rate", "pelvis_height", "torso_pitch", "torso_roll", "mode", "speed")
VEL_ROLES = ("lin_vel_x", "lin_vel_y", "yaw_rate", "speed")
REAL = ("g1c", "g1g", "g1l", "g1u", "g1s")
HELD_OUT = ("g1s",)


def load(ctrl_id_or_path):
    p = ctrl_id_or_path
    if not os.path.exists(p):
        for d in SPEC_DIRS:
            q = os.path.join(d, f"{ctrl_id_or_path}.json")
            if os.path.exists(q):
                p = q
                break
    s = json.load(open(p))
    validate(s)
    return s


def validate(s):
    assert s.get("schema") == "xfeas.ctrl_spec/1", s.get("schema")
    names = set()
    for c in s["commands"]:
        assert c["role"] in ROLES, c
        assert c["name"] not in names, c["name"]
        names.add(c["name"])
        if c["type"] == "continuous":
            lo, hi = c["range"]
            assert lo <= hi, c
        else:
            assert c["values"], c
    roles = [c["role"] for c in s["commands"]]
    assert len(roles) == len(set(roles)), f"duplicate role in {s['ctrl_id']}"
    if "mode" in roles:
        mv = {m["name"] for m in s["modes"]}
        assert set(cmd(s, "mode")["values"]) <= mv, "mode values must be listed in modes"
    return True


def spec_hash(s):
    pub = {k: v for k, v in s.items() if k not in ("binding", "doc_notes", "sources")}
    return hashlib.sha1(json.dumps(pub, sort_keys=True).encode()).hexdigest()[:12]


def cmd(s, role):
    for c in s["commands"]:
        if c["role"] == role:
            return c
    return None


def has(s, role):
    return cmd(s, role) is not None


def rng(s, role):
    c = cmd(s, role)
    return tuple(c["range"]) if c is not None and c["type"] == "continuous" else None


def nominal_cmd(s):
    """spec-level nominal command dict {name: value} (velocities 0)"""
    out = {}
    for c in s["commands"]:
        out[c["name"]] = c["nominal"]
    return out


def mode_info(s, name):
    for m in s.get("modes", []):
        if m["name"] == name:
            return m
    return None


def mode_index(s, name):
    for i, m in enumerate(s.get("modes", [])):
        if m["name"] == name:
            return i
    return -1


def moving_modes_round_robin(s):
    """modes with moving and not static, interleaved over the spec's own mode groups (docs order inside a group); the
    nominal mode first (it is the default walking mode)"""
    c = cmd(s, "mode")
    if c is None:
        return []
    allowed = set(c["values"])
    groups, order = {}, []
    for m in s["modes"]:
        if m["name"] not in allowed or not m.get("moving") or m.get("static"):
            continue
        if m["group"] not in groups:
            groups[m["group"]] = []
            order.append(m["group"])
        groups[m["group"]].append(m)
    out = []
    k = 0
    while any(groups.values()):
        g = order[k % len(order)]
        if groups[g]:
            out.append(groups[g].pop(0))
        k += 1
    nom = c["nominal"]
    out.sort(key=lambda m: 0 if m["name"] == nom else 1)
    return out


def public_view(s):
    """what a model may read: roles / types / units / ranges / nominal / mode attributes (no names, docs, binding)"""
    cm = [
        dict(
            role=c["role"],
            type=c["type"],
            unit=c.get("unit"),
            range=c.get("range"),
            nominal=c["nominal"] if c["type"] == "continuous" else None,
            n_values=len(c.get("values", [])),
        )
        for c in s["commands"]
    ]
    md = [
        dict(
            group=m["group"],
            moving=m["moving"],
            static=m["static"],
            speed_range=m["speed_range"],
            uses_height=m["uses_height"],
            height_range=m["height_range"],
        )
        for m in s.get("modes", [])
    ]
    return dict(rate_hz=s["rate_hz"], commands=cm, modes=md)


def copy_spec(s):
    return copy.deepcopy(s)


if __name__ == "__main__":
    for k in REAL:
        s = load(k)
        print(
            k,
            spec_hash(s),
            [c["role"] for c in s["commands"]],
            len(s["modes"]),
            [m["name"] for m in moving_modes_round_robin(s)][:12],
        )
