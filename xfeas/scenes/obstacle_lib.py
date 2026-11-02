"""
Ground obstacles for the crouch-COST study (built IN MEMORY; assets/school.usd is never written).

REAL ASSETS BY DEFAULT (2026-10-10): make_pool() returns real_assets.RealPool, which rebuilds every box below from
duplicated real school assets (wooden planks / plank terraces / furniture rows / school books; see
xfeas/scenes/real_assets.py). BoxPool (UsdGeom.Cube primitives, plain colour or a random school material) is LEGACY
and only returned with XFEAS_LEGACY_PRIMITIVES=1 (reproducing old results). Every obstacle is a list of boxes given
in the LANE frame (s along the lane, l lateral, z above the lane floor):
    dict(s0, s1, l0, l1, z0, z1, [tilt])     tilt = rotation about the lane-lateral axis (rad, + = rising along s)
Obstacle families (all centred on the lane):
    curb(h)            one bar across the lane, depth 0.25 m, height h (step up + down)
    step(h)            a platform of height h from s0 to the lane end (step up, continue on top)
    stairs(r, n)       n steps of rise r, tread 0.30 m, then a platform
    ramp(deg)          an incline up to a platform
    debris(seed, hmax) 6-10 scattered small boxes (rubble), heights 0.4..1.0 x hmax
    overhang(u, ...)   a slab whose underside is u (the crouch object, same as overhang_lib 'prim')
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math
import numpy as np


def curb(s0, h, depth=0.25, width=2.2):
    return [dict(s0=s0, s1=s0 + depth, l0=-width / 2, l1=width / 2, z0=-0.05, z1=h)]


def step(s0, h, s_end=6.2, width=2.2):
    return [dict(s0=s0, s1=s_end, l0=-width / 2, l1=width / 2, z0=-0.05, z1=h)]


def stairs(s0, r, n=3, tread=0.30, s_end=6.2, width=2.2):
    return [dict(s0=s0 + k * tread, s1=s_end, l0=-width / 2, l1=width / 2, z0=-0.05, z1=(k + 1) * r) for k in range(n)]


def ramp(s0, deg, run=1.4, s_end=6.2, width=2.2, thick=0.3):
    th = math.radians(deg)
    H = run * math.tan(th)
    return [
        dict(s0=s0, s1=s0 + run / math.cos(th), l0=-width / 2, l1=width / 2, z0=-thick, z1=0.0, tilt=th),
        dict(s0=s0 + run, s1=s_end, l0=-width / 2, l1=width / 2, z0=-0.05, z1=H),
    ]


def debris(s0, seed, hmax=0.12, n=None, span=1.0, half_w=0.45):
    r = np.random.default_rng(seed)
    n = n or int(r.integers(6, 11))
    out = []
    for _ in range(n):
        a, b = r.uniform(0.08, 0.22, 2)
        s = s0 + r.uniform(0, span)
        l = r.uniform(-half_w, half_w)
        h = r.uniform(0.4, 1.0) * hmax
        out.append(
            dict(
                s0=s - a / 2, s1=s + a / 2, l0=l - b / 2, l1=l + b / 2, z0=-0.01, z1=h, yaw=float(r.uniform(-0.6, 0.6))
            )
        )
    return out


def overhang(s0, u, depth=0.4, thick=0.10, width=2.4):
    return [dict(s0=s0, s1=s0 + depth, l0=-width / 2, l1=width / 2, z0=u, z1=u + thick)]


def make_pool(stage, n=16, root="/World/CostObs", wall_kinds=None):
    """the obstacle pool every script should use: REAL assets unless XFEAS_LEGACY_PRIMITIVES=1"""
    import real_assets as RA

    if RA.LEGACY:
        return BoxPool(stage, n=n, root=root)
    return RA.RealPool(stage, n=n, root=root, wall_kinds=wall_kinds or RA.WALL_KINDS_TRAIN)


class BoxPool:
    """LEGACY primitive boxes (UsdGeom.Cube). Do not use for anything new (real-asset rule, 2026-10-10)."""

    def __init__(self, stage, n=16, root="/World/CostObs"):
        from pxr import UsdGeom, UsdPhysics, Gf, Usd, UsdShade

        self.Gf, self.n = Gf, n
        self.xf, self.col, self.prims = [], [], []
        for i in range(n):
            c = UsdGeom.Cube.Define(stage, f"{root}/b{i:02d}")
            c.CreateSizeAttr(1.0)
            UsdPhysics.CollisionAPI.Apply(c.GetPrim())
            self.xf.append(UsdGeom.Xformable(c.GetPrim()).MakeMatrixXform())
            self.col.append(c.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.45, 0.4)]))
            self.prims.append(c.GetPrim())
        lk = stage.GetPrimAtPath("/World/School/Geom/Looks")
        self.mats = (
            [UsdShade.Material(p) for p in Usd.PrimRange(lk) if p.IsA(UsdShade.Material)][:400] if lk.IsValid() else []
        )
        self.UsdShade = UsdShade
        self.park()

    def park(self):
        for i, x in enumerate(self.xf):
            x.Set(self.Gf.Matrix4d().SetTranslate(self.Gf.Vec3d(0, -200 - i, -60)))

    def place(self, boxes, P0, yaw, floor, color=None, mat=None):
        """boxes in the lane frame -> world. Returns the number of boxes used."""
        Gf = self.Gf
        d = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        n = np.array([-d[1], d[0], 0.0])
        z = np.array([0, 0, 1.0])
        assert len(boxes) <= self.n, len(boxes)
        self.park()
        for i, b in enumerate(boxes):
            th = b.get("tilt", 0.0)
            yw = b.get("yaw", 0.0)
            ax = d * math.cos(yw) + n * math.sin(yw)
            ay = -d * math.sin(yw) + n * math.cos(yw)
            ax, az = ax * math.cos(th) + z * math.sin(th), -ax * math.sin(th) + z * math.cos(th)
            L, W, H = b["s1"] - b["s0"], b["l1"] - b["l0"], b["z1"] - b["z0"]
            if th:
                # tilted ramp slab: top surface starts at (s0, z=0) and rises along ax; box hangs below its top face
                c = np.r_[P0, floor] + d * b["s0"] + ax * (L / 2) - az * (H / 2) + n * ((b["l0"] + b["l1"]) / 2)
            else:
                c = (
                    np.r_[P0, floor]
                    + d * ((b["s0"] + b["s1"]) / 2)
                    + n * ((b["l0"] + b["l1"]) / 2)
                    + z * ((b["z0"] + b["z1"]) / 2)
                )
            M = Gf.Matrix4d(*(list(ax * L) + [0]), *(list(ay * W) + [0]), *(list(az * H) + [0]), *(list(c) + [1]))
            self.xf[i].Set(M)
            col = color if color is not None else (0.5, 0.45, 0.4)
            self.col[i].Set([Gf.Vec3f(*map(float, col))])
            bind = self.UsdShade.MaterialBindingAPI.Apply(self.prims[i])
            if mat is not None and self.mats:
                bind.Bind(self.mats[(mat + i * 0) % len(self.mats)])
            else:
                bind.UnbindAllBindings()
        return len(boxes)
