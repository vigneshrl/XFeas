"""
REAL-ASSET hazard library (2026-10-10; the DEFAULT for every generator and scene from now on).

Standing rule (user): every obstacle or hazard is built ONLY by duplicating the environment's own real assets with
their own meshes and materials. No UsdGeom.Cube/Sphere/Cylinder/... primitives, no flat-coloured or synthetic
surfaces. The old primitive builders are kept only behind XFEAS_LEGACY_PRIMITIVES=1, for reproducing old results.

Copies follow hazard_copy.CopyLib's audited recipe (analysis/audit/AUDIT_REPORT.md: the exact sub-prim that is the
object, its FULL composed pose relative to /World/School/Geom, school deactivations kept (door_11 bug), instancing
broken, exact triangle-mesh colliders, the asset's OWN materials). RealLib adds on top:
  * a principal frame per source from the composed mesh points (PCA in xy): some school objects sit diagonally in
    their bbox (the door leaf InstancedSM_school_door_4_2 lies at 127 deg; its AABB is 2.1 x 2.5 m, the leaf
    2.58 x 0.73 x 0.14 m). Sources within 0.5 deg of the axes keep the bbox frame.
  * orientation (as_is / flat = thinnest axis vertical / on_side) + yaw + UNIFORM scale, limited to +-15 % (visible
    distortion otherwise); placement by the rotated oriented box (bottom on a given z).
  * builders in a LANE frame (s along the lane from P0, l lateral, z above the lane floor):
      plank_block   stack of real wooden planks (SM_literature_props_1, 3.00 x 0.50 x 0.040 m, closed solid)
                    with any top height: k layers x uniform scale (0.85-1.15) + the bottom layer set into the
                    floor by <= 2 cm  -> curbs, ledges, platforms, decoy piles
      plank_terrace staircase / terrace of plank layers (risers k x 0.040 x s, any tread) -> steps, stairs
      furniture_line a row of real furniture (lockers, bookcases, shelves, wardrobes) with fronts facing a given
                    side, the first piece's side face EXACTLY at a given line (gap edge), filling a given length
                    (up to max_embed into an adjacent wall) -> gap walls, blocked hallways, corridor walls
  * RealPool: drop-in replacement of obstacle_lib.BoxPool (same place()/park() API): every lane-frame box is
    rebuilt from real assets (raised box -> real planks with the same underside; low ground box -> plank stack;
    tall ground box -> furniture row; staircase box sets -> plank terrace; small yawed boxes -> books/school
    debris; tilted ramp boxes -> tilted planks).
  * audit_no_primitives(stage, roots): every Gprim under the roots must be a Mesh that comes from a referenced
    asset layer (not authored in this session) and has a bound material; anything else is a violation.

School catalog (surveys 2026-10-10: isolated renders + ray audits of each copy; sizes are oriented boxes, metres):
  plank      SM_literature_props_1        3.00 x 0.50 x 0.040 wooden board; rays hit top (0.036-0.040) AND bottom
  partition  SM_cloakroom_2               1.80 x 1.00 x 0.25 wooden cloakroom partition (overhang slab when flat)
  bench      SM_gym_props_5 / _21         2.65 x 0.27 x 0.30 gym bench
  door_leaf  BP_school_door_13/.../InstancedSM_school_door_4_2   2.58 x 0.73 x 0.14 panelled door leaf (diagonal)
  locker     BP_school_closet_1           0.975 x 0.726 x 2.00 blue metal locker, back +y (measured: wall 1.8 cm)
  locker2    BP_school_closet_2           1.46 x 0.95 x 2.00 double locker, back +y
  bookcase   BP_school_bookcase_2         0.953 x 0.375 x 1.67, back +y
  bookcase4  BP_school_bookcase_4         0.954 x 0.456 x 1.67, back -y
  shelf      SM_school_closet_4           0.947 x 0.232 x 1.89 (frame: 0.232 x 0.947), back +x
  wardrobe   BP_school_wardrobe_2         0.908 x 0.584 x 1.70, back -y   (TEST scenes only: T1's object family)
  wardrobe1  BP_school_wardrobe_1         0.903 x 0.509 x 1.70, back +y   (TEST scenes only)
  book       SM_school_book_1             0.29 x 0.25 x 0.11 (debris)
Rejected by the survey: SM_wooden_wall_linings_* (8/28/3/37/38 are gym wall BARS = open frames; the solid boards
1/13/18/22/24/31 are ONE-SIDED relief sheets, rays from the back pass through); InstancedSM_school_door_6_2 (a
door FRAME, only 21 % of its bbox has geometry); BP_school_door_5 leaf (D1's door; composes to nothing usable,
g1_overhang.py); the white Rubble_surface decals (deactivated).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math, os
import numpy as np

LEGACY = os.environ.get("XFEAS_LEGACY_PRIMITIVES", "0") == "1"
SHOWCASE_HINT = "Abandoned_school_showcase"

CATALOG = dict(
    plank=dict(src="SM_literature_props_1", t=0.040, t_top=0.0362),
    plank_b=dict(src="SM_literature_props_10", t=0.040, t_top=0.0362),
    partition=dict(src="SM_cloakroom_2"),
    bench=dict(src="SM_gym_props_5"),
    bench_b=dict(src="SM_gym_props_21"),
    door_leaf=dict(src="BP_school_door_13/SM_school_door_4_2/InstancedSM_school_door_4_2"),
    locker=dict(src="BP_school_closet_1", back="+y"),
    locker2=dict(src="BP_school_closet_2", back="+y"),
    bookcase=dict(src="BP_school_bookcase_2", back="+y"),
    bookcase4=dict(src="BP_school_bookcase_4", back="-y"),
    shelf=dict(src="SM_school_closet_4", back="+x"),
    wardrobe=dict(src="BP_school_wardrobe_2", back="-y"),
    wardrobe1=dict(src="BP_school_wardrobe_1", back="+y"),
    book=dict(src="SM_school_book_1"),
    # SUPPORTS for raised objects (flat tops measured 2026-10-10: rays hit the whole top)
    closet3=dict(src="BP_school_closet_3", back="+y"),  # low wooden cabinet 0.955 x 0.357 x 0.864
    bookcase3=dict(src="BP_school_bookcase_3", back="+y"),  # low cabinet with doors 0.953 x 0.375 x 0.815
    desk=dict(src="SM_school_desk_2"),  # school desk 1.20 x 0.65 x 0.757
    table=dict(src="BP_table_room_2"),  # table 1.00 x 0.74 x 0.75
)
# support columns: (kind, may stand on a plank pile). Wardrobes never (T1 family).
SUPPORT_KINDS = ("bench", "desk", "table", "bookcase3", "closet3", "bookcase", "locker")
SUPPORT_ON_PILE = ("closet3", "bookcase3", "bookcase")
L_IN = 0.65  # supports keep the lane open: inner face >= 0.65 m from the lane centre (Spot 0.55 m wide)
WALL_KINDS_TRAIN = ("locker", "bookcase", "bookcase4", "shelf")  # generator walls (never wardrobes: T1; locker2 has a
# swung-open door -> ragged edges, not used)
WALL_KINDS_TEST = ("wardrobe", "wardrobe1", "locker")
SCALE_LIM = (0.85, 1.15)
_BACK_ANG = {"+x": 0.0, "+y": math.pi / 2, "-x": math.pi, "-y": -math.pi / 2}


def _rot(Gf, axis, ang):
    return Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(*map(float, axis)), math.degrees(ang)))


class RealLib:
    """pool of real-asset copies under `root`, placed with principal-frame orientation + uniform scale."""

    def __init__(self, stage, root="/World/RealAssets", catalog=None):
        import hazard_copy as HC
        from pxr import Usd, UsdGeom, Gf

        self.stage, self.Usd, self.UsdGeom, self.Gf, self.HC = stage, Usd, UsdGeom, Gf, HC
        self.catalog = dict(CATALOG, **(catalog or {}))
        old = HC.SOURCES
        HC.SOURCES = {}
        try:
            self.lib = HC.CopyLib(stage)  # geom transforms + oil materials; no sources yet
        finally:
            HC.SOURCES = old
        k = 0
        while stage.GetPrimAtPath(root if k == 0 else f"{root}_{k}").IsValid():  # several builds in one stage
            k += 1
        self.root = root if k == 0 else f"{root}_{k}"
        UsdGeom.Xform.Define(stage, self.root)
        self.frames = {}  # src -> dict(yaw0, dc (3,), size (3,))
        self.pool, self.used = {}, {}
        self.placed = []  # (kind, copy, world AABB) of the current build

    # ------------------------------------------------------------ sources
    def _add_source(self, src):
        L, Usd, UsdGeom, Gf = self.lib, self.Usd, self.UsdGeom, self.Gf
        if src in L.info:
            return
        p = self.stage.GetPrimAtPath(f"{self.HC.GEOM}/{src}")
        if not p.IsValid():
            raise RuntimeError(f"real asset source missing: {self.HC.GEOM}/{src}")
        r = L.bbc.ComputeWorldBound(p).ComputeAlignedRange()
        mn, mx = np.array(r.GetMin()), np.array(r.GetMax())
        Lm = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * L.geom_inv
        inactive = [
            str(q.GetPath()).replace(str(p.GetPath()), "", 1)
            for q in Usd.PrimRange(p, Usd.PrimAllPrimsPredicate)
            if not q.IsActive()
        ]
        L.info[src] = dict(
            cls="real", c0=(mn + mx) / 2, dz=float((mn[2] + mx[2]) / 2 - mn[2]), size=mx - mn, L=Lm, inactive=inactive
        )
        # principal frame from the composed mesh points (instance proxies), relative to the bbox centre
        P = []
        for q in Usd.PrimRange(p, Usd.TraverseInstanceProxies()):
            if q.IsA(UsdGeom.Mesh) and q.IsActive():
                pts = UsdGeom.Mesh(q).GetPointsAttr().Get()
                if pts is None or len(pts) == 0:
                    continue
                M = np.array(UsdGeom.Xformable(q).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
                A = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ M
                P.append(A[:, :3])
        A = np.vstack(P) - L.info[src]["c0"] if P else None
        yaw0 = 0.0
        if A is not None and len(A) > 8:
            # minimum-area bounding rectangle in xy (exact frame for box-like objects; PCA is biased by handles etc.)
            xy = A[:, :2]
            angs = np.radians(np.arange(0.0, 90.0, 0.1))
            areas = []
            for a in angs:
                c, s_ = math.cos(a), math.sin(a)
                u = xy[:, 0] * c + xy[:, 1] * s_
                v = -xy[:, 0] * s_ + xy[:, 1] * c
                areas.append(np.ptp(u) * np.ptp(v))
            a_best = float(angs[int(np.argmin(areas))])
            if math.degrees(min(a_best, math.pi / 2 - a_best)) > 0.3 and min(areas) < 0.995 * areas[0]:
                yaw0 = a_best if a_best <= math.pi / 4 else a_best - math.pi / 2
        Q = level_frame(A) if A is not None and len(A) > 8 else None
        if A is not None and Q is not None:  # tilted plate: level it first
            B = A @ Q
            lo, hi = B.min(0), B.max(0)
            self.frames[src] = dict(yaw0=0.0, Q=Q, dcl=(lo + hi) / 2, size=hi - lo)
            return
        if A is not None:
            c, s_ = math.cos(-yaw0), math.sin(-yaw0)
            B = A @ np.array([[c, s_, 0], [-s_, c, 0], [0, 0, 1.0]])  # row vectors rotated by -yaw0
            lo, hi = B.min(0), B.max(0)
            dcl = (lo + hi) / 2  # OBB centre in the principal frame
            size = hi - lo
        else:
            dcl, size = np.zeros(3), np.asarray(L.info[src]["size"], float)
        self.frames[src] = dict(yaw0=yaw0, dcl=dcl, size=size)

    def size(self, kind):
        src = self.catalog[kind]["src"]
        self._add_source(src)
        return np.asarray(self.frames[src]["size"], float)

    def flat_dims(self, kind):
        """(long, short, thick) of the object laid flat; flat_yaw() turns its long side onto local x"""
        sz = np.sort(self.size(kind))[::-1]
        return float(sz[0]), float(sz[1]), float(sz[2])

    def flat_yaw(self, kind):
        R, P = self.orient_matrix(kind, "flat", 0.0, 1.0)
        return 0.0 if np.ptp(P[:, 0]) >= np.ptp(P[:, 1]) else math.pi / 2

    def get(self, kind):
        """next unused copy of `kind` (created lazily; call while the timeline is STOPPED, colliders follow at reset)"""
        src = self.catalog[kind]["src"]
        self._add_source(src)
        lst = self.pool.setdefault(kind, [])
        k = self.used.get(kind, 0)
        if k >= len(lst):
            c = self.lib.make(f"{self.root}/{kind}_{len(lst):03d}", src)
            c["kind"] = kind
            lst.append(c)
        self.used[kind] = k + 1
        return lst[k]

    def park(self):
        Gf = self.Gf
        for kind, lst in self.pool.items():
            for i, c in enumerate(lst):
                c["xf"].Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(float(i) * 4.0, -400.0 - 5.0 * len(kind), -80.0)))
        self.used = {}
        self.placed = []

    # ------------------------------------------------------------ placement
    def orient_matrix(self, kind, orient="as_is", yaw=0.0, scale=1.0, pitch=0.0, pitch_axis=(0.0, 1.0)):
        """local rotation (about the object's OBB centre, principal frame) -> (Gf matrix, rotated half extents)"""
        Gf = self.Gf
        size = self.size(kind)
        sc = float(scale)
        if not (SCALE_LIM[0] - 1e-6 <= sc <= SCALE_LIM[1] + 1e-6):
            raise ValueError(f"scale {sc:.3f} outside {SCALE_LIM} (visible distortion)")
        if orient == "flat":
            thin = int(np.argmin(size))
            Ro = (
                _rot(Gf, (0, 1, 0), math.pi / 2)
                if thin == 0
                else (_rot(Gf, (1, 0, 0), math.pi / 2) if thin == 1 else Gf.Matrix4d(1.0))
            )
        elif orient == "on_side":  # long axis horizontal, middle axis vertical
            o = np.argsort(size)  # thin, mid, long
            if o[1] == 2:
                Ro = Gf.Matrix4d(1.0)
            elif o[1] == 0:
                Ro = _rot(Gf, (0, 1, 0), math.pi / 2)
            else:
                Ro = _rot(Gf, (1, 0, 0), math.pi / 2)
        else:
            Ro = Gf.Matrix4d(1.0)
        R = Gf.Matrix4d().SetScale(Gf.Vec3d(sc, sc, sc)) * Ro * _rot(Gf, (0, 0, 1), yaw)
        if pitch:
            R = R * _rot(Gf, (float(pitch_axis[0]), float(pitch_axis[1]), 0.0), pitch)
        h = size / 2
        P = np.array(
            [
                list(R.Transform(Gf.Vec3d(sx * h[0], sy * h[1], sz * h[2])))
                for sx in (-1, 1)
                for sy in (-1, 1)
                for sz in (-1, 1)
            ]
        )
        return R, P

    def place(
        self, kind, x, y, z_bottom, orient="as_is", yaw=0.0, scale=1.0, pitch=0.0, pitch_axis=(0.0, 1.0), copy=None
    ):
        """put a copy with its rotated OBB centre at (x, y) and its lowest corner at z_bottom -> dict(copy, mn, mx)"""
        Gf = self.Gf
        c = copy or self.get(kind)
        src = self.catalog[kind]["src"]
        fr = self.frames[src]
        R, P = self.orient_matrix(kind, orient, yaw, scale, pitch, pitch_axis)
        I = self.lib.info[src]
        c0 = I["c0"]
        zc = z_bottom - P[:, 2].min()
        Mrot = _rot(Gf, (0, 0, 1), -fr["yaw0"])
        if fr.get("Q") is not None:
            Q = fr["Q"]
            Mrot = Gf.Matrix4d(
                *(list(map(float, Q[0])) + [0.0]),
                *(list(map(float, Q[1])) + [0.0]),
                *(list(map(float, Q[2])) + [0.0]),
                0.0,
                0.0,
                0.0,
                1.0,
            )
        M = (
            self.lib.geom_w
            * Gf.Matrix4d().SetTranslate(Gf.Vec3d(-float(c0[0]), -float(c0[1]), -float(c0[2])))
            * Mrot
            * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*map(float, -fr["dcl"])))
            * R
            * Gf.Matrix4d().SetTranslate(Gf.Vec3d(float(x), float(y), float(zc)))
        )
        c["xf"].Set(M)
        mn = P.min(0) + np.array([x, y, zc])
        mx = P.max(0) + np.array([x, y, zc])
        rec = dict(kind=kind, src=src, copy=c, mn=mn, mx=mx, scale=float(scale), orient=orient, yaw=float(yaw), M=M)
        self.placed.append(rec)
        return rec

    def shift(self, recs, dxy):
        """translate placed copies by dxy (world, metres) -- e.g. to correct a ray-measured gap edge"""
        Gf = self.Gf
        for r in recs:
            r["M"] = r["M"] * Gf.Matrix4d().SetTranslate(Gf.Vec3d(float(dxy[0]), float(dxy[1]), 0.0))
            r["copy"]["xf"].Set(r["M"])
            r["mn"] = r["mn"] + np.r_[dxy[:2], 0.0]
            r["mx"] = r["mx"] + np.r_[dxy[:2], 0.0]

    # ------------------------------------------------------------ builders (lane frame)
    @staticmethod
    def lane(P0, yaw):
        d = np.array([math.cos(yaw), math.sin(yaw)])
        n = np.array([-d[1], d[0]])
        return np.asarray(P0, float)[:2], d, n

    def solve_stack(self, h, kind="plank", sink_max=0.02):
        """k layers x uniform scale s (+ bottom sink) giving top height h. -> (k, s, sink, h_built)"""
        t, tt = self.catalog[kind]["t"], self.catalog[kind]["t_top"]
        best = None
        for k in range(1, 40):
            s = h / ((k - 1) * t + tt)
            sink = 0.0
            if s > SCALE_LIM[1]:
                continue
            if s < SCALE_LIM[0]:
                s = SCALE_LIM[0]
                sink = ((k - 1) * t + tt) * s - h
            hb = ((k - 1) * t + tt) * s - min(sink, sink_max)
            cand = (abs(hb - h), k, s, min(sink, sink_max), hb)
            if best is None or cand[0] < best[0] - 1e-9:
                best = cand
            if sink <= sink_max:
                break
        return best[1], best[2], best[3], best[4]

    def plank_block(
        self, P0, yaw, floor, s0, s1, l0, l1, h, orient="across", sink_max=0.02, kind="plank", gap=0.002, lat_free=None
    ):
        """stack of planks covering [s0,s1] x [l0,l1] with its TOP at h above the floor; front edge exactly at s0.
        orient 'across': plank long axis across the lane (depth 0.5 s per plank along the lane); 'along': long axis
        along the lane. lat_free=(left, right) free lateral room from the lane centre (shifts planks to stay clear)."""
        P0, d, n = self.lane(P0, yaw)
        k, s, sink, hb = self.solve_stack(h, kind, sink_max)
        Lp, Wp, _ = self.flat_dims(kind)
        Lp, Wp = Lp * s, Wp * s
        t = self.catalog[kind]["t"] * s
        fy = self.flat_yaw(kind)
        if orient == "across":
            dep, wid, pyaw = Wp, Lp, yaw + math.pi / 2 + fy
        else:
            dep, wid, pyaw = Lp, Wp, yaw + fy
        n_a = max(1, int(math.ceil((s1 - s0) / (dep + gap) - 1e-6)))
        n_c = max(1, int(math.ceil((l1 - l0) / (wid + gap) - 1e-6)))
        Wtot = n_c * wid + (n_c - 1) * gap
        lc = 0.5 * (l0 + l1)
        if lat_free is not None:  # keep the pile inside the free corridor when possible
            lo_ok, hi_ok = -lat_free[1] + 0.02, lat_free[0] - 0.02
            if hi_ok - lo_ok >= Wtot:
                lc = float(np.clip(lc, lo_ok + Wtot / 2, hi_ok - Wtot / 2))
        out = []
        for j in range(k):
            zb = floor - sink + j * t
            for ia in range(n_a):
                sc_ = s0 + ia * (dep + gap) + dep / 2
                for ic in range(n_c):
                    lcc = lc - Wtot / 2 + ic * (wid + gap) + wid / 2
                    xy = P0 + d * sc_ + n * lcc
                    out.append(self.place(kind, xy[0], xy[1], zb, "flat", pyaw, s))
        return dict(
            kind="plank_block",
            k=k,
            scale=round(s, 4),
            sink=round(sink, 4),
            h_target=round(h, 4),
            h_built=round(hb, 4),
            s_range=(round(s0, 3), round(s0 + n_a * (dep + gap) - gap, 3)),
            l_range=(round(lc - Wtot / 2, 3), round(lc + Wtot / 2, 3)),
            n=len(out),
            orient=orient,
        )

    def plank_terrace(
        self, P0, yaw, floor, steps, l0, l1, s_end_req, kind="plank", gap=0.002, scale=None, lat_free=None
    ):
        """terrace of plank layers. steps = [(s_start_i, z_top_i)] increasing; the layers of step i begin exactly at
        s_start_i; every layer reaches at least s_end_req and never further back than the layer below it (no
        overhang). Riser heights come from k x 0.040 x s layers (same s everywhere); the bottom layer may sink <= 2 cm.
        -> dict with built tops per step."""
        P0, d, n = self.lane(P0, yaw)
        t, tt = self.catalog[kind]["t"], self.catalog[kind]["t_top"]
        zt = [z for _, z in steps]
        rises = np.diff([0.0] + zt)
        if scale is None:  # one scale for all layers: best fit of every riser
            best = None
            for s in np.linspace(SCALE_LIM[0], SCALE_LIM[1], 61):
                ks = [max(1, int(round(r / (t * s)))) for r in rises]
                err = sum(abs(ks[i] * t * s - rises[i]) for i in range(len(rises)))
                if (
                    best is None
                    or err < best[0] - 1e-9
                    or (abs(err - best[0]) < 1e-9 and abs(s - 1) < abs(best[1] - 1))
                ):
                    best = (err, s, ks)
            _, s, ks = best
        else:
            s = float(scale)
            ks = [max(1, int(round(r / (t * s)))) for r in rises]
        Lp, Wp, _ = self.flat_dims(kind)
        Lp, Wp = Lp * s, Wp * s
        dep, wid, pyaw = Wp, Lp, yaw + math.pi / 2 + self.flat_yaw(kind)  # planks ACROSS the lane: exact tread edges
        n_c = max(1, int(math.ceil((l1 - l0) / (wid + gap) - 1e-6)))
        Wtot = n_c * wid + (n_c - 1) * gap
        lc = 0.5 * (l0 + l1)
        if lat_free is not None:
            lo_ok, hi_ok = -lat_free[1] + 0.02, lat_free[0] - 0.02
            if hi_ok - lo_ok >= Wtot:
                lc = float(np.clip(lc, lo_ok + Wtot / 2, hi_ok - Wtot / 2))
        # ends from the top step down (each layer at least as long as the one above)
        cnt = [0] * len(steps)
        req = s_end_req
        for i in range(len(steps) - 1, -1, -1):
            cnt[i] = max(1, int(math.ceil((req - steps[i][0]) / (dep + gap) - 1e-6)))
            req = steps[i][0] + cnt[i] * (dep + gap) - gap
        tops, z = [], -0.0
        out = []
        layer = 0
        for i, (ss, _) in enumerate(steps):
            for j in range(ks[i]):
                zb = floor + layer * t * s
                for ia in range(cnt[i]):
                    sc_ = ss + ia * (dep + gap) + dep / 2
                    for ic in range(n_c):
                        lcc = lc - Wtot / 2 + ic * (wid + gap) + wid / 2
                        xy = P0 + d * sc_ + n * lcc
                        out.append(self.place(kind, xy[0], xy[1], zb, "flat", pyaw, s))
                layer += 1
            tops.append(round(((layer - 1) * t + tt) * s, 4))
        ends = [round(steps[i][0] + cnt[i] * (dep + gap) - gap, 3) for i in range(len(steps))]
        return dict(
            kind="plank_terrace",
            scale=round(s, 4),
            layers_per_step=ks,
            tops=tops,
            risers=[round(k_ * t * s, 4) for k_ in ks],
            ends=ends,
            l_range=(round(lc - Wtot / 2, 3), round(lc + Wtot / 2, 3)),
            n=len(out),
        )

    def support_plan(self, top, kinds=SUPPORT_KINDS, on_pile=SUPPORT_ON_PILE, max_lat=None):
        """how to build a support column whose TOP is exactly `top` above the floor: one piece of real furniture
        (uniform scale 0.85-1.15), or a pile of real planks, or furniture standing on a plank pile.
        -> dict(kind or None, scale, pile_h, cost, lat) ; lat = lateral thickness of the column"""
        best = None
        Wpl = self.flat_dims("plank")[1]
        for k in kinds:
            H = self.size(k)[2]
            s = top / H
            lat = self._lat_depth(k) * s
            if SCALE_LIM[0] <= s <= SCALE_LIM[1] and (max_lat is None or lat <= max_lat):
                c = abs(s - 1.0)
                if best is None or c < best["cost"]:
                    best = dict(kind=k, scale=s, pile_h=0.0, cost=c, lat=lat)
        if 0.03 <= top <= 0.66:
            c = 0.35 + 0.5 * top
            if best is None or c < best["cost"]:
                best = dict(kind=None, scale=None, pile_h=top, cost=c, lat=Wpl)
        for k in on_pile:
            H = self.size(k)[2]
            for s in np.linspace(SCALE_LIM[0], SCALE_LIM[1], 31):
                p = top - H * s
                if 0.03 <= p <= 0.5:
                    lat = max(self._lat_depth(k) * s, Wpl)
                    c = 0.6 + abs(s - 1.0) + p
                    if (max_lat is None or lat <= max_lat) and (best is None or c < best["cost"]):
                        best = dict(kind=k, scale=float(s), pile_h=float(p), cost=c, lat=lat)
        if best is not None:  # extent ALONG the lane (furniture front width / 3 m plank pile)
            al = 0.0
            if best["kind"] is not None:
                sz = self.size(best["kind"])
                back = self.catalog[best["kind"]].get("back")
                al = (max(sz[0], sz[1]) if back is None else (sz[1] if back in ("+x", "-x") else sz[0])) * best["scale"]
            if best["kind"] is None:  # the TOP piece touches the object: pile only -> the pile's length
                al = self.flat_dims("plank")[0] * SCALE_LIM[1]
            best["along"] = al
        return best

    def _lat_depth(self, k):
        """lateral thickness of a support piece placed with its front towards the lane (desks/tables: short side)"""
        sz = self.size(k)
        back = self.catalog[k].get("back")
        if back is None:
            return float(min(sz[0], sz[1]))
        return float(sz[0] if back in ("+x", "-x") else sz[1])

    def support_column(self, P0, yaw, floor, s_c, side, top, l_in=L_IN, kinds=SUPPORT_KINDS, max_lat=None):
        """real support column beside the lane: inner face at lateral side*l_in, centred at lane position s_c, its top
        exactly `top` above the floor (furniture front towards the lane). -> dict (None if impossible)"""
        P0a, d, n = self.lane(P0, yaw)
        pl = self.support_plan(top, kinds, max_lat=max_lat)
        if pl is None:
            return None
        lat = pl["lat"]
        lc = side * (l_in + lat / 2)
        xy = P0a + d * s_c + n * lc
        pieces = []
        if pl["pile_h"] > 0:
            k, sp, sink, hb = self.solve_stack(pl["pile_h"], "plank", 0.02)
            t = self.catalog["plank"]["t"] * sp
            for j in range(k):  # one plank wide, planks ALONG the lane
                pieces.append(
                    self.place("plank", xy[0], xy[1], floor - sink + j * t, "flat", yaw + self.flat_yaw("plank"), sp)
                )
            pl["pile_built"] = round(hb, 4)
            pl["pile_layers"] = k
        if pl["kind"] is not None:
            k = pl["kind"]
            back = self.catalog[k].get("back")
            if back is None:  # desk / table: long side along the lane
                sz = self.size(k)
                yw = yaw + (0.0 if sz[0] >= sz[1] else math.pi / 2)
            else:  # back points away from the lane
                yw = math.atan2(side * n[1], side * n[0]) - _BACK_ANG[back]
            pieces.append(self.place(k, xy[0], xy[1], floor + pl["pile_h"], "as_is", yw, pl["scale"]))
        top_b = pl["pile_h"] + (self.size(pl["kind"])[2] * pl["scale"] if pl["kind"] else 0.0)
        return dict(
            kind=pl["kind"],
            scale=None if pl["scale"] is None else round(pl["scale"], 4),
            pile=round(pl["pile_h"], 4),
            top=round(top_b, 4),
            side=side,
            l=(round(side * l_in, 3), round(side * (l_in + lat), 3)),
            s_c=round(s_c, 3),
            n_pieces=len(pieces),
        )

    def place_beside(
        self, kind, P0, yaw, floor, s_c, side, l_inner, top=None, orient="as_is", scale=None, z_bottom=None
    ):
        """one support piece beside the lane: orientation `orient`, turned (0 / 90 deg) so its LATERAL extent is the
        smallest, inner face exactly at side*l_inner, centred at lane position s_c, standing on z_bottom (default the
        floor); if `top` is given the uniform scale makes its top exactly `top` (must stay within 0.85-1.15)."""
        P0a, d, n = self.lane(P0, yaw)
        best = None
        for extra in (0.0, math.pi / 2):
            R, P = self.orient_matrix(kind, orient, yaw + extra, 1.0)
            lat = float(np.ptp(P[:, :2] @ n))
            H = float(np.ptp(P[:, 2]))
            if best is None or lat < best[0] - 1e-6:
                best = (lat, extra, H)
        lat, extra, H = best
        sc = float(top / H) if (top is not None and scale is None) else float(scale or 1.0)
        if not (SCALE_LIM[0] - 1e-6 <= sc <= SCALE_LIM[1] + 1e-6):
            return None
        zb = floor if z_bottom is None else z_bottom
        xy = P0a + d * s_c + n * side * (l_inner + lat * sc / 2)
        rec = self.place(
            kind,
            xy[0],
            xy[1],
            zb if top is None else floor + top - H * sc if z_bottom is None else zb,
            orient,
            yaw + extra,
            sc,
        )
        return dict(
            kind=kind,
            orient=orient,
            scale=round(sc, 4),
            top=round((rec["mx"][2] - floor), 4),
            side=side,
            l=(round(side * l_inner, 3), round(side * (l_inner + lat * sc), 3)),
            s_c=round(s_c, 3),
        )

    def raised_planks(self, P0, yaw, floor, s0, s1, l0, l1, u, kind="plank", gap=0.002, lat_free=None):
        """overhang made of real planks laid flat ACROSS the lane, underside exactly u above the floor, front edge at s0,
        depth ~ (s1 - s0) (n planks x 0.5 x s, s in 0.85-1.15)."""
        P0, d, n = self.lane(P0, yaw)
        D = max(s1 - s0, 0.05)
        Lp0, Wp0, _ = self.flat_dims(kind)
        n_a = max(1, int(round(D / Wp0)))
        s = float(np.clip(D / (n_a * Wp0), *SCALE_LIM))
        if abs(n_a * Wp0 * s - D) > 0.05 and n_a > 1:
            alt = float(np.clip(D / ((n_a - 1) * Wp0), *SCALE_LIM))
            if abs((n_a - 1) * Wp0 * alt - D) < abs(n_a * Wp0 * s - D):
                n_a, s = n_a - 1, alt
        dep, wid = Wp0 * s, Lp0 * s
        n_c = max(1, int(math.ceil((l1 - l0) / (wid + gap) - 1e-6)))
        Wtot = n_c * wid + (n_c - 1) * gap
        lc = 0.5 * (l0 + l1)
        if lat_free is not None:
            lo_ok, hi_ok = -lat_free[1] + 0.02, lat_free[0] - 0.02
            if hi_ok - lo_ok >= Wtot:
                lc = float(np.clip(lc, lo_ok + Wtot / 2, hi_ok - Wtot / 2))
        out = []
        for ia in range(n_a):
            sc_ = s0 + ia * (dep + gap) + dep / 2
            for ic in range(n_c):
                lcc = lc - Wtot / 2 + ic * (wid + gap) + wid / 2
                xy = P0 + d * sc_ + n * lcc
                out.append(
                    self.place(kind, xy[0], xy[1], floor + u, "flat", yaw + math.pi / 2 + self.flat_yaw(kind), s)
                )
        return dict(
            kind="raised_planks",
            u=round(u, 4),
            depth=round(n_a * dep + (n_a - 1) * gap, 4),
            scale=round(s, 4),
            thick=round(self.catalog[kind]["t"] * s, 4),
            l_range=(round(lc - Wtot / 2, 3), round(lc + Wtot / 2, 3)),
            n=len(out),
        )

    def furniture_plan(
        self,
        length,
        kinds,
        max_embed=0.06,
        srange=(0.95, 1.05),
        depth_target=None,
        rotated_ok=True,
        max_pieces=14,
        overlap=0.03,
    ):
        """choose pieces (kind, rotated?) whose total width = length .. length + max_embed with ONE uniform scale in
        srange (fewest pieces, scale closest to 1, depth closest to depth_target) -> (pieces, scale) or None"""
        opts = []
        for k in kinds:
            sz = self.size(k)
            back = self.catalog[k].get("back", "+y")
            w_front = sz[1] if back in ("+x", "-x") else sz[0]  # width when the FRONT faces the robot
            d_front = sz[0] if back in ("+x", "-x") else sz[1]
            opts.append((k, False, w_front, d_front))
            if rotated_ok:
                opts.append((k, True, d_front, w_front))  # side faces the robot
        best = None
        best_n = None
        import itertools

        for npc in range(1, max_pieces + 1):
            if best_n is not None and npc > best_n + 1:
                break
            for combo in itertools.combinations_with_replacement(range(len(opts)), npc):
                W0 = sum(opts[i][2] for i in combo)
                ov = overlap * (npc - 1)  # neighbours overlap by `overlap` (handles / recesses)
                e_pref = min(0.03, max_embed)  # prefer reaching 3 cm into the adjacent wall
                for target in (length, length + e_pref, length + max_embed / 2, length + max_embed):
                    s = (target + ov) / W0
                    if not (srange[0] <= s <= srange[1]):
                        continue
                    built = W0 * s - ov
                    if not (length - 1e-4 <= built <= length + max_embed + 1e-4):
                        continue
                    dep = np.mean([opts[i][3] for i in combo]) * s
                    cost = (
                        npc
                        + 8.0 * abs(s - 1.0)
                        + 0.5 * sum(opts[i][1] for i in combo)
                        + 4.0 * abs(built - (length + e_pref))
                    )
                    if depth_target is not None:
                        cost += 6.0 * abs(dep - depth_target) + 3.0 * np.ptp([opts[i][3] for i in combo])
                    if best is None or cost < best[0]:
                        best = (cost, [opts[i] for i in combo], s)
                        best_n = npc if best_n is None else best_n
        if best is None:
            return None
        # rotated pieces go to the OUTER end (against the wall), unrotated near the gap
        pieces = sorted(best[1], key=lambda o: o[1])
        return pieces, best[2], overlap

    def furniture_line(
        self,
        start_xy,
        u,
        f,
        length,
        floor,
        kinds,
        max_embed=0.06,
        srange=(0.95, 1.05),
        depth_target=None,
        rotated_ok=True,
        plan=None,
    ):
        """row of furniture: the FIRST piece's side face exactly at start_xy, pieces side by side along unit u for
        `length` (+ <= max_embed); front faces point along f (unit, horizontal); all fronts in the plane through
        start_xy normal to f. -> dict(pieces, scale, built_len, depths)"""
        start_xy, u, f = np.asarray(start_xy, float), np.asarray(u, float), np.asarray(f, float)
        P = plan or self.furniture_plan(length, kinds, max_embed, srange, depth_target, rotated_ok)
        if P is None:
            raise RuntimeError(f"no furniture combination for length {length:.3f} with {kinds}")
        pieces, s = P[0], P[1]
        ov = P[2] if len(P) > 2 else 0.0
        cum = 0.0
        recs = []
        f_ang = math.atan2(f[1], f[0])
        for k, rot, w, dd in pieces:
            back = self.catalog[k].get("back", "+y")
            b_ang = _BACK_ANG[back]
            # back must point along -f (away from the viewer). Rotated (side to the viewer): the flat BACK faces the
            # previous piece / the gap (-u) and the front with its handles faces outwards (into the wall), so
            # neighbouring pieces close without a slit and the gap edge is a flat panel
            want = (f_ang + math.pi) if not rot else math.atan2(-u[1], -u[0])
            yaw = want - b_ang
            ww, dep = w * s, dd * s
            c = start_xy + u * (cum + ww / 2) - f * (dep / 2)
            rec = self.place(k, c[0], c[1], floor, "as_is", yaw, s)
            rec.update(rotated=rot, width=round(ww, 4), depth=round(dep, 4))
            recs.append(rec)
            cum += ww - ov
        cum += ov
        return dict(
            pieces=[(r["kind"], r["rotated"], r["width"], r["depth"]) for r in recs],
            scale=round(s, 4),
            built_len=round(cum, 4),
            length=round(length, 4),
            recs=recs,
        )


def principal_frame(A):
    """points relative to the bbox centre (N,3) -> (yaw0, OBB centre in the principal frame, OBB size): minimum-area
    rectangle in xy; sources within 0.3 deg of the axes keep the bbox frame"""
    yaw0 = 0.0
    xy = A[:, :2]
    angs = np.radians(np.arange(0.0, 90.0, 0.1))
    areas = []
    for a in angs:
        c, s_ = math.cos(a), math.sin(a)
        areas.append(np.ptp(xy[:, 0] * c + xy[:, 1] * s_) * np.ptp(-xy[:, 0] * s_ + xy[:, 1] * c))
    a_best = float(angs[int(np.argmin(areas))])
    if math.degrees(min(a_best, math.pi / 2 - a_best)) > 0.3 and min(areas) < 0.995 * areas[0]:
        yaw0 = a_best if a_best <= math.pi / 4 else a_best - math.pi / 2
    c, s_ = math.cos(-yaw0), math.sin(-yaw0)
    B = A @ np.array([[c, s_, 0], [-s_, c, 0], [0, 0, 1.0]])
    lo, hi = B.min(0), B.max(0)
    return yaw0, (lo + hi) / 2, hi - lo


def level_frame(A):
    """objects lying slightly TILTED in the school (a door leaf lying on debris): if the thinnest principal axis is
    1-20 deg off vertical and the object is plate-like, return the 3x3 row-vector rotation Q (v_new = v @ Q) that
    makes it level (thin axis -> z, long axis -> x); else None. A level copy rests flat on flat support tops."""
    X = A - A.mean(0)
    w, V = np.linalg.eigh(np.cov(X.T))  # ascending
    thin, mid, lng = V[:, 0], V[:, 1], V[:, 2]
    if thin[2] < 0:
        thin = -thin
    tilt = math.degrees(math.acos(min(1.0, abs(thin[2]))))
    if not (1.0 <= tilt <= 20.0) or math.sqrt(max(w[0], 0)) > 0.3 * math.sqrt(w[1]):
        return None
    x = lng - thin * np.dot(lng, thin)
    x /= np.linalg.norm(x)
    y = np.cross(thin, x)
    return np.stack([x, y, thin], axis=1)  # columns = new axes in old coords -> v_new = v @ Q


def world_points(prim):
    """all composed mesh vertices under prim, world frame (N, 3)"""
    from pxr import Usd, UsdGeom

    P = []
    for q in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if q.IsA(UsdGeom.Mesh) and q.IsActive():
            mesh = UsdGeom.Mesh(q)
            pts = mesh.GetPointsAttr().Get()
            if pts is None or len(pts) == 0:
                continue
            idx = mesh.GetFaceVertexIndicesAttr().Get()
            pts = np.asarray(pts, float)
            if idx is not None and len(idx):  # only vertices used by faces (assets carry stray points)
                pts = pts[np.unique(np.asarray(idx, int))]
            M = np.array(UsdGeom.Xformable(q).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            P.append((np.c_[pts, np.ones(len(pts))] @ M)[:, :3])
    return np.vstack(P) if P else np.zeros((0, 3))


def supports_under(lib, prim, P0, yaw, floor, l_in=L_IN, room=(1.0, 1.0)):
    """NO FLOATING OBJECTS for a pre-physics scene build: from the object's mesh vertices find, on each side of the
    lane, where it reaches beyond |l| = l_in and its LOWEST point over a column footprint there; build a real support
    column with exactly that top. -> list of column dicts (None for a side without geometry)"""
    P0a, d, n = lib.lane(P0, yaw)
    A = world_points(prim)
    s_ = (A[:, :2] - P0a) @ d
    l_ = (A[:, :2] - P0a) @ n
    z_ = A[:, 2] - floor
    out = []
    for i, side in enumerate((+1, -1)):
        m = (side * l_ >= l_in) & (side * l_ <= l_in + 0.55)
        if not m.any():
            out.append(None)
            continue
        s_c = float(np.median(s_[m]))
        T = float(z_[m & (np.abs(s_ - s_c) <= 0.55)].min())
        for _ in range(3):  # top = lowest vertex over the column's ACTUAL footprint
            pl = lib.support_plan(T, max_lat=room[i])
            if pl is None:
                break
            lat_top = pl["lat"] if pl["kind"] is None else lib._lat_depth(pl["kind"]) * pl["scale"]
            l_lo = l_in + 0.5 * (pl["lat"] - lat_top)  # top piece is centred on the column
            fm = (side * l_ >= l_lo) & (side * l_ <= l_lo + lat_top) & (np.abs(s_ - s_c) <= pl["along"] / 2)
            if not fm.any():
                break
            T_new = float(z_[fm].min())
            if abs(T_new - T) < 1e-4:
                break
            T = T_new
        out.append(lib.support_column(P0a, yaw, floor, s_c, side, T, l_in=l_in, max_lat=room[i]))
    return out


# ---------------------------------------------------------------------------- BoxPool replacement
class RealPool:
    """drop-in replacement of obstacle_lib.BoxPool: place(boxes, P0, yaw, floor, color=None, mat=None) rebuilds the
    lane-frame boxes from REAL assets (colour/material arguments are ignored: real assets carry their own)."""

    def __init__(self, stage, n=16, root="/World/CostObs", wall_kinds=WALL_KINDS_TRAIN, lib=None):
        self.lib = lib or RealLib(stage, root=root)
        self.wall_kinds = wall_kinds
        self.n = n
        self.log = []
        self.lines = []  # furniture lines of the last place(): (records, unit u, inner lateral l)
        self.lat_free = None  # optional (left, right) free room from the lane centre, set by the caller

    def park(self):
        self.lib.park()
        self.log = []
        self.lines = []

    @staticmethod
    def _is_terrace(bx):
        if len(bx) < 2:
            return False
        l0, l1, s1 = bx[0]["l0"], bx[0]["l1"], bx[0]["s1"]
        ok = all(abs(b["l0"] - l0) < 1e-6 and abs(b["l1"] - l1) < 1e-6 and abs(b["s1"] - s1) < 1e-6 for b in bx)
        bs = sorted(bx, key=lambda b: b["s0"])
        return ok and all(bs[i]["z1"] < bs[i + 1]["z1"] for i in range(len(bs) - 1))

    def place(self, boxes, P0, yaw, floor, color=None, mat=None):
        L = self.lib
        L.park()
        self.log = []
        self.lines = []
        P0a, d, n = L.lane(P0, yaw)
        ground = [b for b in boxes if b["z0"] <= 0.02 and not b.get("tilt") and not b.get("yaw")]
        raised = [b for b in boxes if b["z0"] > 0.02 and not b.get("tilt")]
        tilted = [b for b in boxes if b.get("tilt")]
        small = [b for b in boxes if b.get("yaw") and b["z0"] <= 0.02]
        # staircases first: ground boxes sharing lateral range and back end with rising tops (any height)
        groups = {}
        for b in ground:
            groups.setdefault((round(b["l0"], 3), round(b["l1"], 3), round(b["s1"], 3)), []).append(b)
        terr = [bx for bx in groups.values() if self._is_terrace(bx)]
        in_terr = {id(b) for bx in terr for b in bx}
        walls = [b for b in ground if id(b) not in in_terr and b["z1"] - b["z0"] > 0.6]
        low = [b for b in ground if id(b) not in in_terr and b["z1"] - b["z0"] <= 0.6]
        groups = {("t", i): bx for i, bx in enumerate(terr)}
        for b in low:
            groups.setdefault((round(b["l0"], 3), round(b["l1"], 3), round(b["s1"], 3), "low"), []).append(b)
        for key, bx in groups.items():
            if key[0] == "t":
                bs = sorted(bx, key=lambda b: b["s0"])
                self.log.append(
                    L.plank_terrace(
                        P0a,
                        yaw,
                        floor,
                        [(b["s0"], b["z1"]) for b in bs],
                        bs[0]["l0"],
                        bs[0]["l1"],
                        bs[-1]["s1"] - 0.0,
                        lat_free=self.lat_free,
                    )
                )
            else:
                for b in bx:
                    D = b["s1"] - b["s0"]
                    orient = "across" if D < 1.2 else "along"
                    self.log.append(
                        L.plank_block(
                            P0a,
                            yaw,
                            floor,
                            b["s0"],
                            b["s1"],
                            b["l0"],
                            b["l1"],
                            b["z1"],
                            orient=orient,
                            lat_free=self.lat_free,
                        )
                    )
        for b in raised:
            self.log.append(
                L.raised_planks(P0a, yaw, floor, b["s0"], b["s1"], b["l0"], b["l1"], b["z0"], lat_free=self.lat_free)
            )
        for b in walls:
            inner_l, outer_l = (b["l0"], b["l1"]) if abs(b["l0"]) <= abs(b["l1"]) else (b["l1"], b["l0"])
            sgn = 1.0 if outer_l > inner_l else -1.0
            start = P0a + d * b["s0"] + n * inner_l
            r = L.furniture_line(
                start,
                n * sgn,
                -d,
                abs(outer_l - inner_l),
                floor,
                self.wall_kinds,
                max_embed=0.08,
                srange=(0.88, 1.12),
                depth_target=b["s1"] - b["s0"],
            )
            self.lines.append(dict(recs=r.pop("recs"), u=n * sgn, inner_l=inner_l))
            self.log.append(dict(kind="furniture_line", **r))
        for b in tilted:  # ramp: planks across, tilted about the lateral axis
            th = float(b["tilt"])
            Ls = b["s1"] - b["s0"]
            Wp = L.flat_dims("plank")[1]
            fy = L.flat_yaw("plank")
            n_a = max(1, int(round(Ls / Wp)))
            s = float(np.clip(Ls / (n_a * Wp), *SCALE_LIM))
            t = L.catalog["plank"]["t"] * s
            for ia in range(n_a):
                a = (ia + 0.5) * Wp * s
                sc_ = b["s0"] + a * math.cos(th) + 0.5 * t * math.sin(th)
                zc = a * math.sin(th) - 0.5 * t * math.cos(th)
                xy = P0a + d * sc_ + n * 0.5 * (b["l0"] + b["l1"])
                # lowest corner of the tilted plank
                R, Pc = L.orient_matrix("plank", "flat", yaw + math.pi / 2 + fy, s, pitch=-th, pitch_axis=(n[0], n[1]))
                L.place(
                    "plank",
                    xy[0],
                    xy[1],
                    floor + zc + Pc[:, 2].min(),
                    "flat",
                    yaw + math.pi / 2 + fy,
                    s,
                    pitch=-th,
                    pitch_axis=(n[0], n[1]),
                )
            self.log.append(dict(kind="ramp_planks", n=n_a, scale=round(s, 3), tilt=round(th, 3)))
        for b in small:  # debris: real school books
            h = b["z1"] - b["z0"]
            hb = L.size("book")[2]
            s = float(np.clip(h / hb, *SCALE_LIM))
            sink = max(0.0, hb * s - h)
            c = P0a + d * 0.5 * (b["s0"] + b["s1"]) + n * 0.5 * (b["l0"] + b["l1"])
            L.place("book", c[0], c[1], floor - min(sink, 0.04), "as_is", yaw + float(b["yaw"]), s)
            self.log.append(dict(kind="book", h=round(h, 3), h_built=round(hb * s - min(sink, 0.04), 3)))
        return len(boxes)


def forbid_primitive(what):
    """guard for the old synthetic-primitive builders (xe_arena friction lanes, pilot boxes, ...): they may only run to
    reproduce old results (XFEAS_LEGACY_PRIMITIVES=1)"""
    if not LEGACY:
        raise RuntimeError(
            f"{what}: primitive geometry is not allowed (real-asset rule, 2026-10-10). "
            "Use real_assets.RealLib / RealPool, or set XFEAS_LEGACY_PRIMITIVES=1 only to reproduce old results."
        )


# ---------------------------------------------------------------------------- audit
def audit_no_primitives(stage, roots, skip=()):
    """every Gprim under `roots` must be a UsdGeom.Mesh whose definition comes from a REFERENCED asset layer (not a
    prim authored in this session / the root layer) and that has a bound material. -> dict(ok, n_gprims, bad[...])"""
    from pxr import Usd, UsdGeom, UsdShade

    root_layer = stage.GetRootLayer()
    sess = stage.GetSessionLayer()
    local = {root_layer.identifier, sess.identifier}
    n, bad = 0, []
    for r in roots:
        rp = stage.GetPrimAtPath(r)
        if not rp.IsValid():
            continue
        for q in Usd.PrimRange(rp):
            ps = str(q.GetPath())
            if any(ps.startswith(s) for s in skip):
                continue
            if not q.IsA(UsdGeom.Gprim):
                continue
            n += 1
            if not q.IsA(UsdGeom.Mesh):
                bad.append((ps, f"non-mesh primitive {q.GetTypeName()}"))
                continue
            defs = [sp.layer.identifier for sp in q.GetPrimStack() if sp.typeName]
            if not defs or all(lid in local for lid in defs):
                bad.append((ps, "mesh defined in this session (not from an asset)"))
                continue
            m, _ = UsdShade.MaterialBindingAPI(q).ComputeBoundMaterial()
            if not m:
                bad.append((ps, "no material bound"))
                continue
            mdefs = [sp.layer.identifier for sp in m.GetPrim().GetPrimStack() if sp.typeName]
            if not mdefs or all(lid in local for lid in mdefs):
                bad.append((ps, f"material {m.GetPath()} authored in this session"))
                continue
    return dict(ok=not bad, n_gprims=n, n_bad=len(bad), bad=bad[:20])


def audit_stage(stage, skip=("/World/School",)):
    """whole-stage variant: every Gprim under /World outside the environment (and the given skip roots, e.g. the
    robot) must pass audit_no_primitives."""
    return audit_no_primitives(stage, ["/World"], skip=tuple(skip))
