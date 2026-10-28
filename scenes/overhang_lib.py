"""
Overhang objects for the crouch data (built IN MEMORY at runtime; scenes/assets/school.usd is never written).

  * COPIES of school assets (hazard_copy.CopyLib machinery: exact sub-prim, full composed pose, school deletions kept,
    exact triangle-mesh colliders, the asset's own materials), re-oriented: an upright board/door leaf is laid FLAT
    (thin axis -> vertical), then yawed relative to the lane, slanted about the lane-lateral axis and rolled about
    the lane axis, uniformly scaled, and lifted so its (rotated) bbox bottom sits at the requested underside height.
  * PRIMITIVE slabs/beams (UsdGeom.Cube, analytic box collider, random colour or a random school material): LEGACY ONLY
    (XFEAS_LEGACY_PRIMITIVES=1, for reproducing results from before 2026-10-10). By default no primitive is created and
    the generator's former 'prim' share goes to real classes: 'partition' (school cloakroom partition SM_cloakroom_2,
    1.80 x 1.00 x 0.25 m laid flat), 'bench' (school gym benches SM_gym_props_5/_21, 2.65 x 0.27-0.49 x 0.30 m) and
    more 'plank' (real wooden boards). See scenes/real_assets.py.

The D1 test door (SM_school_door_3 family, BP_school_door_5) is NOT in the pool. 'cabinet' is held out of training
entirely (validation-only object type).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math
import numpy as np

SOURCES = dict(
    door=["BP_school_door_1/SM_school_door_6_2/InstancedSM_school_door_6_2",
          # BP_school_door_17 leaf removed: its copy composes to NOTHING (no mesh, no collider; audit interim1)
          "BP_school_door_13/SM_school_door_4_2/InstancedSM_school_door_4_2"],
    plank=["SM_literature_props_1", "SM_wooden_wall_linings_3"],
    # board (blackboard / history / chemistry boards) REMOVED after audit interim2: copies invisible or colliders the
    # robot passes through (1.6-1.8 cm thin meshes) -> labels inconsistent with what the camera sees
    grating=["SM_grating_window_1"],
    shelf=["SM_school_bookcase_7", "BP_school_closet_3/SM_school_closet_6_1"],
    cabinet=["BP_school_hanging_cabinet_1/SM_school_hanging_cabinet_1_1"],
    lamp=["BP_school_lighting_13/SM_school_lighting_2", "SM_school_lighting_5"],
    window=["BP_school_window_14/SM_school_window_1_1"],
    table=["SM_directors_table_1"],
    debris=["Cube2", "Cube106"],          # the school's two broken washbasins (textured meshes named 'Cube*')
    partition=["SM_cloakroom_2"],          # REAL replacement of the primitive slab (2026-10-10)
    bench=["SM_gym_props_5", "SM_gym_props_21"],
)
HELDOUT_TYPES = ("cabinet",)
RA_GEOM = "/World/School/Geom"


def rot_axis(Gf, axis, ang):
    return Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(*map(float, axis)), math.degrees(ang)))


class OverhangLib:
    def __init__(self, stage, rng, sources=None, root="/World/Overhang", primitives=None):
        import hazard_copy as HC
        import real_assets as RA
        sources = sources or SOURCES
        self.primitives = RA.LEGACY if primitives is None else bool(primitives)
        from pxr import Usd, UsdGeom, UsdShade, Gf
        self.stage, self.Gf, self.UsdGeom, self.rng = stage, Gf, UsdGeom, rng
        old = HC.SOURCES
        HC.SOURCES = sources                       # CopyLib reads its source list from the module
        try:
            self.lib = HC.CopyLib(stage)
        finally:
            HC.SOURCES = old
        self.copies = {}
        for cls, names in sources.items():
            for n in names:
                c = self.lib.make(f"{root}/{cls}_{n.split('/')[-1]}", n)      # parked at z -50
                I = self.lib.info[n]
                size = np.asarray(I["size"], float)
                c["pre"] = Gf.Matrix4d(1.0)
                if not self.primitives:     # real mode: objects lying diagonally in their bbox (door_4_2 at 127 deg) are
                    A = RA.world_points(stage.GetPrimAtPath(f"{RA_GEOM}/{n}")) - I["c0"]      # placed by their own frame
                    Q = RA.level_frame(A) if len(A) > 8 else None
                    if Q is not None:          # tilted plate (broken door frame lying on debris): level it
                        B = A @ Q; lo_, hi_ = B.min(0), B.max(0)
                        c["pre"] = (Gf.Matrix4d(*(list(map(float, Q[0])) + [0.0]), *(list(map(float, Q[1])) + [0.0]),
                                                *(list(map(float, Q[2])) + [0.0]), 0.0, 0.0, 0.0, 1.0)
                                    * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*map(float, -(lo_ + hi_) / 2))))
                        size = np.asarray(hi_ - lo_, float)
                    elif len(A) > 8:
                        yaw0, dcl, osz = RA.principal_frame(A)
                        if yaw0 != 0.0:
                            c["pre"] = (Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(-yaw0)))
                                        * Gf.Matrix4d().SetTranslate(Gf.Vec3d(*map(float, -dcl))))
                            size = np.asarray(osz, float)
                thin = int(np.argmin(size))
                if thin == 0 and size[2] > 2 * size[0]:
                    base = rot_axis(Gf, (0, 1, 0), math.pi / 2)      # thin x upright board -> flat
                elif thin == 1 and size[2] > 2 * size[1]:
                    base = rot_axis(Gf, (1, 0, 0), math.pi / 2)      # thin y upright board -> flat
                else:
                    base = Gf.Matrix4d(1.0)
                c.update(cls=cls, size=size, base=base)
                self.copies.setdefault(cls, []).append(c)
        self.cube = None
        if not self.primitives:                     # real-asset mode: no primitive slab exists at all
            self.UsdShade = UsdShade
            self.park_all()
            return
        # primitive slab (LEGACY)
        self.prim_xf = UsdGeom.Xform.Define(stage, f"{root}/prim")
        cube = UsdGeom.Cube.Define(stage, f"{root}/prim/box"); cube.CreateSizeAttr(1.0)
        from pxr import UsdPhysics
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        self.cube = cube
        self.cube_scale = UsdGeom.Xformable(cube.GetPrim()).AddScaleOp()
        self.prim_m = self.prim_xf.MakeMatrixXform()
        self.cube_color = cube.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.4, 0.3)])
        self.mats = [UsdShade.Material(p) for p in Usd.PrimRange(stage.GetPrimAtPath("/World/School/Geom/Looks"))
                     if p.IsA(UsdShade.Material)][:400] if stage.GetPrimAtPath("/World/School/Geom/Looks").IsValid() else []
        self.UsdShade = UsdShade
        self.park_all()

    def park_all(self):
        for v in self.copies.values():
            for c in v:
                c["xf"].Set(self.Gf.Matrix4d().SetTranslate(self.Gf.Vec3d(0, 0, -50)))
        if self.cube is not None:
            self.prim_m.Set(self.Gf.Matrix4d().SetTranslate(self.Gf.Vec3d(0, 0, -55)))

    def _orient(self, local, yaw, slant, roll, d, n):
        """local rotation/scale (row-vector) -> full orientation matrix about the object centre"""
        Gf = self.Gf
        return (local * rot_axis(Gf, (0, 0, 1), yaw) * rot_axis(Gf, (float(n[0]), float(n[1]), 0.0), slant)
                * rot_axis(Gf, (float(d[0]), float(d[1]), 0.0), roll))

    def _extent(self, R, half):
        Gf = self.Gf
        P = np.array([list(R.Transform(Gf.Vec3d(sx * half[0], sy * half[1], sz * half[2])))
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return P

    def place(self, spec, lane_xy, lane_yaw, floor, s_front, lat):
        """spec: cls, name (copy) or prim dims, across (bool), yaw_rel, slant, roll, scale, u (underside above floor,
        bbox bottom after rotation). Puts the object's near edge (along the lane) at s_front. -> geometry dict"""
        Gf = self.Gf
        d = np.array([math.cos(lane_yaw), math.sin(lane_yaw)]); n = np.array([-d[1], d[0]])
        if spec["cls"] == "prim":
            if self.cube is None:
                raise RuntimeError("primitive slab requested but primitives are disabled (real-asset rule); "
                                   "set XFEAS_LEGACY_PRIMITIVES=1 only to reproduce old results")
            dims = np.array([spec["depth"], spec["width"], spec["thick"]], float)       # along, across, up (lane frame)
            local = Gf.Matrix4d(1.0)
            half = dims / 2
            yaw = lane_yaw + spec["yaw_rel"]
            R = self._orient(local, yaw, spec["slant"], spec["roll"], d, n)
        else:
            c = spec["copy"]
            sc = float(spec["scale"])
            S = Gf.Matrix4d().SetScale(Gf.Vec3d(sc, sc, sc))
            local = S * c["base"]
            half0 = c["size"] / 2
            Pb = self._extent(c["base"], half0)                          # after laying flat
            hx, hy = np.ptp(Pb[:, 0]), np.ptp(Pb[:, 1])
            # which horizontal axis runs ACROSS the lane: long one across (spec['across']) or along
            long_is_x = hx >= hy
            want_x_along = (not long_is_x) if spec["across"] else long_is_x
            yaw = lane_yaw + (0.0 if want_x_along else math.pi / 2) + spec["yaw_rel"]
            R = self._orient(local, yaw, spec["slant"], spec["roll"], d, n)
            half = half0
        P = self._extent(R, half)
        along = P[:, :2] @ d; across = P[:, :2] @ n
        zmin = P[:, 2].min()
        depth = float(np.ptp(along)); width = float(np.ptp(across)); thick = float(np.ptp(P[:, 2]))
        s_c = s_front + depth / 2.0
        xy = np.asarray(lane_xy, float) + s_c * d + lat * n
        zc = floor + spec["u"] - zmin
        T = Gf.Matrix4d().SetTranslate(Gf.Vec3d(float(xy[0]), float(xy[1]), float(zc)))
        if spec["cls"] == "prim":
            self.cube_scale.Set(Gf.Vec3f(*map(float, spec_dims(spec))))
            self.prim_m.Set(R * T)
            col = spec.get("color", (0.5, 0.4, 0.3))
            self.cube_color.Set([Gf.Vec3f(*map(float, col))])
            b = self.UsdShade.MaterialBindingAPI.Apply(self.cube.GetPrim())
            if spec.get("mat") is not None and self.mats:
                b.Bind(self.mats[spec["mat"] % len(self.mats)])
            else:
                b.UnbindAllBindings()
        else:
            c = spec["copy"]; L = self.lib
            c0 = L.info[c["name"]]["c0"]
            M = (L.geom_w * Gf.Matrix4d().SetTranslate(Gf.Vec3d(-float(c0[0]), -float(c0[1]), -float(c0[2]))) * c.get("pre", Gf.Matrix4d(1.0)) * R * T)
            c["xf"].Set(M)
        return dict(depth=round(depth, 3), width=round(width, 3), thick=round(thick, 3), s_front=round(s_front, 3),
                    s_end=round(s_front + depth, 3), center=[round(float(xy[0]), 3), round(float(xy[1]), 3)],
                    u_bbox=round(float(spec["u"]), 3))


def spec_dims(spec):
    return (spec["depth"], spec["width"], spec["thick"])
