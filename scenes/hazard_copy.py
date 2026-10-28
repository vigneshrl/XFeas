"""
ONE implementation of "make a physically faithful copy of a school hazard object", shared by collect_posture.py
and scene_hazards.py (they diverged before). Fixes from the independent audit (analysis/audit/AUDIT_REPORT.md):

  * source = the exact sub-prim that IS the hazard (e.g. the single flat window frame inside the
    BP_school_window_11 blueprint, whose other 7 pieces stand in a wall opening);
  * the copy carries the source's FULL composed transform relative to /World/School/Geom (parents included),
    so sub-prims rotated by their parent lie exactly as in the school;
  * prims the school deactivated stay deactivated in the copy;
  * exact triangle-mesh colliders; oil binds the school's OWN physics material PM_Oil (friction combine 'min',
    mu 0.088) and its own visual material, so look AND physics equal the shipped spill.

Usage: lib = CopyLib(stage); prim = lib.make(path, cls, x, y, yaw, floor_z)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math
import numpy as np

SHOWCASE = xfeas_paths.SCENE_ASSETS + "/Collected_Abandoned_school_showcase" + "/Abandoned_school_showcase.usd"
GEOM = "/World/School/Geom"
# class -> source sub-prims (paths relative to /World/School/Geom == showcase /Root)
SOURCES = dict(
    lamp=["BP_school_lighting_106", "SM_school_lighting_6", "SM_school_lighting_9"],
    door=["InstancedSM_school_door_6_2"],                                      # BP_school_door_11: excluded (audit 2.2)
    window=["BP_school_window_11/SM_school_window_1_3/InstancedSM_school_window_1_3"],   # the flat floor frame
    mat=["SM_gym_props_14", "SM_gym_props_12", "SM_gym_props_17"],
    oil=["Oil_surface_02"])


class CopyLib:
    def __init__(self, stage):
        from pxr import Usd, UsdGeom, Gf
        self.stage, self.Usd, self.UsdGeom, self.Gf = stage, Usd, UsdGeom, Gf
        self.bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        geom = stage.GetPrimAtPath(GEOM)
        self.geom_inv = UsdGeom.Xformable(geom).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
        self.geom_w = UsdGeom.Xformable(geom).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        self.info = {}
        for cls, names in SOURCES.items():
            for n in names:
                src = stage.GetPrimAtPath(f"{GEOM}/{n}")
                if not src.IsValid():
                    raise RuntimeError(f"hazard source missing: {GEOM}/{n}")
                r = self.bbc.ComputeWorldBound(src).ComputeAlignedRange()
                mn, mx = np.array(r.GetMin()), np.array(r.GetMax())
                L = UsdGeom.Xformable(src).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * self.geom_inv
                inactive = [str(p.GetPath()).replace(str(src.GetPath()), "", 1)
                            for p in Usd.PrimRange(src, Usd.PrimAllPrimsPredicate) if not p.IsActive()]
                self.info[n] = dict(cls=cls, c0=(mn + mx) / 2, dz=float((mn[2] + mx[2]) / 2 - mn[2]), size=mx - mn,
                                    L=L, inactive=inactive)
        self._oil_phys = stage.GetPrimAtPath(f"{GEOM}/Looks/PM_Oil")
        self._oil_vis = stage.GetPrimAtPath(f"{GEOM}/Oil_surface_M")
        if not (self._oil_phys.IsValid() and self._oil_vis.IsValid()):
            raise RuntimeError("school oil materials not found (PM_Oil / Oil_surface_M)")
        self.n = 0

    def placement(self, name, x, y, yaw, floor_z, sxy=1.0):
        """parent matrix putting the copy's bbox centre at (x, y), bottom on floor_z, rotated by yaw about z;
        sxy scales the footprint (used only for oil, a flat decal, to tile it onto uneven surfaces)"""
        Gf, I = self.Gf, self.info[name]
        c0 = I["c0"]
        return (self.geom_w * Gf.Matrix4d().SetTranslate(Gf.Vec3d(-float(c0[0]), -float(c0[1]), -float(c0[2])))
                * Gf.Matrix4d().SetScale(Gf.Vec3d(float(sxy), float(sxy), 1.0))
                * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw)))
                * Gf.Matrix4d().SetTranslate(Gf.Vec3d(float(x), float(y), float(floor_z + I["dz"]))))

    def make(self, path, name, x=0.0, y=0.0, yaw=0.0, floor_z=-50.0, sxy=1.0):
        from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
        I = self.info[name]
        par = UsdGeom.Xform.Define(self.stage, path)
        ch = self.stage.DefinePrim(path + "/asset")
        ch.GetReferences().AddReference(SHOWCASE, f"/Root/{name}")
        xa = UsdGeom.Xformable(ch); xa.ClearXformOpOrder(); xa.AddTransformOp().Set(I["L"])   # full composed pose
        for rel in I["inactive"]:                                                           # keep the school's deletions
            q = self.stage.GetPrimAtPath(path + "/asset" + rel)
            if q.IsValid():
                q.SetActive(False)
        for q in Usd.PrimRange(ch):
            if q.IsInstance():
                q.SetInstanceable(False)
        for q in Usd.PrimRange(ch):
            if q.IsA(UsdGeom.Gprim):
                UsdPhysics.CollisionAPI.Apply(q)
                if q.IsA(UsdGeom.Mesh):
                    UsdPhysics.MeshCollisionAPI.Apply(q).CreateApproximationAttr().Set("none")
                if I["cls"] == "oil":
                    b = UsdShade.MaterialBindingAPI.Apply(q)
                    b.Bind(UsdShade.Material(self._oil_vis), UsdShade.Tokens.strongerThanDescendants)
                    b.Bind(UsdShade.Material(self._oil_phys), UsdShade.Tokens.strongerThanDescendants, "physics")
        xf = par.MakeMatrixXform()
        xf.Set(self.placement(name, x, y, yaw, floor_z, sxy))
        self.n += 1
        return dict(name=name, xf=xf, prim=par.GetPrim())

    def move(self, copy, x, y, yaw, floor_z):
        copy["xf"].Set(self.placement(copy["name"], x, y, yaw, floor_z))
