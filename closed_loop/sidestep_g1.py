"""
G1 + AMO with UPPER-BODY / LEG COLLIDERS for the narrow-passage study.

Why: AMO's released g1.xml ships collision geometry only for the pelvis contour and 4 small spheres per foot (every
other <geom> has contype=0 conaffinity=0 or is commented out), and the Isaac import keeps exactly that
(sidestep_probe.py: horizontal raycasts hit only pelvis_contour_link, collision width 0.13 m). With it, a wall
narrower than the shoulders would be walked straight through by the arms and torso. For a gap test the body must
touch the wall, so this factory adds, IN MEMORY, convex-hull colliders built from each link's own visual mesh for
every link except the pelvis (has its contour collider), the ankles/feet (contact geometry left exactly as
released, so the gait is unchanged) and the IMU frame. Articulation self-collision stays OFF (as shipped), so the
new shapes only touch the environment. Controller, gains, masses and inertias are untouched; the USD on disk is
not modified.

    --robot g1a --robot-factory sidestep_g1:make_g1c
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import g1_amo

SKIP = ("pelvis", "pelvis_contour_link", "left_ankle_pitch_link", "left_ankle_roll_link", "right_ankle_pitch_link",
        "right_ankle_roll_link", "imu_in_torso")


def add_colliders(stage, prim_path):
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    base = stage.GetPrimAtPath(prim_path + "/pelvis")
    n = 0
    names = []
    for link in base.GetChildren():
        nm = link.GetName()
        if nm in SKIP or not link.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        vis = stage.GetPrimAtPath(f"{prim_path}/pelvis/{nm}/visuals")
        if not vis.IsValid():
            continue
        L = UsdGeom.Xformable(link).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        Linv = L.GetInverse()
        k = 0
        for m in Usd.PrimRange(vis, Usd.TraverseInstanceProxies()):
            if not m.IsA(UsdGeom.Mesh):
                continue
            src = UsdGeom.Mesh(m)
            M = UsdGeom.Xformable(m).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * Linv
            dst = UsdGeom.Mesh.Define(stage, f"{prim_path}/pelvis/{nm}/sidestep_col_{k}")
            dst.CreatePointsAttr(src.GetPointsAttr().Get())
            dst.CreateFaceVertexCountsAttr(src.GetFaceVertexCountsAttr().Get())
            dst.CreateFaceVertexIndicesAttr(src.GetFaceVertexIndicesAttr().Get())
            dst.CreatePurposeAttr(UsdGeom.Tokens.guide)                     # not rendered
            UsdGeom.Xformable(dst.GetPrim()).MakeMatrixXform().Set(Gf.Matrix4d(M))
            UsdPhysics.CollisionAPI.Apply(dst.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(dst.GetPrim()).CreateApproximationAttr("convexHull")
            k += 1; n += 1
        names.append(nm)
    print(f"[g1c] added {n} convex-hull colliders on {len(names)} links: {names}", flush=True)
    return n


def make_g1c(kind, prim_path, position, yaw=0.0):
    import omni.usd
    rob = g1_amo.make_g1a(kind, prim_path, position, yaw)
    add_colliders(omni.usd.get_context().get_stage(), prim_path)
    return rob


make_g1c.SPEC = g1_amo.make_g1a.SPEC
