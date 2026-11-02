"""
School scene in its ORIGINAL state, cleaned of asset defects. Since 2026-10-07 this is BAKED into
assets/school.usd (bake_scene_clean.py; the pre-clean file is scenes/assets/school_ORIGINAL_before_clean_20261007.usd).
apply(stage) is idempotent, so calling it right after opening the school is still fine and changes nothing.

  original_lighting: only the lights shipped with the Abandoned School asset (/World/School/Geom:
      1 DistantLight + 7 SphereLights, identical to Abandoned_school_showcase.usd). Everything this
      project added on top (build_scenes.py 'lighting B': NavDome/RoofSun/Emerg*/Headlight;
      school_scene.py RoomLights; ProbeDome) is deactivated.
  fixture_lights: the shipped lights cover the ground floor only -- the upper west corridor (U01)
      and the west-wing classrooms (G01) receive exactly zero light. There, the existing ceiling
      lamp fixtures are 'switched on': exact copies of the shipped SphereLight (same intensity,
      radius, shaping, transform), moved under those fixtures. No other light is changed.
  remove_hanging: deactivates the ceiling fluorescent lamps + cables hanging > 0.5 m above a floor
      (395 prims, school_hanging.json) -- several dangle from one cable and read as floating
      objects -- and SM_lathe_02, a workshop lathe misplaced into the gym 0.67 m above its floor.
      Lamps lying ON a floor are kept: BP_school_lighting_106 is the fallen-lamp hazard of U01.
  fix_collision: the 24 props that ship with NO collider (both drilling machines -- robots walked
      through the one in the gym --, books, biology props) get static convex-hull colliders.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, os

THIS = os.path.dirname(os.path.abspath(__file__))
ADDED_LIGHTS = [
    "/World/RoomLights",
    "/World/ProbeDome",
    "/World/ExpDome",
    "/World/NavDome",
    "/World/RoofSun",
    "/World/Headlight",
] + [f"/World/Emerg{i}" for i in range(6)]
FLOATING = [
    "/World/School/Geom/BP_lathe_1/SM_lathe_02",
    "/World/School/Geom/Rubble_surface_01",
    "/World/School/Geom/Rubble_surface_02",
]  # untextured white 1 mm cubes (removed 2026-10-10)
SHIPPED_LAMP = "/World/School/Geom/SphereLight_01"  # z 2.91 above its floor (0.0)
# (x, y, floor z) under existing ceiling fixtures in the zones with no shipped light
FIXTURES = [
    (-26.25, 1.31, 5.07),
    (-26.25, 4.18, 5.07),
    (-26.27, 7.20, 5.07),
    (-26.27, 10.28, 5.07),
    (-26.6, 13.6, 5.07),
    (-30.23, 1.24, 0.68),
    (-30.23, -1.32, 0.68),
    (-30.24, -4.29, 0.68),
    (-30.26, -7.55, 0.68),
]


def original_lighting(stage):
    n = 0
    for p in ADDED_LIGHTS:
        pr = stage.GetPrimAtPath(p)
        if pr.IsValid() and pr.IsActive():
            pr.SetActive(False)
            n += 1
    return n


def fixture_lights(stage, extra=False):
    from pxr import UsdGeom, UsdLux, Gf

    src = stage.GetPrimAtPath(SHIPPED_LAMP)
    if not src.IsValid():
        return 0
    M = UsdGeom.Xformable(src).ComputeLocalToWorldTransform(0)
    t0 = M.ExtractTranslation()
    UsdGeom.Xform.Define(stage, "/World/FixtureLights")
    fx = list(FIXTURES) + (json.load(open(os.path.join(THIS, "school_extra_fixtures.json"))) if extra else [])
    for i, (x, y, fz) in enumerate(fx):
        L = UsdLux.SphereLight.Define(stage, f"/World/FixtureLights/L{i}")
        for a in src.GetAttributes():
            if a.GetName().startswith("inputs:") and a.HasAuthoredValue():
                L.GetPrim().CreateAttribute(a.GetName(), a.GetTypeName()).Set(a.Get())
        Mi = Gf.Matrix4d(M)
        Mi.SetTranslateOnly(Gf.Vec3d(x, y, fz + t0[2]))
        UsdGeom.Xformable(L.GetPrim()).MakeMatrixXform().Set(Mi)
    return len(fx)


def remove_hanging(stage):
    n = 0
    for p in json.load(open(os.path.join(THIS, "school_hanging.json"))) + FLOATING:
        pr = stage.GetPrimAtPath(p)
        if pr.IsValid() and pr.IsActive():
            pr.SetActive(False)
            n += 1
    return n


def fix_collision(stage):
    from pxr import Usd, UsdGeom, UsdPhysics

    n = 0
    for p in json.load(open(os.path.join(THIS, "school_nocollision.json"))):
        u = stage.GetPrimAtPath(p)
        if not u.IsValid():
            continue
        if u.IsInstance():
            u.SetInstanceable(False)
        for m in Usd.PrimRange(u):
            if m.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(m)
                UsdPhysics.MeshCollisionAPI.Apply(m).CreateApproximationAttr().Set("convexHull")
                n += 1
    return n


def apply(stage, lighting=True, hanging=True, collision=True, extra_fixtures=False):
    """extra_fixtures: also switch on the ceiling fixtures of every other unlit area (data collection
    only; none within 7 m of a test hazard, so the six test scenes look identical either way)"""
    nl = original_lighting(stage) if lighting else 0
    nf = fixture_lights(stage, extra_fixtures) if lighting else 0
    nh = remove_hanging(stage) if hanging else 0
    nc = fix_collision(stage) if collision else 0
    print(
        f"[scene_clean] original lighting (-{nl} added lights, +{nf} unlit-zone fixtures), "
        f"removed {nh} hanging/floating prims, colliders on {nc} meshes",
        flush=True,
    )
