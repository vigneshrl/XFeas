"""
Build terrain tiles (catalog.py specs) into the CURRENT USD stage. Must run inside a SimulationApp started with the
Isaac Lab venv (isaac_sim/isaaclab_python.sh), because geometry comes from isaaclab.terrains:
  * height-field layers: isaaclab.terrains.height_field.hf_terrains.<fn>.__wrapped__  (raw int16 height fields,
    summed, multiplied by a smoothstep taper that ramps the relief in over the first / last 1 m -- it is a ramp,
    not a flat margin) -> isaaclab ...height_field.utils.convert_height_field_to_mesh
  * trimesh terrains (stairs): isaaclab.terrains.trimesh.mesh_terrains.<fn>  (trimesh meshes, concatenated)
Every tile becomes one UsdGeom.Mesh (+ st UVs in metres / uv_m, vertex normals) with an exact triangle-mesh
collider, a visual material (OmniPBR on the NVIDIA library textures, or a vMaterials_2 MDL) and a PhysX material
bound with purpose "physics". Dressing (rocks, grass, roots, puddles, oil slick) is added on top.

    tiles = build_field(stage, spec)   # spec = catalog.field_spec(...)
    tiles[i]["height_at"](x, y)        # world terrain height (m) of the tile's mesh (no dressing)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math, os
import numpy as np

from catalog import VEG, VIS, PHYS

ROCKS = sorted(os.path.join(VEG, "Rocks", f) for f in os.listdir(os.path.join(VEG, "Rocks")) if f.endswith(".usda"))
OMNIPBR = "OmniPBR.mdl"


def _foot_material():
    try:
        import foot_material
    except ImportError:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "xembody"))
        import foot_material
    return foot_material


# ------------------------------------------------------------------ materials
class Materials:
    def __init__(self, stage, root="/World/TerrainLooks", proot="/World/TerrainPhysMats"):
        from pxr import UsdGeom
        self.stage, self.root, self.proot = stage, root, proot
        UsdGeom.Scope.Define(stage, root); UsdGeom.Scope.Define(stage, proot)
        self.vis, self.phys = {}, {}

    def visual(self, name, v=None):
        if name in self.vis:
            return self.vis[name]
        from pxr import UsdShade, Sdf, Gf
        v = v or VIS[name]
        path = f"{self.root}/{name}"
        mat = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, path + "/Shader")
        sh.SetSourceAsset(OMNIPBR if v["kind"] == "omnipbr" else v["mdl"], "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR" if v["kind"] == "omnipbr" else v["sub"], "mdl")
        out = sh.CreateOutput("out", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput("mdl").ConnectToSource(out)
        mat.CreateVolumeOutput("mdl").ConnectToSource(out)
        mat.CreateDisplacementOutput("mdl").ConnectToSource(out)
        if v["kind"] == "omnipbr":
            I = lambda n, t, x: sh.CreateInput(n, t).Set(x)
            A = Sdf.ValueTypeNames
            if v.get("diffuse"):
                I("diffuse_texture", A.Asset, Sdf.AssetPath(v["diffuse"]))
            if v.get("color") is not None:
                I("diffuse_color_constant", A.Color3f, Gf.Vec3f(*v["color"]))
            if v.get("tint") is not None:
                I("diffuse_tint", A.Color3f, Gf.Vec3f(*v["tint"]))
            if v.get("desat"):
                I("albedo_desaturation", A.Float, float(v["desat"]))
            if v.get("normal"):
                I("normalmap_texture", A.Asset, Sdf.AssetPath(v["normal"]))
                I("bump_factor", A.Float, float(v.get("bump", 1.0)))
            if v.get("orm") and not v.get("orm_off"):
                I("enable_ORM_texture", A.Bool, True)
                I("ORM_texture", A.Asset, Sdf.AssetPath(v["orm"]))
                I("reflection_roughness_texture_influence", A.Float, 1.0)
            else:
                I("enable_ORM_texture", A.Bool, False)
                I("reflection_roughness_texture_influence", A.Float, 0.0)
            I("reflection_roughness_constant", A.Float, float(v.get("rough", 0.6)))
            I("specular_level", A.Float, float(v.get("specular", 0.5)))
            I("metallic_constant", A.Float, 0.0)
            if v.get("opacity") is not None:          # e.g. an oil film: the floor texture shows through
                I("enable_opacity", A.Bool, True); I("opacity_constant", A.Float, float(v["opacity"]))
                I("opacity_mode", A.Int, 0)
        self.vis[name] = mat
        return mat

    def physics(self, name, p=None):
        if name in self.phys:
            return self.phys[name]
        from pxr import UsdShade, UsdPhysics, PhysxSchema
        p = p or PHYS[name]
        mat = UsdShade.Material.Define(self.stage, f"{self.proot}/{name}")
        m = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        FM = _foot_material()
        if FM.enabled():
            # XFEAS_FOOT_MATERIAL=1: one convention for every surface (foot_material.py). The tile's mu_s / mu_d are
            # the INTENDED contact friction against a rubber sole; stored as multiply factors mu / RUBBER, so
            # contact = RUBBER * factor = the intended value for any robot with the rubber contact material.
            fs, fd = FM.factors(p["mu_s"], p["mu_d"]); combine = "multiply"
        else:      # legacy (all existing terrain results): absolute mu with the 'min' / 'max' combine of the catalog
            fs, fd, combine = float(p["mu_s"]), float(p["mu_d"]), p["combine"]
        m.CreateStaticFrictionAttr().Set(float(fs))
        m.CreateDynamicFrictionAttr().Set(float(fd))
        m.CreateRestitutionAttr().Set(float(p["restitution"]))
        x = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        x.CreateFrictionCombineModeAttr().Set(combine)
        x.CreateRestitutionCombineModeAttr().Set("min")
        if p.get("compliant_k", 0) > 0:
            x.CreateCompliantContactStiffnessAttr().Set(float(p["compliant_k"]))
            x.CreateCompliantContactDampingAttr().Set(float(p["compliant_c"]))
        self.phys[name] = mat
        return mat


def bind(prim, vis=None, phys=None):
    from pxr import UsdShade
    b = UsdShade.MaterialBindingAPI.Apply(prim)
    if vis is not None:
        b.Bind(vis, UsdShade.Tokens.strongerThanDescendants)
    if phys is not None:
        b.Bind(phys, UsdShade.Tokens.strongerThanDescendants, "physics")


# ------------------------------------------------------------------ geometry (Isaac Lab generators)
def _fit(a, nx, ny):
    a = a[:nx, :ny]
    return np.pad(a, ((0, nx - a.shape[0]), (0, ny - a.shape[1])), mode="edge")


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def height_field(geo, seed):
    """sum of Isaac Lab height-field layers -> int16 grid (nx, ny) in units of vscale, tapered at the edges"""
    from isaaclab.terrains.height_field import hf_terrains as H, hf_terrains_cfg as C
    L, W = geo["size"]; hs, vs = geo["hscale"], geo["vscale"]
    nx, ny = int(round(L / hs)) + 1, int(round(W / hs)) + 1
    Z = np.zeros((nx, ny), np.float64)
    taper = True
    for k, ly in enumerate(geo["hf"]):
        np.random.seed((seed + 7919 * k) % (2 ** 31))
        fn, base = ly["fn"], dict(size=(L, W), horizontal_scale=hs, vertical_scale=vs, border_width=0.0)
        if fn == "flat":
            continue
        if fn == "random_uniform":
            cfg = C.HfRandomUniformTerrainCfg(noise_range=tuple(ly["noise"]), noise_step=ly["step"],
                                              downsampled_scale=ly["down"], **base)
            a = H.random_uniform_terrain.__wrapped__(1.0, cfg)
        elif fn == "wave":
            cfg = C.HfWaveTerrainCfg(amplitude_range=(ly["amp"], ly["amp"]), num_waves=ly["waves"], **base)
            a = H.wave_terrain.__wrapped__(1.0, cfg)
        elif fn == "pyramid_slope":
            cfg = C.HfPyramidSlopedTerrainCfg(slope_range=(ly["slope"], ly["slope"]), platform_width=ly["platform"], **base)
            a = H.pyramid_sloped_terrain.__wrapped__(1.0, cfg); taper = False
        elif fn == "discrete_obstacles":
            cfg = C.HfDiscreteObstaclesTerrainCfg(obstacle_width_range=tuple(ly["width"]),
                                                  obstacle_height_range=tuple(ly["height"]), num_obstacles=ly["num"],
                                                  platform_width=0.0, **base)
            a = H.discrete_obstacles_terrain.__wrapped__(1.0, cfg)
        else:
            raise ValueError(fn)
        Z += _fit(np.asarray(a, np.float64), nx, ny)
    if taper and geo.get("zero_mean"):
        # opt-in (run_terrain.py --zero_mean 1): Isaac Lab's random_uniform relief is all >= 0 (noise_range [0, h]),
        # so with the taper every rough tile starts with a net 4-5 cm RISE over its first metre (blind AMO G1 stalls
        # on it). Zero-mean relief keeps the tile's mean level = the approach level. Default off = original tiles.
        Z = Z - Z.mean()
    if taper:
        m = geo.get("flat_margin", 1.0)
        x = np.arange(nx) * hs; y = np.arange(ny) * hs
        w = _smooth(np.minimum(x, L - x) / m)[:, None] * _smooth(np.minimum(y, W - y) / 0.4)[None, :]
        Z = Z * w
    return np.round(Z).astype(np.int16)


def tile_mesh(spec):
    """-> vertices (N,3) tile-local (x 0..L, y -W/2..W/2), faces (M,3), height function (local x, y) -> z"""
    geo = spec["geometry"]; L, W = geo["size"]
    if "mesh" in geo:
        import trimesh
        from isaaclab.terrains.trimesh import mesh_terrains as MT, mesh_terrains_cfg as MC
        g = geo["mesh"]
        np.random.seed(spec["seed"] % (2 ** 31))
        cfg = MC.MeshPyramidStairsTerrainCfg(size=(L, W), step_height_range=(g["step_h"], g["step_h"]),
                                             step_width=g["step_w"], platform_width=g["platform"])
        meshes, _ = MT.pyramid_stairs_terrain(1.0, cfg)
        m = trimesh.util.concatenate(meshes)
        V = np.asarray(m.vertices, float); F = np.asarray(m.faces, np.int64)
        V[:, 1] -= W / 2
        # analytic height of a pyramid of steps (for stance detection / spawning)
        sh, sw, pf = g["step_h"], g["step_w"], g["platform"]

        def h(x, y):
            dx = np.minimum(x, L - x); dy = np.minimum(y + W / 2, W / 2 - y)
            k = np.floor(np.minimum(dx, dy) / sw) + 1
            nmax = min((L - pf) // (2 * sw) + 1, (W - pf) // (2 * sw) + 1)
            return np.where((x >= 0) & (x <= L) & (np.abs(y) <= W / 2), np.minimum(k, nmax) * sh, 0.0)
        # isaaclab's stairs mesh has a ground slab under z=0 -> keep only the top surface for UVs/normals (fine as is)
        return V, F, h
    from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
    Zi = height_field(geo, spec["seed"])
    V, F = convert_height_field_to_mesh(Zi, geo["hscale"], geo["vscale"], None)
    V = np.asarray(V, float); F = np.asarray(F, np.int64)
    V[:, 1] -= W / 2
    Zm = Zi.astype(float) * geo["vscale"]; hs = geo["hscale"]

    def h(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float) + W / 2
        inside = (x >= 0) & (x <= L) & (y >= 0) & (y <= W)
        fx = np.clip(x / hs, 0, Zm.shape[0] - 1.001); fy = np.clip(y / hs, 0, Zm.shape[1] - 1.001)
        i, j = fx.astype(int), fy.astype(int); u, v = fx - i, fy - j
        z = (Zm[i, j] * (1 - u) * (1 - v) + Zm[i + 1, j] * u * (1 - v) + Zm[i, j + 1] * (1 - u) * v + Zm[i + 1, j + 1] * u * v)
        return np.where(inside, z, 0.0)
    return V, F, h


def _vertex_normals(V, F):
    n = np.zeros_like(V)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for k in range(3):
        np.add.at(n, F[:, k], fn)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return n


def make_mesh(stage, path, V, F, uv_m, collider=True, normals=True, double_sided=False, triplanar=False):
    from pxr import UsdGeom, UsdPhysics, Vt, Gf, Sdf
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.reshape(-1).astype(np.int32)))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDoubleSidedAttr(double_sided)
    if normals:
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(_vertex_normals(V, F).astype(np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if triplanar:      # per-face planar projection (no stretched texture on vertical faces, e.g. stair risers)
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        ax = np.argmax(np.abs(fn), axis=1)
        P = V[F]                                                     # (M, 3, 3)
        uv = np.where((ax == 2)[:, None, None], P[:, :, [0, 1]], np.where((ax == 0)[:, None, None], P[:, :, [1, 2]], P[:, :, [0, 2]]))
        st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        st.Set(Vt.Vec2fArray.FromNumpy((uv.reshape(-1, 2) / float(uv_m)).astype(np.float32)))
        normals = False
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9), 3, axis=0).astype(np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    else:
        st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray.FromNumpy((V[:, :2] / float(uv_m)).astype(np.float32)))
    lo, hi = V.min(0), V.max(0)
    m.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    if collider:
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr().Set("none")
    return m


# ------------------------------------------------------------------ dressing
def _blob(rng, a, b, n=48, rough=0.25):
    """irregular closed outline (ellipse a x b with low-frequency radial noise) -> (n,2)"""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.ones(n)
    for k in (2, 3, 5):
        r += rough / k * np.sin(k * t + rng.uniform(0, 2 * np.pi))
    return np.stack([a * r * np.cos(t), b * r * np.sin(t)], 1)


def _fan(P, z):
    """triangle fan over a star-shaped outline P (n,2) at height z(x,y) -> V, F"""
    n = len(P)
    V = np.vstack([[0.0, 0.0], P])
    F = np.array([[0, 1 + i, 1 + (i + 1) % n] for i in range(n)])
    return V, F


def add_rocks(stage, path, rng, D, h, L, W, ox, oy, mats, phys_mat, lane_free=False):
    from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, Usd
    UsdGeom.Xform.Define(stage, path)
    n = 0
    for i in range(D["n"]):
        x = rng.uniform(0.6, L - 0.6); y = rng.uniform(-W / 2 + 0.3, W / 2 - 0.3)
        s = rng.uniform(*D["scale"])
        p = f"{path}/r{i:03d}"
        xf = UsdGeom.Xform.Define(stage, p)
        xf.GetPrim().GetReferences().AddReference(ROCKS[int(rng.integers(len(ROCKS)))])
        z = float(h(np.array(x), np.array(y)))
        sz = 0.01 * s                                     # rock files are in cm, ~0.2 m across at s = 1
        xf.AddTranslateOp().Set(Gf.Vec3d(ox + x, oy + y, z + (0.06 * s if D["dynamic"] else 0.02 * s)))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*map(float, rng.uniform(-25, 25, 2)), float(rng.uniform(0, 360))))
        xf.AddScaleOp().Set(Gf.Vec3f(sz, sz, sz * (1.0 if D["dynamic"] else float(rng.uniform(0.5, 1.0)))))
        for q in Usd.PrimRange(xf.GetPrim()):
            if q.IsInstanceable():
                q.SetInstanceable(False)
        for q in Usd.PrimRange(xf.GetPrim()):
            if q.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(q)
                UsdPhysics.MeshCollisionAPI.Apply(q).CreateApproximationAttr().Set("convexHull")
        if D["dynamic"]:
            UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
            UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateDensityAttr().Set(2400.0)
        bind(xf.GetPrim(), None, phys_mat)
        n += 1
    return n


def add_grass(stage, path, rng, D, h, L, W, ox, oy, mats):
    """procedural blades (5 vertices / 3 triangles each): tufts + a short carpet. Visual only (no collider)."""
    roots, hts, rad = [], [], []
    for _ in range(D["tufts"]):
        c = np.array([rng.uniform(0.2, L - 0.2), rng.uniform(-W / 2 + 0.2, W / 2 - 0.2)])
        k = int(rng.integers(*D["blades_per_tuft"]))
        r = rng.uniform(0, 0.07, k) ** 0.8; t = rng.uniform(0, 2 * np.pi, k)
        roots.append(c + np.stack([r * np.cos(t), r * np.sin(t)], 1))
        hts.append(rng.uniform(D["h"][0] * 1.4, D["h"][1], k)); rad.append(np.stack([np.cos(t), np.sin(t)], 1))
    nc = int(D["carpet_per_m2"] * L * W)
    roots.append(np.stack([rng.uniform(0.05, L - 0.05, nc), rng.uniform(-W / 2 + 0.05, W / 2 - 0.05, nc)], 1))
    hts.append(rng.uniform(0.025, D["h"][0] * 1.2, nc)); t = rng.uniform(0, 2 * np.pi, nc)
    rad.append(np.stack([np.cos(t), np.sin(t)], 1))
    R = np.vstack(roots); H = np.concatenate(hts); O = np.vstack(rad); n = len(R)
    yaw = rng.uniform(0, np.pi, n); wv = np.stack([np.cos(yaw), np.sin(yaw)], 1)   # blade width direction
    wid = rng.uniform(0.004, 0.009, n); lean = rng.uniform(0.15, 0.6, n)
    z0 = h(R[:, 0], R[:, 1]) - 0.005
    V = np.zeros((n, 5, 3))
    for j, (f, ww) in enumerate([(0.0, 1.0), (0.0, -1.0), (0.55, 0.7), (0.55, -0.7), (1.0, 0.0)]):
        off = O * (lean * f * f * H)[:, None]
        V[:, j, :2] = R + off + wv * (wid * ww)[:, None]
        V[:, j, 2] = z0 + H * f * (1 - 0.25 * lean * f)
    V[:, :, 0] += ox; V[:, :, 1] += oy
    base = (np.arange(n) * 5)[:, None]
    F = (np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4]])[None] + base[:, :, None]).reshape(-1, 3)
    m = make_mesh(stage, path, V.reshape(-1, 3), F, VIS[D["vis"]]["uv_m"], collider=False, normals=False, double_sided=True)
    bind(m.GetPrim(), mats.visual(D["vis"]))
    return n


def _tube(P, R, nring=10):
    """swept tube along polyline P (n,3) with radii R (n,) -> V, F, st (u around, v along, in metres)"""
    n = len(P); V, UV = [], []
    T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    up = np.array([0.0, 0.0, 1.0]); s_acc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    for i in range(n):
        nx = np.cross(T[i], up); nx /= max(np.linalg.norm(nx), 1e-9); ny = np.cross(nx, T[i])
        for k in range(nring + 1):
            a = 2 * np.pi * k / nring
            V.append(P[i] + R[i] * (math.cos(a) * nx + math.sin(a) * ny)); UV.append([a * R[i], s_acc[i]])
    F = []
    for i in range(n - 1):
        for k in range(nring):
            p0 = i * (nring + 1) + k; p1 = p0 + 1; q0 = p0 + nring + 1; q1 = q0 + 1
            F += [[p0, q0, p1], [p1, q0, q1]]
    return np.array(V), np.array(F), np.array(UV)


def add_roots(stage, path, rng, D, h, L, W, ox, oy, mats):
    """surface tree roots: tapered, wandering bark-textured tubes crossing the lane, ~1/3 buried, exact colliders"""
    from pxr import UsdGeom, UsdPhysics, Vt, Sdf
    UsdGeom.Xform.Define(stage, path)
    vis, phys = mats.visual(D["vis"]), mats.physics(D["phys"])
    for i in range(D["n"]):
        x = rng.uniform(1.6, L - 1.3); y = rng.uniform(-W / 2, -W / 2 + 1.0) if rng.random() < 0.5 else rng.uniform(W / 2 - 1.0, W / 2)
        ang = (np.pi / 2 if y < 0 else -np.pi / 2) + rng.uniform(-0.6, 0.6)
        r0 = rng.uniform(*D["dia"]) / 2; ln = rng.uniform(2.0, 5.0); n = int(ln / 0.08)
        P = [np.array([x, y])]; a_ = ang
        for k in range(n - 1):
            a_ += rng.normal(0, 0.08); P.append(P[-1] + 0.08 * np.array([math.cos(a_), math.sin(a_)]))
        P = np.array(P); R = r0 * np.linspace(1.0, 0.25, n) * (1 + 0.12 * np.sin(np.linspace(0, rng.uniform(6, 14), n)))
        Z = h(P[:, 0], P[:, 1]) + R * 0.35
        P3 = np.column_stack([P[:, 0] + ox, P[:, 1] + oy, Z])
        V, F, UV = _tube(P3, R)
        m = make_mesh(stage, f"{path}/root{i:02d}", V, F, 1.0)
        UsdGeom.PrimvarsAPI(m).GetPrimvar("st").Set(Vt.Vec2fArray.FromNumpy((UV / VIS[D["vis"]]["uv_m"]).astype(np.float32)))
        bind(m.GetPrim(), vis, phys)
    return int(D["n"])


DEBRIS = sorted(os.path.join(VEG, "Debris", f) for f in os.listdir(os.path.join(VEG, "Debris")) if f.endswith(".usd"))


def add_debris(stage, path, rng, D, h, L, W, ox, oy, mats):
    """leaf-litter clusters (NVIDIA Vegetation/Debris), scaled to a footprint of size m. Visual only."""
    from pxr import UsdGeom, Gf, Usd
    UsdGeom.Xform.Define(stage, path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    for i in range(D["n"]):
        f = DEBRIS[int(rng.integers(len(DEBRIS)))]
        xf = UsdGeom.Xform.Define(stage, f"{path}/d{i:02d}")
        inner = UsdGeom.Xform.Define(stage, f"{path}/d{i:02d}/o")
        ref = UsdGeom.Xform.Define(stage, f"{path}/d{i:02d}/o/a")
        ref.GetPrim().GetReferences().AddReference(f)
        bb = cache.ComputeRelativeBound(ref.GetPrim(), inner.GetPrim()).ComputeAlignedRange()
        ext = bb.GetSize(); foot = max(ext[0], ext[1], 1e-6)
        s = rng.uniform(*D["size"]) / foot
        c = bb.GetMidpoint()
        x, y = rng.uniform(0.3, L - 0.3), rng.uniform(-W / 2 + 0.2, W / 2 - 0.2)
        z = float(h(np.array(x), np.array(y)))
        xf.AddTranslateOp().Set(Gf.Vec3d(ox + x, oy + y, z + 0.003))
        xf.AddRotateZOp().Set(float(rng.uniform(0, 360)))
        xf.AddScaleOp().Set(Gf.Vec3f(s, s, s))
        inner.AddTranslateOp().Set(Gf.Vec3d(-c[0], -c[1], -bb.GetMin()[2]))
    return int(D["n"])


def add_puddles(stage, path, rng, D, h, L, W, ox, oy, mats):
    """water/mud puddles: flat irregular discs set at the 40th height percentile inside them -> the terrain cuts
    organic shores. Visual only (no collider): the feel is the mud material underneath."""
    from pxr import UsdGeom
    UsdGeom.Xform.Define(stage, path)
    vis = mats.visual(D["vis"])
    for i in range(D["n"]):
        cx, cy = rng.uniform(1.2, L - 1.2), rng.uniform(-W / 2 + 0.6, W / 2 - 0.6)
        r = rng.uniform(*D["r"])
        P = _blob(rng, r, r * rng.uniform(0.5, 1.0), rough=0.35)
        th = rng.uniform(0, np.pi); c, s = math.cos(th), math.sin(th)
        P = P @ np.array([[c, s], [-s, c]])
        xs = cx + P[:, 0] * rng.uniform(0, 1, len(P)); ys = cy + P[:, 1] * rng.uniform(0, 1, len(P))
        z = float(np.percentile(h(xs, ys), 40)) + 0.004
        V2, F = _fan(P, z)
        V = np.column_stack([V2[:, 0] + ox + cx, V2[:, 1] + oy + cy, np.full(len(V2), z)])
        m = make_mesh(stage, f"{path}/p{i:02d}", V, F, VIS[D["vis"]]["uv_m"], collider=False)
        bind(m.GetPrim(), vis)


def add_slick(stage, path, rng, D, h, L, W, ox, oy, mats):
    """oil slick across the lane: thin irregular sheet 2 mm above the floor WITH a collider + the oil PhysX material,
    so the slippery footprint is exactly the visible one (concrete around it keeps the tile's firm material)."""
    from pxr import UsdGeom
    a, b = D["length"] / 2, D["width"] / 2
    cx = L / 2
    UsdGeom.Xform.Define(stage, path)
    blobs = [(cx, 0.0, a, b, 0.35)]
    for k in range(int(rng.integers(4, 9))):                    # satellite drips / splashes around the main pool
        ang = rng.uniform(0, 2 * np.pi); rr = rng.uniform(0.8, 1.25)
        blobs.append((cx + rr * a * math.cos(ang), rr * b * math.sin(ang), *(rng.uniform(0.1, 0.45, 2)), 0.5))
    for k, (bx, by, ba, bb, ro) in enumerate(blobs):
        P = _blob(rng, ba, bb, n=160, rough=ro)
        for f_ in (7, 11, 17):                                   # ragged, fractal-ish rim
            t_ = np.arctan2(P[:, 1], P[:, 0])
            P *= (1 + 0.05 * np.sin(f_ * t_ + rng.uniform(0, 6.3)))[:, None]
        V2, F = _fan(P, 0)
        z = np.maximum(h(bx + V2[:, 0], by + V2[:, 1]), 0) + 0.002 + 0.0005 * k
        V = np.column_stack([V2[:, 0] + ox + bx, V2[:, 1] + oy + by, z])
        m = make_mesh(stage, f"{path}/s{k:02d}", V, F, VIS[D["vis"]]["uv_m"], collider=True)
        bind(m.GetPrim(), mats.visual(D["vis"]), mats.physics(D["phys"]))
    return dict(x0=cx - a * 1.4, x1=cx + a * 1.4, n_blobs=len(blobs))


# ------------------------------------------------------------------ field
def build_field(stage, spec, ground=True):
    from pxr import UsdGeom
    mats = Materials(stage)
    UsdGeom.Xform.Define(stage, "/World/Terrain")
    if ground:   # neutral surroundings (firm, matte asphalt) at z = 0, with the tile footprints left open
        rects = []
        ys = sorted((t["origin"][1] - t["geometry"]["size"][1] / 2, t["origin"][1] + t["geometry"]["size"][1] / 2)
                    for t in spec["tiles"])
        L = max(t["geometry"]["size"][0] for t in spec["tiles"])
        x0, x1, y0, y1 = -15.0, L + 15.0, ys[0][0] - 15.0, ys[-1][1] + 15.0
        rects += [(x0, 0.0, y0, y1), (L, x1, y0, y1)]
        prev = y0
        for a, b in ys:
            rects.append((0.0, L, prev, a)); prev = b
        rects.append((0.0, L, prev, y1))
        V, F = [], []
        for (a0, a1, b0, b1) in rects:
            if a1 - a0 < 1e-6 or b1 - b0 < 1e-6:
                continue
            k = len(V); V += [[a0, b0, 0.0], [a1, b0, 0.0], [a1, b1, 0.0], [a0, b1, 0.0]]
            F += [[k, k + 1, k + 2], [k, k + 2, k + 3]]
        g = make_mesh(stage, "/World/Terrain/Ground", np.array(V, float), np.array(F), 3.0)
        gv = dict(VIS["asphalt"]); gv["tint"] = (0.6, 0.6, 0.6)
        bind(g.GetPrim(), mats.visual("ground_asphalt", gv), mats.physics("firm"))
    out = []
    for t in spec["tiles"]:
        rng = np.random.default_rng(t["seed"])
        L, W = t["geometry"]["size"]; ox, oy, oz = t["origin"]
        V, F, h_local = tile_mesh(t)
        Vw = V + np.array([ox, oy, oz])
        root = f"/World/Terrain/{t['id']}"
        UsdGeom.Xform.Define(stage, root)
        m = make_mesh(stage, root + "/Ground", Vw, F, t["visual"]["uv_m"], triplanar="mesh" in t["geometry"])
        bind(m.GetPrim(), mats.visual(t["visual"]["name"]), mats.physics(t["physics"]["name"]))
        info = dict(id=t["id"], type=t["type"], bucket=t["bucket"], origin=t["origin"], size=[L, W],
                    n_tris=int(len(F)), z_max=round(float(V[:, 2].max()), 3))
        for k, D in enumerate(t["dressing"]):
            p = f"{root}/{D['kind']}{k}"
            if D["kind"] == "rocks":
                info["rocks"] = add_rocks(stage, p, rng, D, h_local, L, W, ox, oy, mats, mats.physics(t["physics"]["name"]))
            elif D["kind"] == "grass":
                info["blades"] = add_grass(stage, p, rng, D, h_local, L, W, ox, oy, mats)
            elif D["kind"] == "roots":
                info["root_segments"] = add_roots(stage, p, rng, D, h_local, L, W, ox, oy, mats)
            elif D["kind"] == "debris":
                info["debris"] = add_debris(stage, p, rng, D, h_local, L, W, ox, oy, mats)
            elif D["kind"] == "puddles":
                add_puddles(stage, p, rng, D, h_local, L, W, ox, oy, mats)
            elif D["kind"] == "slick":
                info["slick"] = add_slick(stage, p, rng, D, h_local, L, W, ox, oy, mats)
        hw = (lambda hl, ox_, oy_, oz_: (lambda x, y: hl(np.asarray(x) - ox_, np.asarray(y) - oy_) + oz_))(h_local, ox, oy, oz)
        info["height_at"] = hw
        out.append(info)
    return out, mats
