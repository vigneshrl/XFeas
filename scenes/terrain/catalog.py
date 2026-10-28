"""
TERRAIN CATALOG (pure python, seedable, no Isaac): every terrain tile is ONE JSON object
    {type, bucket, seed, difficulty, size, geometry, visual, physics, dressing}
so a tile is reproducible from its spec alone. scenes/terrain/build.py turns a spec into USD in Isaac
(geometry through Isaac Lab's isaaclab.terrains height-field / trimesh generators, PBR look from the NVIDIA
material library mirrored under Collected_isaac_envs/_mirror, PhysX material bound with purpose "physics").

Buckets
  Every surface is a photoreal PBR texture from the NVIDIA library; dressing = NVIDIA Vegetation assets (rocks,
  leaf-litter clusters) or textured procedural meshes (grass blades, bark-textured roots). No flat-colour shapes.
  rough     : dirt_bumpy, grass_tufts, gravel_loose, rubble_rocky, sand_dunes, slope_pyramid, stairs_pyramid,
              roots_leaves
  slippery  : mud_wet, sticky_clay, sticky_rubber, oil_spill, wet_tile, ice

Physics notes (PhysX contacts are rigid; see analysis/TERRAIN_ISAACLAB.md sections 3 and 8):
  * TWO conventions, chosen by the env var XFEAS_FOOT_MATERIAL (isaac_sim/foot_material.py):
      =1 (new data / benchmarks): every material uses combine 'multiply'; mu_s / mu_d below are the INTENDED contact
         friction against a rubber sole (0.9 / 0.8) and are stored as factors mu / rubber, so they apply exactly
         (verified by scripts/terrain/friction_check.py). The 'combine' field below is then ignored.
      =0 / unset (legacy, all results before 2026-10-10): the notes below.
  * mu_s / mu_d are the TILE's static / dynamic friction. combine = PhysX friction-combine mode of the tile material.
    Legacy: Spot's / G1's feet have NO material (PhysX default mu 0.5, 'average'); materials shipped inside a robot
    are forced to 'min' (robots.py:_fix_contact_materials). PhysX picks the mode of
    HIGHER priority (max > multiply > min > average) of the two materials, so
      - combine 'min' : contact mu = min(tile, foot)  -> a LOW tile mu always wins (oil, ice, wet tile, mud)
      - combine 'max' : contact mu = max(tile, foot)  -> a HIGH tile mu always wins (sticky)
    'average' / 'multiply' on a tile would be overridden by the feet's 'min', so they are not used.
  * compliant_k / compliant_c (N/m, N s/m): PhysxMaterialAPI compliantContactStiffness / Damping. With k > 0 the
    contact is a spring-damper instead of a hard constraint: feet sink a little and the damper dissipates energy
    (mud / clay / grass / rubber). 0 = rigid.
  * No runtime "drag zone" is used: everything that makes a tile feel different lives in the USD physics
    material, so any consumer (episode_runner, gensim) gets the same feel by loading the tile.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import json, math
import numpy as np

MIRROR = xfeas_paths.ISAAC_ASSETS
MAT = MIRROR + "/Assets/Isaac/5.1/NVIDIA/Materials"
VEG = MIRROR + "/Assets/Isaac/5.1/NVIDIA/Assets/Vegetation"

BUCKETS = dict(rough="1_rough_terrain", slippery="3_slippery_surfaces")
TILE_SIZE = (12.0, 8.0)          # x (walking direction) x y, metres
HSCALE = 0.05                    # height-field horizontal resolution (m)
VSCALE = 0.002                   # height-field vertical resolution (m)


def lerp(r, d):
    return float(r[0] + (r[1] - r[0]) * d)


def r3(x):
    return round(float(x), 4)


# ------------------------------------------------------------------ visual materials (PBR textures, real scale)
def _pick(d, names):
    """first existing texture among the library's naming variants (BaseColor/baseColor, Normal/N)"""
    import os
    for n in names:
        if os.path.exists(d + n + ".png"):
            return d + n + ".png"
    return d + names[0] + ".png"


def _pbr(folder, base, prefix=None, **kw):
    d = f"{MAT}/{folder}/{base}/{prefix or base}_"
    v = dict(kind="omnipbr", diffuse=_pick(d, ["BaseColor", "baseColor"]), normal=_pick(d, ["Normal", "N"]),
             orm=_pick(d, ["ORM"]))
    v.update(kw)
    return v


def _vmat(rel, sub, **kw):
    v = dict(kind="mdl", mdl=f"{MAT}/{rel}", sub=sub)
    v.update(kw)
    return v


# uv_m = metres of ground covered by one texture repeat (st = xy / uv_m)
VIS = {
    "dirt":          _pbr("Base/Natural", "Dirt", uv_m=2.0),
    "soil_rocky":    _pbr("Base/Natural", "Soil_Rocky", uv_m=2.0),
    "grass":         _pbr("Base/Natural", "Grass_Countryside", uv_m=2.0),
    "grass_blade":   _pbr("Base/Natural", "Grass_Cut", uv_m=1.5, tint=(0.85, 1.0, 0.75)),
    "sand":          _pbr("Base/Natural", "Sand", uv_m=2.5),
    "asphalt":       _pbr("Base/Natural", "Asphalt", uv_m=3.0),
    "leaves":        _vmat("vMaterials_2/Ground/Ground_Leaves_Oak.mdl", "Ground_Leaves_Oak", uv_m=1.0),
    "gravel":        _pbr("Base/Stone", "Gravel", uv_m=0.9),
    "pebble":        _pbr("Base/Natural", "Soil_Rocky", uv_m=0.3, tint=(0.8, 0.78, 0.74)),
    "fieldstone":    _pbr("Base/Stone", "Fieldstone", uv_m=2.0),
    "concrete":      _pbr("Base/Masonry", "Concrete_Rough", uv_m=3.0, tint=(0.7, 0.7, 0.7)),
    "concrete_smooth": _pbr("Base/Masonry", "Concrete_Smooth", uv_m=3.0),
    "bark":          dict(kind="omnipbr", diffuse=f"{MAT}/vMaterials_2/Wood/textures/bark_oak_diff.jpg",
                          normal=f"{MAT}/vMaterials_2/Wood/textures/bark_oak_norm.jpg", orm=None, rough=0.8, uv_m=0.5,
                          tint=(0.42, 0.31, 0.22), bump=1.5),
    # ---- slippery / sticky looks (textures of the library, roughness / tint overridden for the wet / glossy state)
    "mud_wet":       _pbr("Base/Natural", "Sand", uv_m=1.2, tint=(0.20, 0.15, 0.10), rough=0.16, orm_off=True, bump=1.6),
    "mud_puddle":    dict(kind="omnipbr", diffuse=f"{MAT}/Base/Natural/Sand/Sand_BaseColor.png",
                          normal=f"{MAT}/Base/Natural/Water/Water_Normal.png", orm=None,
                          tint=(0.10, 0.075, 0.05), rough=0.03, uv_m=1.2, bump=0.25, specular=0.5),
    "clay":          _pbr("Base/Natural", "Dirt", uv_m=1.5, tint=(1.0, 0.70, 0.55), rough=0.5, orm_off=True, bump=0.8),
    "rubber_track":  _vmat("vMaterials_2/Ground/Ground_Hard_Court.mdl", "Ground_Hard_Court_Red_Desaturated", uv_m=1.0),
    "oil":           dict(kind="omnipbr", diffuse=f"{MAT}/Base/Masonry/Concrete_Smooth/Concrete_Smooth_BaseColor.png",
                          normal=f"{MAT}/Base/Natural/Water/Water_Normal.png", orm=None,
                          tint=(0.22, 0.19, 0.15), rough=0.08, uv_m=3.0, bump=0.2, specular=0.6),
    "wet_tile":      _pbr("Base/Stone", "Slate", prefix="Slate_Tile", uv_m=1.5, rough=0.05, orm_off=True, tint=(0.85, 0.85, 0.85)),
    "ice":           _pbr("Base/Masonry", "Concrete_Polished", uv_m=4.0, tint=(0.55, 0.68, 0.80), rough=0.03,
                          orm_off=True, desat=0.8, bump=0.25),
}

# ------------------------------------------------------------------ physics materials
PHYS = {
    "firm":    dict(mu_s=0.9, mu_d=0.8, restitution=0.0, combine="min", compliant_k=0.0, compliant_c=0.0),
    "grass":   dict(mu_s=0.6, mu_d=0.5, restitution=0.0, combine="min", compliant_k=4.0e4, compliant_c=6.0e2),
    "gravel":  dict(mu_s=0.55, mu_d=0.45, restitution=0.0, combine="min", compliant_k=0.0, compliant_c=0.0),
    "sand":    dict(mu_s=0.55, mu_d=0.45, restitution=0.0, combine="min", compliant_k=3.0e4, compliant_c=1.0e3),
    "leaves":  dict(mu_s=0.5, mu_d=0.4, restitution=0.0, combine="min", compliant_k=0.0, compliant_c=0.0),
    "mud":     dict(mu_s=0.25, mu_d=0.2, restitution=0.0, combine="min", compliant_k=1.5e4, compliant_c=2.5e3),
    "clay":    dict(mu_s=1.8, mu_d=1.6, restitution=0.0, combine="max", compliant_k=1.5e4, compliant_c=3.0e3),
    "rubber":  dict(mu_s=1.6, mu_d=1.4, restitution=0.0, combine="max", compliant_k=5.0e4, compliant_c=1.5e3),
    "oil":     dict(mu_s=0.088, mu_d=0.088, restitution=0.0, combine="min", compliant_k=0.0, compliant_c=0.0),
    "wet_tile": dict(mu_s=0.2, mu_d=0.15, restitution=0.0, combine="min", compliant_k=0.0, compliant_c=0.0),
    "ice":     dict(mu_s=0.05, mu_d=0.03, restitution=0.0, combine="min", compliant_k=0.0, compliant_c=0.0),
}

# ------------------------------------------------------------------ terrain types
# geometry: list of height-field layers (Isaac Lab hf generators, summed) OR one trimesh generator.
#   ranges are (easy, hard) and interpolated by the tile's difficulty, like Isaac Lab's curriculum.
TYPES = {
    "dirt_bumpy": dict(bucket="rough", vis="dirt", phys="firm",
        hf=[dict(fn="random_uniform", noise=((0.0, 0.03), (0.0, 0.10)), step=0.01, down=0.25),
            dict(fn="random_uniform", noise=((0.0, 0.01), (0.0, 0.02)), step=0.005, down=None)],
        dressing=[dict(kind="rocks", n=(10, 30), scale=(0.25, 0.6), dynamic=False)]),
    "grass_tufts": dict(bucket="rough", vis="grass", phys="grass",
        hf=[dict(fn="wave", amp=(0.02, 0.10), waves=2),
            dict(fn="random_uniform", noise=((0.0, 0.03), (0.0, 0.07)), step=0.01, down=0.5)],
        dressing=[dict(kind="grass", tufts=(450, 800), blades_per_tuft=(25, 50), carpet_per_m2=(150, 250),
                       h=(0.06, 0.22))]),
    "gravel_loose": dict(bucket="rough", vis="gravel", phys="gravel",
        hf=[dict(fn="random_uniform", noise=((0.0, 0.02), (0.0, 0.05)), step=0.005, down=None),
            dict(fn="random_uniform", noise=((0.0, 0.02), (0.0, 0.04)), step=0.01, down=0.4)],
        dressing=[dict(kind="rocks", n=(120, 220), scale=(0.12, 0.3), dynamic=True)]),
    "rubble_rocky": dict(bucket="rough", vis="soil_rocky", phys="firm",
        hf=[dict(fn="random_uniform", noise=((0.0, 0.04), (0.0, 0.10)), step=0.01, down=0.35),
            dict(fn="random_uniform", noise=((0.0, 0.01), (0.0, 0.03)), step=0.004, down=None)],
        dressing=[dict(kind="rocks", n=(60, 140), scale=(0.5, 2.0), dynamic=False)]),
    "sand_dunes": dict(bucket="rough", vis="sand", phys="sand",
        hf=[dict(fn="wave", amp=(0.06, 0.25), waves=3),
            dict(fn="random_uniform", noise=((0.0, 0.01), (0.0, 0.03)), step=0.005, down=0.6)],
        dressing=[]),
    "slope_pyramid": dict(bucket="rough", vis="asphalt", phys="firm",
        hf=[dict(fn="pyramid_slope", slope=(0.08, 0.35), platform=2.0)], dressing=[]),
    "stairs_pyramid": dict(bucket="rough", vis="concrete", phys="firm",
        mesh=dict(fn="pyramid_stairs", step_h=(0.04, 0.16), step_w=0.32, platform=2.0), dressing=[]),
    "roots_leaves": dict(bucket="rough", vis="leaves", phys="leaves",
        hf=[dict(fn="random_uniform", noise=((0.0, 0.02), (0.0, 0.05)), step=0.01, down=0.4)],
        dressing=[dict(kind="roots", n=(8, 18), dia=((0.05, 0.09), (0.09, 0.16))),
                  dict(kind="debris", n=(40, 80), size=(0.3, 0.9)),
                  dict(kind="rocks", n=(5, 12), scale=(0.3, 0.6), dynamic=False)]),
    # ---------------- slippery / sticky
    "mud_wet": dict(bucket="slippery", vis="mud_wet", phys="mud",
        hf=[dict(fn="random_uniform", noise=((0.0, 0.03), (0.0, 0.07)), step=0.01, down=0.5),
            dict(fn="random_uniform", noise=((0.0, 0.006), (0.0, 0.012)), step=0.003, down=None)],
        dressing=[dict(kind="puddles", n=(6, 14), r=(0.25, 0.9), vis="mud_puddle")]),
    "sticky_clay": dict(bucket="slippery", vis="clay", phys="clay",
        hf=[dict(fn="random_uniform", noise=((0.0, 0.02), (0.0, 0.05)), step=0.01, down=0.4),
            dict(fn="random_uniform", noise=((0.0, 0.005), (0.0, 0.01)), step=0.0025, down=None)],
        dressing=[]),
    "sticky_rubber": dict(bucket="slippery", vis="rubber_track", phys="rubber",
        hf=[dict(fn="flat")], dressing=[]),
    "oil_spill": dict(bucket="slippery", vis="concrete_smooth", phys="firm",
        hf=[dict(fn="flat")],
        dressing=[dict(kind="slick", vis="oil", phys="oil", length=(4.0, 6.5), width=(1.6, 3.0))]),
    "wet_tile": dict(bucket="slippery", vis="wet_tile", phys="wet_tile", hf=[dict(fn="flat")], dressing=[]),
    "ice": dict(bucket="slippery", vis="ice", phys="ice",
        hf=[dict(fn="random_uniform", noise=((0.0, 0.004), (0.0, 0.01)), step=0.002, down=0.8)], dressing=[]),
}
ROUGH = [k for k, v in TYPES.items() if v["bucket"] == "rough"]
SLIPPERY = [k for k, v in TYPES.items() if v["bucket"] == "slippery"]


def _rng_range(rng, r):
    return float(rng.uniform(*r))


def sample_tile(rng, ttype, difficulty=None, size=TILE_SIZE):
    """One tile spec. Seed of the tile = an int drawn from rng, so the build is reproducible from the JSON alone."""
    T = TYPES[ttype]
    d = float(rng.uniform(0.2, 1.0)) if difficulty is None else float(difficulty)
    geo = dict(size=list(size), hscale=HSCALE, vscale=VSCALE, border=0.0, flat_margin=1.0)
    if "mesh" in T:
        m = T["mesh"]
        geo["mesh"] = dict(fn=m["fn"], step_h=r3(lerp(m["step_h"], d)), step_w=m["step_w"], platform=m["platform"])
    else:
        layers = []
        for L in T["hf"]:
            fn = L["fn"]
            if fn == "random_uniform":
                hi = lerp((L["noise"][0][1], L["noise"][1][1]), d)
                layers.append(dict(fn=fn, noise=[0.0, r3(hi)], step=L["step"], down=L["down"]))
            elif fn == "wave":
                layers.append(dict(fn=fn, amp=r3(lerp(L["amp"], d)), waves=L["waves"]))
            elif fn == "pyramid_slope":
                layers.append(dict(fn=fn, slope=r3(lerp(L["slope"], d)), platform=L["platform"]))
            elif fn == "discrete_obstacles":
                layers.append(dict(fn=fn, width=list(L["width"]),
                                   height=[r3(lerp((L["height"][0][0], L["height"][1][0]), d)),
                                           r3(lerp((L["height"][0][1], L["height"][1][1]), d))],
                                   num=int(round(lerp(L["num"], d)))))
            else:
                layers.append(dict(fn=fn))
        geo["hf"] = layers
    dress = []
    for D in T["dressing"]:
        k = D["kind"]
        if k == "rocks":
            dress.append(dict(kind=k, n=int(round(lerp(D["n"], d))), scale=list(D["scale"]), dynamic=D["dynamic"]))
        elif k == "grass":
            dress.append(dict(kind=k, tufts=int(round(lerp(D["tufts"], d))), blades_per_tuft=list(D["blades_per_tuft"]),
                              carpet_per_m2=int(round(lerp(D["carpet_per_m2"], d))), h=list(D["h"]), vis="grass_blade"))
        elif k == "roots":
            dress.append(dict(kind=k, n=int(round(lerp(D["n"], d))),
                              dia=[r3(lerp((D["dia"][0][0], D["dia"][1][0]), d)), r3(lerp((D["dia"][0][1], D["dia"][1][1]), d))],
                              vis="bark", phys="firm"))
        elif k == "debris":
            dress.append(dict(kind=k, n=int(round(lerp(D["n"], d))), size=list(D["size"])))
        elif k == "puddles":
            dress.append(dict(kind=k, n=int(round(lerp(D["n"], d))), r=list(D["r"]), vis=D["vis"]))
        elif k == "slick":
            dress.append(dict(kind=k, vis=D["vis"], phys=D["phys"], length=r3(_rng_range(rng, D["length"])),
                              width=r3(_rng_range(rng, D["width"]))))
    return dict(type=ttype, bucket=T["bucket"], seed=int(rng.integers(1 << 30)), difficulty=r3(d),
                geometry=geo, visual=dict(name=T["vis"], **VIS[T["vis"]]), physics=dict(name=T["phys"], **PHYS[T["phys"]]),
                dressing=dress)


def field_spec(seed, types=None, difficulty=None, spacing=14.0, size=TILE_SIZE):
    """A FIELD = tiles side by side along +y (each tile is a lane along +x). Returns a JSON-able dict."""
    rng = np.random.default_rng(seed)
    types = types or (ROUGH + SLIPPERY)
    tiles = []
    for i, t in enumerate(types):
        s = sample_tile(rng, t, difficulty, size)
        s.update(id=f"{t}_{i:02d}", origin=[0.0, r3(i * spacing), 0.0])
        tiles.append(s)
    return dict(seed=int(seed), spacing=spacing, tiles=tiles)


if __name__ == "__main__":
    import sys
    print(json.dumps(field_spec(int(sys.argv[1]) if len(sys.argv) > 1 else 0, difficulty=0.6), indent=1)[:3000])
