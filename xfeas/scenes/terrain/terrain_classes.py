#!/usr/bin/env python
"""
THE terrain class definitions -- single source of truth.

This module exists because of a specific failure. The demo corridor was authored with its own
colour table (saturated blues/browns, chosen to fight a rendering problem) while the training
data used these neutral school-floor greys. Measured on the rendered floor pixels:

    training (terr_s0)   mean RGB [0.365 0.347 0.333]   saturation 0.052
    demo corridor        mean RGB 0.13 .. 0.98          saturation 0.075 .. 0.301

Our model is trained on the first and was asked to score the second. It extrapolated and
pinned at P(fail) ~ 1.0 for every class, so the action-conditioned map degenerated to
"always slow" and lost the closed-loop run -- while its OWN held-out predictions on real
school terrain spread 0.014 .. 1.00 with a median of 0.49. The model was fine; the demo was
out of distribution.

So collection and demo now import the SAME numbers. Any demo built from this module is in
distribution by construction. Visibility problems must be solved with LIGHTING, never by
changing these values.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)

# ---------------------------------------------------------------------------------------
# APPEARANCE, v2 -- re-authored 2026-08-18, and the training data RE-COLLECTED with it.
#
# v1 authored these as linear albedos around 0.42-0.62. Measured on the labelled 48 px crops
# the model actually trains on, that rendered as:
#
#     gravel  mean RGB [0.502 0.497 0.480]      rubble  [0.504 0.514 0.501]
#
# i.e. pale near-white pads. Debris that is meant to read as broken concrete looked like
# white foam, and a demo built faithfully from these numbers inherits the same look. The
# earlier attempt to fix it in the DEMO ONLY -- by re-authoring the demo's colours to
# saturated hues -- is what put the demo 6x outside the training distribution and pinned our
# model at P(fail) ~ 1.0, losing the closed-loop run.
#
# So the appearance is fixed HERE, once, and the training set is re-collected from it. The
# empirical transfer under the school's own lights is display ~= 0.50 at linear 0.42, so a
# target display of ~0.28 (dirty concrete rather than foam) needs linear ~0.12.
#
# PHYSICS IS DELIBERATELY UNCHANGED. Every friction value below is identical to v1, so the
# outcomes -- and therefore the labels -- are directly comparable with the v1 results.
# What changes is only what the camera sees.
#
# The brightness ordering that the friction benchmark relies on (drier = brighter, more
# slippery = darker) is preserved.
# ---------------------------------------------------------------------------------------

# (name, friction range, base colour, colour jitter)
FLAT = [
    ("dry", (0.80, 0.95), (0.20, 0.18, 0.15), 0.045),
    ("dusty", (0.45, 0.60), (0.24, 0.22, 0.18), 0.045),
    ("wet", (0.28, 0.40), (0.08, 0.09, 0.11), 0.025),
    ("contaminated", (0.10, 0.16), (0.06, 0.06, 0.07), 0.020),
    ("oil", (0.05, 0.11), (0.03, 0.03, 0.04), 0.015),
]

# (name, n_pieces, size range (m), height range (m), colour, friction)
ROUGH = [
    ("gravel", 90, (0.03, 0.06), (0.02, 0.04), (0.13, 0.12, 0.11), 0.7),
    ("rubble", 26, (0.09, 0.17), (0.05, 0.11), (0.16, 0.14, 0.12), 0.7),
    ("slabs", 7, (0.35, 0.55), (0.05, 0.12), (0.15, 0.15, 0.14), 0.7),
    ("obstacles", 9, (0.14, 0.22), (0.10, 0.18), (0.12, 0.11, 0.10), 0.7),
    ("unstable", 40, (0.06, 0.12), (0.03, 0.07), (0.14, 0.13, 0.10), 0.22),
]

CLASS_NAMES = [c[0] for c in FLAT] + [c[0] for c in ROUGH]

# Per-piece colour jitter for the rough classes, as a FRACTION of the base albedo. It used to
# be an absolute +-0.06 in both the collector and the demo, which was ~14% against the v1 base
# of 0.42 but ~46% against the v2 base of 0.13 -- gravel came out as pastel confetti rather
# than one dirty material with tonal variation.
PIECE_JITTER = 0.22

# ACTIONS: [mode(0 fwd, 1 retreat), speed]
ACTIONS = [(0, 0.5), (0, 0.9), (1, 0.5)]

# Specular roughness used when authoring the flat materials. Wet-looking classes are shiny.
SHINY = ("oil", "wet", "contaminated")


def flat_by_name(n):
    return next(c for c in FLAT if c[0] == n)


def rough_by_name(n):
    return next(c for c in ROUGH if c[0] == n)


def is_rough(n):
    return any(c[0] == n for c in ROUGH)


# ---------------------------------------------------------------------------------------
# MATERIAL AUTHORING -- shared by the collector and the demo.
#
# These used to be two separate functions: the collector authored UsdPreviewSurface and the
# demo authored OmniPBR. Same numbers, different shader, different pixels -- a silent drift
# channel between training and demo appearance on top of the colour drift. Both now call
# this, so a panel is in distribution by construction.
#
# OmniPBR (MDL) is the shader this RTX path actually consumes; a slab authored as
# UsdPreviewSurface with diffuseColor 0.04 still rendered cream.
# ---------------------------------------------------------------------------------------
def author_material(stage, prim, col, rough):
    from pxr import UsdGeom, UsdShade, Gf, Sdf

    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*[float(c) for c in col])])
    mpath = f"{prim.GetPath()}_M"
    vm = UsdShade.Material.Define(stage, mpath)
    sh = UsdShade.Shader.Define(stage, mpath + "/Shader")
    sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateIdAttr("OmniPBR")
    sh.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in col]))
    sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(rough))
    sh.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
    vm.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    vm.CreateDisplacementOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    vm.CreateVolumeOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI.Apply(prim)
    UsdShade.MaterialBindingAPI(prim).Bind(vm, UsdShade.Tokens.strongerThanDescendants)


def author_physics(stage, prim, fr, path):
    from pxr import UsdPhysics, UsdShade

    try:
        from pxr import PhysxSchema
    except Exception:
        PhysxSchema = None
    UsdPhysics.CollisionAPI.Apply(prim)
    mp = UsdShade.Material.Define(stage, path).GetPrim()
    UsdPhysics.MaterialAPI.Apply(mp)
    m = UsdPhysics.MaterialAPI(mp)
    m.CreateStaticFrictionAttr(fr)
    m.CreateDynamicFrictionAttr(fr * 0.8)
    m.CreateRestitutionAttr(0.0)
    if PhysxSchema:
        PhysxSchema.PhysxMaterialAPI.Apply(mp).CreateFrictionCombineModeAttr("min")
    UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(mp), UsdShade.Tokens.strongerThanDescendants, "physics")


# ---------------------------------------------------------------------------------------
# LIGHTING -- also shared. The collector previously relied on the school's own room lights
# while the demo added a fill rig, so identical geometry and materials still rendered at
# different exposures. Anything that changes the pixels has to be common to both.
# ---------------------------------------------------------------------------------------
FILL_INTENSITY = 320.0
FILL_HEIGHT = 2.3
FILL_RADIUS = 0.7


def author_fill_lights(stage, path_prefix, xs, y, z, mult=1.0):
    """A row of gentle sphere lights over a region, so the local floor is legible without
    blowing out. Returns the number created."""
    from pxr import UsdGeom, UsdLux, Gf

    n = 0
    for j, x in enumerate(xs):
        lp = UsdLux.SphereLight.Define(stage, f"{path_prefix}{j}")
        lp.CreateIntensityAttr(FILL_INTENSITY * float(mult))
        lp.CreateRadiusAttr(FILL_RADIUS)
        UsdGeom.Xformable(lp.GetPrim()).MakeMatrixXform().Set(
            Gf.Matrix4d().SetTranslate(Gf.Vec3d(float(x), float(y), float(z) + FILL_HEIGHT))
        )
        n += 1
    return n


# The collector lights every observation with a HEADLIGHT riding just above the camera eye.
# At 12000 it is by far the dominant illuminant in the training crops -- the school's own room
# lights and any fill rig are second-order next to it. The demo originally had no headlight at
# all, only a 320-lux fill, which is a large exposure mismatch on the exact pixels the model
# scores. Both now build the same rig from here.
HEADLIGHT_INTENSITY = 12000.0
HEADLIGHT_RADIUS = 0.12
HEADLIGHT_UP = 0.35


def author_headlight(stage, path="/World/Headlight"):
    """Create the headlight and return the xform op used to move it with the camera."""
    from pxr import UsdGeom, UsdLux

    hl = UsdLux.SphereLight.Define(stage, path)
    hl.CreateIntensityAttr(HEADLIGHT_INTENSITY)
    hl.CreateRadiusAttr(HEADLIGHT_RADIUS)
    return UsdGeom.Xformable(hl.GetPrim()).MakeMatrixXform()


def place_headlight(op, eye):
    """Put the headlight HEADLIGHT_UP above the camera eye, as the collector does."""
    from pxr import Gf

    op.Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2]) + HEADLIGHT_UP)))
