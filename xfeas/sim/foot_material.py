"""
OPT-IN robot contact materials + ONE friction-combine convention for every surface.   XFEAS_FOOT_MATERIAL=1

Why: the shipped robot USDs have no physics material on their feet (Spot relic, AMO G1) or a wrong one (Jackal 0.2),
so PhysX used its built-in default (mu 0.5, combine 'average', the LOWEST priority). Every surface therefore had to
use 'min' (low-mu surfaces) or 'max' (sticky) to apply at all, and min(tile, 0.5) capped all grippy surfaces at 0.5.

Flag OFF (default, env var unset or 0): NOTHING here runs -> every existing result (stock, VAMOS, navloco, XFeas,
oracle) stays bit-reproducible.

Flag ON, convention = PhysX friction combine 'multiply' on EVERY material (robot feet, terrain tiles, the school's
own PM_Oil / PM_Rubble, and the scene's default material that all unbound floors/walls use):

    contact mu = mu_robot_contact * f_surface

  * robot contact links carry the ABSOLUTE rubber value against the reference surface (dry concrete):
        RUBBER = mu_s 0.9, mu_d 0.8 (feet of Spot and G1, tyres of Jackal / Carter)
  * every surface carries a dimensionless factor f = intended contact mu / RUBBER (separately for static and dynamic),
    so a surface designed as "contact mu = m" gets exactly m against a rubber sole.  Dry concrete / asphalt /
    the school's untagged floors = factor 1.0 (exactly Isaac Lab's terrain material: multiply, 1.0 / 1.0).
  * 'multiply' has priority over 'min' and 'average' in PhysX (max > multiply > min > average), so a surface that
    still says 'min' (e.g. PM_Oil as authored in the collected school USD) is resolved as multiply: that is why
    apply_scene() converts PM_Oil's VALUES too (0.0882 / 0.0706 -> factors 0.098 / 0.088), in the session layer only
    (the USD on disk is never modified). No surface may use 'max' under this convention.

Sources for RUBBER (no manufacturer publishes a foot/tyre friction for Spot, G1 or Jackal -> literature + assumption):
  * rubber on dry concrete: 0.6-0.85 (Engineering ToolBox, "Friction and friction coefficients"); tyre peak / sliding
    on dry asphalt or concrete 0.8-0.9 / 0.75 (J.Y. Wong, Theory of Ground Vehicles, table of average road-adhesion coefficients). 0.9 / 0.8 is the
    upper end of these ranges = a clean, grippy rubber sole. ASSUMPTION: Spot's, G1's and Jackal's rubber are the same.
  * consistent with what the controllers were trained on: Isaac Lab Spot flat env randomises robot static friction
    in [0.3, 1.0] / dynamic [0.3, 0.8] against a multiply 1.0 terrain (isaaclab_tasks .../config/spot/flat_env_cfg.py);
    Isaac Lab's base locomotion env uses 0.8 / 0.6 (velocity_env_cfg.py). AMO (G1) was trained in Isaac Gym (not checked).
  * Iris: aerial, its rotor/landing-gear contacts never touch the ground in flight -> no material (documented).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os, re

RUBBER = dict(mu_s=0.9, mu_d=0.8)
# rigid-body link names that touch the ground (matched on the link prim's name)
CONTACT_LINKS = [
    ("spot_foot", re.compile(r"^[fh][lr]_foot$")),  # Spot (stock 'spot' and relic 'spotp')
    ("g1_foot", re.compile(r"^(left|right)_ankle_roll_link$")),  # Unitree G1 (AMO): 4 r=5 mm spheres per foot
    ("tyre", re.compile(r"wheel", re.I)),  # Jackal (4 tyres), Carter (2 tyres + casters)
    # additive 2026-10-15 (comparison videos): URMA's ANYmal C carries its foot spheres on the *_SHANK links
    # (xfeas/sim/controllers/urma/urma_robot.py feet map); without this entry it kept the PhysX default under the flag. No earlier
    # result is affected: anyu was never run with XFEAS_FOOT_MATERIAL=1 before this edit.
    ("anymal_foot", re.compile(r"^[LR][FH]_SHANK$")),
]
MAT_ROOT = "/World/XFeasPhysMaterials"


def enabled():
    return os.environ.get("XFEAS_FOOT_MATERIAL", "0").strip().lower() not in ("", "0", "false", "no", "off")


def factors(mu_s, mu_d):
    """intended contact mu (against the rubber reference) -> the surface's multiply factors"""
    return float(mu_s) / RUBBER["mu_s"], float(mu_d) / RUBBER["mu_d"]


def _material(stage, name, mu_s, mu_d, restitution=0.0):
    from pxr import UsdShade, UsdPhysics, PhysxSchema

    path = f"{MAT_ROOT}/{name}"
    p = stage.GetPrimAtPath(path)
    if p.IsValid():
        return UsdShade.Material(p)
    mat = UsdShade.Material.Define(stage, path)
    m = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    m.CreateStaticFrictionAttr().Set(float(mu_s))
    m.CreateDynamicFrictionAttr().Set(float(mu_d))
    m.CreateRestitutionAttr().Set(float(restitution))
    x = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    x.CreateFrictionCombineModeAttr().Set("multiply")
    x.CreateRestitutionCombineModeAttr().Set("min")
    return mat


def apply_robot(stage, prim_path):
    """bind the rubber material (multiply) to every contact link of the robot under prim_path. -> {link: material}"""
    from pxr import Usd, UsdPhysics, UsdShade

    mat = _material(stage, "robot_rubber", RUBBER["mu_s"], RUBBER["mu_d"])
    out = {}
    for p in Usd.PrimRange(stage.GetPrimAtPath(prim_path)):
        if not p.HasAPI(UsdPhysics.RigidBodyAPI) or p.IsInstanceProxy():
            continue
        for tag, rx in CONTACT_LINKS:
            if rx.search(p.GetName()):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")
                out[str(p.GetPath())] = tag
                break
    return out


def _physics_scene(stage):
    from pxr import UsdPhysics

    for p in stage.Traverse():
        if p.IsA(UsdPhysics.Scene):
            return p
    return None


def apply_scene(stage, school_root="/World/School"):
    """(idempotent) default material = dry floor (factor 1.0, multiply) bound to the physics scene -> used by every
    collider without its own material (the school's floors and walls, hazard copies, procedural walls); the school's
    own physics materials (PM_Oil, PM_Rubble: 'min', absolute mu) -> multiply with factors, in the SESSION layer."""
    from pxr import Usd, UsdPhysics, UsdShade, PhysxSchema

    info = dict(default=None, converted=[])
    sc = _physics_scene(stage)
    if sc is not None:
        fs, fd = factors(RUBBER["mu_s"], RUBBER["mu_d"])  # = 1.0, 1.0
        mat = _material(stage, "default_floor", fs, fd)
        UsdShade.MaterialBindingAPI.Apply(sc).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")
        info["default"] = str(sc.GetPath())
    root = stage.GetPrimAtPath(school_root)
    if root.IsValid():
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for p in Usd.PrimRange(root):
                if not p.HasAPI(UsdPhysics.MaterialAPI) or p.GetCustomDataByKey("xfeas_converted"):
                    continue
                m = UsdPhysics.MaterialAPI(p)
                x = PhysxSchema.PhysxMaterialAPI.Apply(p)
                mode = x.GetFrictionCombineModeAttr().Get()
                s, d = m.GetStaticFrictionAttr().Get(), m.GetDynamicFrictionAttr().Get()
                fs, fd = factors(s if s is not None else 0.5, d if d is not None else 0.5)
                m.GetStaticFrictionAttr().Set(fs)
                m.GetDynamicFrictionAttr().Set(fd)
                x.GetFrictionCombineModeAttr().Set("multiply")
                p.SetCustomDataByKey("xfeas_converted", True)
                info["converted"].append(
                    dict(path=str(p.GetPath()), mode_was=mode, mu_was=[s, d], factors=[round(fs, 4), round(fd, 4)])
                )
    return info


def apply(prim_path):
    """hook called by RobotAdapter._fix_contact_materials (every robot factory) when the flag is on"""
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    links = apply_robot(stage, prim_path)
    sc = apply_scene(stage)
    print(
        f"[foot_material] XFEAS_FOOT_MATERIAL=1: rubber mu {RUBBER['mu_s']}/{RUBBER['mu_d']} (multiply) on {links}; "
        f"scene default floor factor 1.0 on {sc['default']}; converted {[(c['path'].split('/')[-1], c['factors']) for c in sc['converted']]}",
        flush=True,
    )
    return links, sc
