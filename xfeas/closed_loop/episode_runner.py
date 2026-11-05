#!/usr/bin/env python
"""Runs evaluation episodes in Isaac Sim: builds the scene, spawns the robot with its controller, runs one method
(stock, XFeas or a baseline) without a map towards a relative goal, scores success (within 0.5 m, no fall) and
records video. Every method is scored by this same runner.

Details:
MAP-FREE evaluation episodes on the six school failure scenes. Every method (stock, ours,
baselines) is a plugin with the SAME inputs and is scored by the SAME runner.

What a method gets (and nothing else):
  * its robot's stock controller (it sends body-frame twists [vx, vy, vz, wz]; clipped to limits)
  * odometry: the robot's own pose/velocity (bs), proprioception via ctx['rob'].proprio
  * the goal position relative to the robot (obs['goal_rel'], body frame) -- GPS+compass style;
    obs['goal_xy'] = pose + goal_rel for convenience
  * any onboard sensors it builds itself in setup (cameras on the robot, PhysX laser fans, ...)
NO map, no route, no waypoints, no hazard locations. Scene geometry is used only by the runner
(spawn, outcome, chase-camera video).

Method plugin (module:Class, or 'stock' / 'xfeas' from local_nav.py):
    name, label                 strings for logs / video
    setup(ctx)                  ctx: stage, rep, world, app, rob, spec, kind, hz, pxq, carb, robot_root
    reset(ep)                   ep: speed_frac, floor_z0, scene, trial
    act(obs) -> dict(twist=np.array(4), [info=str], [giveup=True], [decisions=list])
                                obs: t, dt, bs, rob, goal_rel, goal_xy, goal_dist   (called at 50 Hz)

Outcomes: success (within 0.5 m of the goal), fall, stuck (no 0.2 m
of progress toward the goal for 15 s), gave_up (method says so), timeout.

    $ISAACSIM_PYTHON xfeas/closed_loop/episode_runner.py         --robot h1 --scene S2 --method stock --speed 0.6 --trials 3 --tag _ep_stock
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import argparse, importlib, json, math, os, sys, time
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
USD = os.path.join(ROOT, "assets", "school.usd")

# scene: route (for start/goal/floor only), hazard (for logging distance only), description.
# start = route point 5 m before the hazard, goal = 4 m past it (S5: route start).
SCENES = {
    "S1": dict(route="G05", hazard=(-17.26, 1.05), desc="oil spill on the gym floor (mu 0.09)", b=5.0, a=4.0),
    "S2": dict(route="G00", hazard=(-17.30, 1.50), desc="oil spill on the gym floor (mu 0.09)", b=5.0, a=4.0),
    "S3": dict(route="U01", hazard=(-26.24, 7.43), desc="fallen ceiling lamp, upper corridor (9 cm)", b=5.0, a=4.0),
    "S4": dict(route="G05", hazard=(-13.65, 0.80), desc="broken window frame on the floor (7 cm)", b=5.0, a=4.0),
    "S5": dict(route="G08", hazard=(-11.30, 3.10), desc="broken door lying on the floor (18 cm)", b=5.0, a=4.0),
    "S6": dict(route="G01", hazard=(-29.06, -7.41), desc="narrow doorway, west wing (1.3 m)", b=5.0, a=4.0),
    # crouch test: the thick door BP_school_door_5 raised so its underside is 0.90-0.94 m above the floor on this
    # lane (Spot standing top 1.01 m hits it; at the lowest commanded body height its top is 0.86 m). Explicit lane.
    "D1": dict(
        route=None,
        hazard=(-4.0, -2.1),
        desc="raised thick door: underside 0.92 m (crouch to pass)",
        start=(-4.0, -6.0),
        goal=(-4.0, 1.5),
        floor=0.681,
    ),
}
# D1 SUPPORTED (2026-10-10, no floating objects): a real door leaf resting on a low cabinet and a shelf unit lying on its
# side, underside 0.91 m over the lane (Spot crouch band 0.88-0.955 m); lane 0.15 m further from the wall. The old
# floating door (baked into school.usd on 2026-10-07, deactivated by the 2026-10-10 bake) only with D1_LEGACY_DOOR=1.
if os.environ.get("D1_LEGACY_DOOR", "0") != "1":
    SCENES["D1"] = dict(
        route=None,
        hazard=(-4.15, -2.1),
        desc="door leaf on a cabinet + shelf unit: underside 0.91 m (crouch to pass)",
        start=(-4.15, -6.0),
        goal=(-4.15, 1.5),
        floor=0.681,
        u=0.906,
        s_front=3.535,
        build="overhang_scenes:build_d1",
    )
BUILTIN = {
    "stock_straight": "door_methods:StraightStock",
    "xfeaspos_straight": "door_methods:StraightXFeas",
    "crouch_straight": "door_methods:StraightCrouch",
    "stock": "local_nav:LocalNav",
    "xfeas": "local_nav:XFeasLocal",
    "xfeas5": "local_nav:XFeasLocalV2",
    "xfeaspos": "local_nav:XFeasPosture",
}
# overhang study (xfeas/scenes/overhang_scenes.py): V1 = validation scene (selector decisions), T1 = held-out test
SCENES.update(
    {
        "V1": dict(
            route=None,
            hazard=(-22.84, 4.6),
            desc="raised door leaf (validation lane), underside 0.93 m",
            start=(-22.95, 0.05),
            goal=(-22.48, 6.23),
            floor=0.022,
            u=0.93,
            build="overhang_scenes:build",
        ),
        "T1": dict(
            route=None,
            hazard=(-4.7, 9.8),
            desc="wardrobe lying on two low cabinets (held-out object + place), underside 0.91 m",
            start=(-4.25, 5.35),
            goal=(-4.893, 11.517),
            floor=5.101,
            u=0.885,
            support_room=(0.6, 0.5),
            build="overhang_scenes:build",
        ),
        "V3": dict(
            route=None,
            hazard=(-24.8, -6.18),
            desc="raised slab (validation lane 2), underside 0.91 m",
            start=(-21.649, -6.327),
            goal=(-27.343, -6.061),
            floor=5.101,
            u=0.91,
            s_front=3.0,
            build="overhang_scenes:build",
        ),
        "T0": dict(
            route=None,
            hazard=(-4.7, 9.8),
            desc="no-overhang control: T1's lane with nothing over it",
            start=(-4.25, 5.35),
            goal=(-4.893, 11.517),
            floor=5.101,
        ),
        # V1's start is not a valid spawn (oracle stuck at t=0); V2 = validation lane oh0062 (upper corridor, val zone)
        "V2": dict(
            route=None,
            hazard=(-25.0, -6.17),
            desc="door frame on two low cabinets (validation lane 2), underside 0.91 m",
            start=(-21.649, -6.327),
            goal=(-27.343, -6.061),
            floor=5.101,
            u=0.91,
            s_front=3.0,
            build="overhang_scenes:build",
        ),
    }
)
BUILTIN["xfeashold_straight"] = "oh_methods:StraightXFeasHold"  # overhang study: selector variant (scripts/overhang)
BUILTIN["xfeas_fullstack"] = (
    "xfeas_fullstack:XFeasFullStack"  # ours on the camera planner, posture-aware band (xfeas_fullstack.py)
)
# overhang-2 (crouch-cost) scenes, xfeas/scenes/overhang_scenes.py:build_cost. H1/H2 = held-out TEST (upper floor,
# inside the D1 exclusion rectangle), W2 = validation copy of H2 on the V2/V3 validation lane (selector decisions only)
SCENES.update(
    {
        "H1": dict(
            route=None,
            hazard=(-4.49, 3.5),
            desc="high doorway frame, lintel 1.30 m (passable standing)",
            start=(-4.43, 6.5),
            goal=(-4.56, 0.3),
            floor=5.101,
            build="overhang_scenes:build_cost",
        ),
        "H2": dict(
            route=None,
            hazard=(-4.47, 4.3),
            desc="low slab 0.92 m, then an 11 cm curb 1.6 m behind it",
            start=(-4.43, 6.5),
            goal=(-4.56, 0.3),
            floor=5.101,
            stand_goal_dist=3.3,
            build="overhang_scenes:build_cost",
        ),
        "W2": dict(
            route=None,
            hazard=(-23.65, -6.23),
            desc="(validation) low slab 0.92 m, then an 11 cm curb",
            start=(-21.649, -6.327),
            goal=(-27.343, -6.061),
            floor=5.101,
            stand_goal_dist=2.8,
            build="overhang_scenes:build_cost",
        ),
    }
)
BUILTIN["crouchstand_straight"] = "oh_methods:StraightCrouchThenStand"  # privileged oracle for H2/W2
# narrow-passage scenes N* (sidestep study): procedural walls from sidestep_scene.py, explicit lanes
try:
    import sidestep_scene as _NS

    SCENES.update(_NS.runner_scenes())
    BUILTIN["sidestep_oracle"] = "sidestep_methods:SidestepOracle"
except ImportError:
    _NS = None
# G1 overhang study (xfeas/scenes/g1_overhang.py): D1g = slab with D1's door footprint, G1-sized underside, on an
# upper-floor lane >= 6.3 m from every training/validation lane and same-floor test scene (u from G1's crouch band,
# measured on training lane oh0003 by g1_overhang_probe.py, never on a test scene)
SCENES["D1g"] = dict(
    route=None,
    hazard=(-6.875, -6.35),
    desc="slab with D1 door footprint, G1-sized underside 1.10 m",
    start=(-10.05, -6.35),
    goal=(-4.55, -6.35),
    floor=5.101,
    u=1.10,
    s_front=2.8,
    build="g1_overhang:build_d1g",
)
BUILTIN["crouchpitch_straight"] = "g1_overhang:StraightCrouchPitchG1"  # G1 oracle: lowest height + training-lane pitch
# climb study (scripts/climb): G1 = 3 stacked school gym mats, G2 = 4 x 8 cm box stairs, explicit lanes in the gym;
# geometry frozen in data/climb/test_scenes.json (make_test_scenes.py); needs scripts/climb on PYTHONPATH
try:
    import climb_scenes as _CS

    SCENES.update(_CS.runner_scenes())
    BUILTIN["climb_oracle"] = "climb_methods:ClimbOracle"
    BUILTIN["xfeashold_v"] = "climb_methods:StraightXFeasHoldV"
except ImportError:
    pass
# REAL school staircases (xfeas/scenes/stairs_scenes.py): SP = probe flight (east), ST = test (west, ground -> upper
# floor). Optional scene keys used below only when present: path (progress along a 3-D polyline), goal_z (success
# needs the base above that floor), t_max, fall_tilt (stairs fall rule). Needs scripts/stairs on PYTHONPATH.
try:
    import stairs_scenes as _SS

    SCENES.update(_SS.runner_scenes())
    BUILTIN["stairs_stock"] = "stairs_methods:StairsStock"
    BUILTIN["stairs_oracle"] = "stairs_methods:StairsOracle"
except ImportError:
    _SS = None
# gen1 closed-loop arms (xfeas/_support/closed_loop/gen_methods.py; additive 2026-10-11; utility via XFEAS_GEN_UTILITY):
BUILTIN["xfeasgen_straight"] = "gen_methods:StraightXFeasGen"
BUILTIN["xfeasgen_fullstack"] = "gen_methods:XFeasGenFullStack"
NICE = dict(
    success="REACHED GOAL",
    fall="FELL",
    stuck="STUCK",
    gave_up="GAVE UP",
    timeout="TIMED OUT",
    invalid_start="BAD START",
)


def scene_geometry(sid):
    sc = SCENES[sid]
    if sc.get("route") is None:  # explicit straight lane (D1)
        P = np.array([sc["start"], sc["goal"]], float)
        return dict(
            start=P[0],
            goal=P[1],
            yaw=math.atan2(*(P[1] - P[0])[::-1]),
            pts=P,
            floor=float(sc["floor"]),
            level=0,
            straight=float(np.hypot(*(P[1] - P[0]))),
        )
    d = json.load(open(os.path.join(ROOT, "data", "nav_baseline", "routes.json")))
    rt = [r for r in d["routes"] if r["id"] == sc["route"]][0]
    P = np.asarray(rt["pts"], float)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    ih = int(np.argmin(np.hypot(*(P - np.asarray(sc["hazard"])).T)))
    i = int(np.searchsorted(s, max(0.0, s[ih] - sc["b"])))
    j = min(int(np.searchsorted(s, s[ih] + sc["a"])), len(P) - 1)
    k = min(i + 6, len(P) - 1)
    z = rt.get("z") or [None] * len(P)
    return dict(
        start=P[i],
        goal=P[j],
        yaw=math.atan2(*(P[k] - P[i])[::-1]),
        pts=P,
        floor=float(z[i]) if z[i] is not None else 0.0,
        level=1 if sc["route"].startswith("U") else 0,
        straight=float(np.hypot(*(P[j] - P[i]))),
    )


def build_scene(stage, sid):
    """build scene `sid` in memory exactly once: if it is BAKED into school.usd (baked_scenes.py) only its recording
    camera is registered; baked objects of OTHER scenes that touch this scene's lane are deactivated (in memory);
    D1_LEGACY_DOOR=1 restores the old raised door instead of the supported D1."""
    import baked_scenes as BK

    sc = SCENES[sid]
    G = scene_geometry(sid)
    is_baked, off = BK.prepare(stage, sid, dict(start=G["start"], yaw=G["yaw"], len=G["straight"], floor=G["floor"]))
    if off:
        print(f"[ep] {sid}: deactivated baked objects of other scenes on this lane: {off}", flush=True)
    if sid == "D1" and os.environ.get("D1_LEGACY_DOOR", "0") == "1":
        door = stage.GetPrimAtPath(
            "/World/School/Geom/BP_school_door_5/SM_school_door_3_3/InstancedSM_school_door_3_04"
        )
        if door.IsValid():
            door.SetActive(True)
        for r in BK.baked(stage).get("D1", []):
            if stage.GetPrimAtPath(r).IsValid():
                stage.GetPrimAtPath(r).SetActive(False)
        return "legacy"
    if is_baked:
        print(f"[ep] {sid}: baked into school.usd ({BK.baked(stage)[sid]}), not rebuilt", flush=True)
        try:
            if _NS is not None and sid in _NS.N_SCENES:
                _NS._register_cam(_NS.N_SCENES[sid])
            if sc.get("build", "").startswith("climb_scenes"):
                import climb_scenes

                climb_scenes.register_cam(sid)
        except Exception as e:
            print("[ep] rec cam not registered:", e, flush=True)
        return "baked"
    if sc.get("build"):
        _m, _f = sc["build"].split(":")
        getattr(importlib.import_module(_m), _f)(stage, sid, sc)
    if _NS is not None and sid in _NS.N_SCENES:
        _NS.build(stage, _NS.N_SCENES[sid])
    return "built"


def method_label(method, spec):
    """banner label: the method's own label; the generic 'STOCK controller' arms name the robot's actual controller
    (e.g. URMA on ANYmal C runs through the stock_straight arm but is NOT the Spot/G1 stock controller)"""
    lab = getattr(method, "label", method.name)
    if os.environ.get("XFEAS_LABEL"):  # additive 2026-10-11 (gen1 videos): explicit banner label for the arm
        return os.environ["XFEAS_LABEL"].replace("~", " ")
    ctrl = (getattr(spec, "extra", None) or {}).get("controller")
    if ctrl and lab.upper().startswith("STOCK"):
        rest = lab.split(",", 1)[1].strip() if "," in lab else ""
        lab = f"{ctrl}: unchanged policy" + (f", {rest}" if rest else "")
    return lab


def baked_roots(stage, sid):
    import baked_scenes as BK

    return BK.baked(stage).get(sid, [])


def load_method(spec):
    spec = BUILTIN.get(spec, spec)
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--scene", required=True, help="S1..S6 or comma list")
    ap.add_argument("--method", default="stock")
    ap.add_argument("--speed", type=float, default=0.6, help="fraction of max_cmd (methods may go slower)")
    ap.add_argument("--trials", type=int, default=1, help="trial 0 nominal start; k>0 seeded +-10 cm / +-5 deg")
    ap.add_argument("--tag", default="")
    ap.add_argument("--video", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "episodes"))
    ap.add_argument(
        "--lighting",
        default="original",
        choices=["original", "day", "dusk", "night_white", "night_yellow"],
        help="scene_lighting.py",
    )
    ap.add_argument(
        "--hazards",
        default="none",
        choices=["none", "dup"],
        help="dup: pack copies of the scene's hazard across the whole path width (scene_hazards.py); one scene per run",
    )
    ap.add_argument(
        "--robot-factory", default="", help="module:fn for a robot not in robots.py (fn.SPEC = its RobotSpec)"
    )
    ap.add_argument("--vid-out", default=os.path.join(ROOT, "videos", "episodes"))
    a = ap.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True, "width": 1280, "height": 720, "renderer": "RaytracedLighting"})
    import carb, omni.usd, omni.replicator.core as rep
    from omni.isaac.core.world import World
    from omni.physx import get_physx_scene_query_interface
    from pxr import UsdGeom, UsdLux, Gf, Sdf
    import robots as RB, scene_clean, chase as CH
    from video import write_mp4

    kind = a.robot
    factory = None
    if a.robot_factory:  # robot not in robots.py (e.g. the A1 of a baseline):
        mod, fn = a.robot_factory.split(":")  # module:fn -> fn(kind, prim_path, pos, yaw) returns an adapter
        factory = getattr(importlib.import_module(mod), fn)  # with .spec and the RobotAdapter interface
        spec = factory.SPEC
    else:
        spec = RB.SPECS[kind]
    HZ = spec.phys_hz
    omni.usd.get_context().open_stage(USD)
    for _ in range(20):
        app.update()
    world = World(physics_dt=1.0 / HZ, rendering_dt=1 / 50.0, stage_units_in_meters=1.0)
    stage = world.stage
    for p in ("/World/Spot", "/World/Surfaces"):
        if stage.GetPrimAtPath(p).IsValid():
            stage.GetPrimAtPath(p).SetActive(False)
    scene_clean.apply(stage)
    if a.hazards == "dup":  # copies must exist before physics starts (colliders)
        import scene_hazards

        if len(a.scene.split(",")) != 1:
            raise SystemExit("--hazards dup needs exactly one scene per run (S1 and S4 share a route)")
        Gh = scene_geometry(a.scene)
        scene_hazards.build(
            stage, a.scene, Gh["start"], Gh["goal"], Gh["floor"], Gh["level"], Gh["pts"], SCENES[a.scene]["hazard"]
        )
    CHASE_YAW = None
    for _sid in a.scene.split(","):  # scene objects (in memory, or already baked into school.usd)
        print(f"[ep] scene {_sid}: {build_scene(stage, _sid)}", flush=True)
    if _NS is not None and a.scene in _NS.N_SCENES:  # narrow passage: chase cam keeps lane heading
        CHASE_YAW = _NS.N_SCENES[a.scene]["heading"]
    try:  # real-asset rule (2026-10-10): report any primitive geometry we created
        import real_assets as _RA

        _au = _RA.audit_stage(stage)
        print(f"[ep] no-primitive audit: ok={_au['ok']} gprims={_au['n_gprims']} bad={_au['bad'][:3]}", flush=True)
    except Exception as _e:
        print("[ep] no-primitive audit failed:", _e, flush=True)
    import scene_lighting

    LIGHT = scene_lighting.Lighting(stage)
    print("[ep] lighting", LIGHT.set(a.lighting, np.random.default_rng(7)), flush=True)
    world.reset()  # live physics once before spawning (Spot renders only then)
    for _ in range(10):
        world.step(render=False)
    world.stop()
    pxq = get_physx_scene_query_interface()

    scenes = a.scene.split(",")
    G0 = scene_geometry(scenes[0])
    root = f"/World/{kind}"
    rob = (factory or RB.make_robot)(
        kind, root, [G0["start"][0], G0["start"][1], G0["floor"] + spec.spawn_dz], yaw=G0["yaw"]
    )
    method = load_method(a.method)
    tag = a.tag or f"_ep_{method.name}"
    rec = None
    if a.video:
        up = 2.0 if kind == "iris" else 3.0
        rec = CH.ChaseRecorder(
            stage,
            rep,
            world,
            os.path.join(a.vid_out, "frames", f"_tmp_{kind}{tag}"),
            res=(1280, 720),
            fps=30,
            phys_hz=HZ,
            focal=13.0,
            back=3.5,
            side=0.3,
            up=up,
            look_ahead=1.5,
        )
        pose0 = rec._pose

        def pose_clamped(bs, _p=pose0):  # keep the chase eye under the ceiling / in front of walls
            eye, tgt = _p(bs if CHASE_YAW is None else dict(bs, yaw=CHASE_YAW))
            o = np.asarray(bs["pos"], float) + np.array([0, 0, 0.5])
            h = pxq.raycast_closest(carb.Float3(*map(float, o)), carb.Float3(0, 0, 1), 6.0)
            if h and h.get("hit"):
                eye = eye.copy()
                eye[2] = min(eye[2], o[2] + float(h["distance"]) - 0.35)
            dv = eye - o
            n = float(np.linalg.norm(dv))
            if n > 1e-3:
                h = pxq.raycast_closest(carb.Float3(*map(float, o)), carb.Float3(*map(float, dv / n)), n)
                if h and h.get("hit") and not str(h.get("collision", "")).startswith(root):
                    eye = o + dv / n * max(0.6, float(h["distance"]) - 0.3)
            return eye, tgt

        rec._pose = pose_clamped
        rc_name = os.environ.get("REC_CAM", "")
        if not rc_name and a.lighting.startswith("night"):
            rc_name = "sceneside"  # night: the top-down chase view is nearly black -> side camera + fill
        if rc_name == "sceneside":
            import rec_cam

            _s0 = SCENES[scenes[0]]
            _G = scene_geometry(scenes[0])
            os.environ["REC_SCENE_LANE"] = json.dumps(
                rec_cam.scene_side_cfg(
                    _G["start"], _G["goal"], _G["floor"], _s0["hazard"], tall=kind.startswith(("g1", "h1"))
                )
            )
        if rc_name:  # comparison videos: scripted recording camera + fill light (rec_cam.py)
            import rec_cam

            rec_cam.install(rec, stage, rep, pxq, carb, spec, root, rc_name)
    try:  # v4 (2026-10-19): headlight on the shared camera mount for EVERY arm
        # (stock arms too); a method with a decision camera re-sets it
        import camera_mount as _CMT, scene_lighting as _SL4

        if not _CMT.is_legacy(kind):
            _SL4.HEAD_MOUNT_FN = _CMT.MountedCamera(stage, rob, kind, root=root, spec=spec).light_matrix
    except Exception as _e4:
        print("[camera_mount] headlight mount not set:", repr(_e4), flush=True)
    ctx = dict(
        stage=stage,
        rep=rep,
        world=world,
        app=app,
        rob=rob,
        spec=spec,
        kind=kind,
        hz=HZ,
        pxq=pxq,
        carb=carb,
        robot_root=root,
    )
    method.setup(ctx)
    rep.orchestrator.step(rt_subframes=1, delta_time=0.0, pause_timeline=False)  # prime
    world.reset()
    world.add_physics_callback("drive", lambda dt: rob.step(dt))
    for _ in range(int(1.0 * HZ)):
        world.step(render=False)
    os.makedirs(os.path.join(a.out, tag.strip("_")), exist_ok=True)
    os.makedirs(a.vid_out, exist_ok=True)
    out_jsonl = os.path.join(a.out, tag.strip("_"), f"runs_{kind}.jsonl")

    def reset_policy_state():
        c = getattr(rob, "ctrl", None)
        for at in ("_previous_action", "_old_action", "action"):
            if c is not None and hasattr(c, at):
                try:
                    setattr(c, at, np.zeros_like(getattr(c, at)))
                except Exception:
                    pass

    rec_all = rec  # (additive 2026-10-15) EP_VIDEO_TRIALS="0,2": film only these trials;
    _VT = os.environ.get("EP_VIDEO_TRIALS", "")  # unset = every trial, as before. Rendering never steps physics.
    for sid in scenes:
        G = scene_geometry(sid)
        for trial in range(
            int(os.environ.get("XF_TRIAL0", "0")), a.trials
        ):  # 11-03: XF_TRIAL0 = first trial index (split a cell over GPUs)
            if str(trial) in os.environ.get("EP_SKIP_TRIALS", "").split(
                ","
            ):  # (additive 2026-10-21) resume: trial already recorded
                print(f"[ep] {sid} trial {trial}: skipped (EP_SKIP_TRIALS)", flush=True)
                continue
            rec = rec_all if (rec_all is None or not _VT or str(trial) in _VT.split(",")) else None
            rng = np.random.default_rng(
                1000 * trial + int(''.join(ch for ch in sid[1:] if ch.isdigit()) or 0)
            )  # D1g -> 1
            x0, y0, yaw0 = float(G["start"][0]), float(G["start"][1]), G["yaw"]
            if trial > 0:
                lat = rng.uniform(-0.10, 0.10)
                yaw0 += rng.uniform(-math.radians(5), math.radians(5))
                x0, y0 = x0 - lat * math.sin(G["yaw"]), y0 + lat * math.cos(G["yaw"])
            ok = False
            for attempt in range(3):
                if attempt and spec.family != "aerial":
                    world.reset()
                    rob.ready = False
                    for _ in range(int(0.5 * HZ)):
                        world.step(render=False)
                if hasattr(
                    rob, "reset_setpoints"
                ):  # 2026-10-21: a trial must not inherit the last trial's mode / speed / height
                    rob.reset_setpoints()  # (SONIC kept a layer-commanded mode -> every later start 'invalid')
                rob.teleport(x0, y0, yaw0, G["floor"])
                reset_policy_state()
                rob.command(np.zeros(4))
                for _ in range(int(2.0 * HZ)):
                    world.step(render=False)
                bs = rob.base_state()
                if (
                    bs is not None
                    and not rob.fallen(bs)
                    and np.hypot(*(bs["pos"][:2] - [x0, y0])) < (1.0 if kind == "iris" else 0.5)
                ):
                    ok = True
                    break
            run = dict(
                robot=kind,
                scene=sid,
                route=SCENES[sid]["route"],
                desc=SCENES[sid]["desc"],
                method=method.name,
                trial=trial,
                speed_frac=a.speed,
                start=[round(x0, 3), round(y0, 3)],
                goal=np.round(G["goal"], 3).tolist(),
                straight_m=round(G["straight"], 2),
                map_given=False,
                hazards=a.hazards,
                lighting=LIGHT.info,
            )
            if os.environ.get("XFEAS_FOOT_MATERIAL", "0") not in (
                "",
                "0",
            ):  # opt-in contact materials (foot_material.py)
                run["foot_material"] = 1
            if not ok:
                run.update(outcome="invalid_start")
                open(out_jsonl, "a").write(json.dumps(run) + "\n")
                continue
            method.reset(dict(speed_frac=a.speed, floor_z0=G["floor"], scene=sid, trial=trial))
            goal = np.asarray(G["goal"], float)
            v_nom = max(a.speed * spec.max_cmd[0], spec.v_min)
            T_MAX = 3.0 * G["straight"] / v_nom + 40.0
            _sc = SCENES[sid]
            _PP = None  # stairs keys (optional)
            if _sc.get("t_max"):
                T_MAX = float(_sc["t_max"])
            if _sc.get("path") and _SS is not None:
                _PP = _SS.PathProgress(_sc["path"])
            every = max(1, HZ // 50)
            fdir = os.path.join(a.vid_out, "frames", f"{kind}_{sid}{tag}_t{trial}")
            if rec is not None:
                rec.dir = fdir
                os.makedirs(fdir, exist_ok=True)
                for f in os.listdir(fdir):
                    os.remove(os.path.join(fdir, f))
                rec.n, rec.k, rec.pos, rec.tags = 0, 0, None, []
            finfo, traj, info, decisions = [], [], "", []
            best, t_prog, outcome, min_dh, t = 1e9, 0.0, None, 1e9, 0.0
            tw = np.zeros(4)
            wall0 = time.time()
            _FJ = None
            if os.environ.get("EP_FALL_RULE", "") == "mode_aware":
                import fall_judge as _FJM

                _FJ = _FJM.FallJudge(rob, spec)
            for step in range(int(T_MAX * HZ)):
                world.step(render=False)
                t = step / HZ
                bs = rob.base_state()
                if rec is not None and bs is not None:
                    n0 = rec.n
                    rec.tick(bs)
                    if rec.n > n0:
                        finfo.append((round(t, 2), info, float(tw[0])))
                if step % every:
                    continue
                if _FJ is not None:  # EP_FALL_RULE=mode_aware (additive 2026-10-19): same rule, every arm
                    _fz = None
                    if _PP is not None and bs is not None:
                        _s, _ = _PP.update(float(bs["pos"][0]), float(bs["pos"][1]))
                        _dprog = _PP.total - _s
                        _fz = _PP.surface_z(_s)
                    if _FJ.fallen(
                        t, bs, _fz, _sc.get("fall_tilt", spec.fall_tilt) if _PP is not None else spec.fall_tilt
                    ):
                        outcome = "fall"
                        break
                elif _PP is not None and bs is not None:  # stairs: path progress + slope-aware fall rule
                    _s, _ = _PP.update(float(bs["pos"][0]), float(bs["pos"][1]))
                    _dprog = _PP.total - _s
                    if (
                        bs["tilt"] > _sc.get("fall_tilt", spec.fall_tilt)
                        or float(bs["pos"][2]) - _PP.surface_z(_s) < spec.min_z
                    ):
                        outcome = "fall"
                        break
                elif bs is None or rob.fallen(bs):
                    outcome = "fall"
                    break
                x, y, yaw = float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])
                d = goal - np.array([x, y])
                c, s_ = math.cos(yaw), math.sin(yaw)
                obs = dict(
                    t=t,
                    dt=every / HZ,
                    bs=bs,
                    rob=rob,
                    goal_rel=np.array([c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]]),
                    goal_xy=goal.copy(),
                    goal_dist=float(np.hypot(*d)),
                )
                LIGHT.follow(bs, spec.cam)
                r = method.act(obs)
                tw = np.asarray(r["twist"], float)
                info = r.get("info", "")
                decisions = r.get("decisions", decisions)
                rob.command(tw)
                if r.get("giveup"):
                    outcome = "gave_up"
                    break
                dg = float(np.hypot(*d))
                min_dh = min(min_dh, float(np.hypot(x - SCENES[sid]["hazard"][0], y - SCENES[sid]["hazard"][1])))
                if len(traj) == 0 or t - traj[-1][0] >= 0.1:
                    traj.append(
                        [
                            round(t, 2),
                            round(x, 3),
                            round(y, 3),
                            round(float(bs["pos"][2]), 3),
                            round(yaw, 3),
                            round(float(bs["v_b"][0]), 3),
                            round(float(tw[0]), 3),
                            round(float(tw[3]), 3),
                            round(math.degrees(bs["tilt"]), 1),
                        ]
                    )
                dprog = _dprog if _PP is not None else dg
                if dprog < best - 0.2:
                    best, t_prog = dprog, t
                if dg < 0.5 and (_sc.get("goal_z") is None or float(bs["pos"][2]) > _sc["goal_z"] + 0.15):  # strict
                    outcome = "success"
                    break
                if t - t_prog > 15.0:
                    outcome = "stuck"
                    break
            outcome = outcome or "timeout"
            bs = rob.base_state()
            run.update(
                outcome=outcome,
                success=outcome == "success",
                t_end=round(t, 2),
                dist_to_goal=round(float(np.hypot(*(goal - bs["pos"][:2]))), 2) if bs is not None else None,
                min_dist_hazard=round(min_dh, 2),
                max_tilt_deg=max([r_[-1] for r_ in traj] or [0]),
                mean_vx_cmd=round(float(np.mean([r_[6] for r_ in traj])) if traj else 0.0, 3),
                wall_s=round(time.time() - wall0, 1),
                decisions=decisions,
                frames=rec.n if rec else 0,
            )
            if _FJ is not None:
                run.update(fall_rule="mode_aware", fall_info=_FJ.info())
            np.save(
                os.path.join(a.out, tag.strip("_"), f"traj_{kind}_{sid}_t{trial}.npy"), np.asarray(traj, np.float32)
            )
            open(out_jsonl, "a").write(json.dumps(run) + "\n")
            print(
                f"[ep] {kind} {sid} {method.name} trial {trial}: {outcome} t={t:.1f} min_dist_hazard={min_dh:.2f}",
                flush=True,
            )
            if rec is not None and rec.n:  # overlay + mp4 in the ML env (no ffmpeg in Isaac python)
                json.dump(
                    dict(
                        frames=finfo,
                        label=method_label(method, spec),
                        robot=kind,
                        scene=sid,
                        desc=SCENES[sid]["desc"],
                        outcome=outcome,
                    ),
                    open(os.path.join(fdir, "meta.json"), "w"),
                )
                import subprocess

                subprocess.run(
                    [
                        xfeas.paths.XFEAS_PY,
                        os.path.join(THIS, "..", "sim", "episode_video.py"),
                        fdir,
                        os.path.join(a.vid_out, f"{kind}_{sid}{tag}_t{trial}.mp4"),
                    ],
                    check=False,
                )
    print("[ep] DONE", flush=True)
    app.close()


if __name__ == "__main__":
    main()
