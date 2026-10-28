"""
Spot gym course level 1 with the 45 deg jog (experiments/4_obstacle_course/spot/L1, SpotJogCourse) as an episode_runner
scene for G1 runs (additive 2026-10-22). Built exactly by the published rebuild recipe (task_L1.json 'rebuild', same
sequence as render_gym_course_spot_jog.py --variant jog): school.usd in memory, gym_course_levels.prepare_level(stage, 1),
make_spot_level_spec(1, 0) + spec['jog'] = JOG, SpotJogCourse.plan_jog, B.build(spec).

  GJ1F  start = task start (-23.226, 2.6); path = task polyline up to the stairs foot (-12.696, 3.661); goal = 0.35 m
        before the stairs foot on the lane (G1 planner modes do not climb stairs; same convention as CC1F), no goal_z.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import json, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
for _p in (os.path.join(ROOT, "closed_loop", "baselines"), os.path.join(ROOT, "scenes"), os.path.join(ROOT, "scenes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TASK_JSON = xfeas_paths.COURSES + "/4_obstacle_course/spot/L1/task_L1.json"
TASK = json.load(open(TASK_JSON))
SID = "GJ1F"


def _scene():
    P = [tuple(map(float, p["xyz"])) for p in TASK["path"]]
    i_foot = [i for i, p in enumerate(TASK["path"]) if "stairs foot" in p["what"]][0]
    foot = np.array(P[i_foot])
    prev = np.array(P[i_foot - 1])
    d = (foot - prev)[:2]; d /= np.linalg.norm(d)
    goal = foot[:2] - 0.35 * d
    path = P[:i_foot] + [(float(goal[0]), float(goal[1]), float(foot[2]))]
    L = float(np.sum(np.hypot(*np.diff(np.asarray(path)[:, :2], axis=0).T)))
    start = tuple(map(float, TASK["start"]["xy"]))
    return dict(route=None, hazard=tuple(TASK["turns"][0]["xy"]) if TASK.get("turns") and "xy" in TASK["turns"][0] else tuple(P[3][:2]),
                desc="Spot gym course L1 with the 45 deg jog in the tilted section -> foot of the stairs (G1 run)",
                start=start, goal=(float(goal[0]), float(goal[1])), floor=float(TASK["start"]["z"]), path=path, goal_z=None,
                t_max=6.0 * L + 60.0, fall_tilt=1.0, build="jog_scene:build")


def runner_scenes():
    import stairs_scenes as SS
    sc = _scene()
    SS.STAIR_SCENES[SID] = dict(path=sc["path"], floor=sc["floor"], goal_z=None, desc=sc["desc"])
    return {SID: sc}


def build(stage, sid, sc):
    import gym_course_spot as GS
    import gym_course_levels as LV
    import gym_course_spot_jog as GJ
    off = LV.prepare_level(stage, 1)
    spec = GS.make_spot_level_spec(1, 0)
    spec["level"] = 1
    spec["jog"] = dict(GJ.JOG); spec["variant"] = "spot_jog"
    B = GJ.SpotJogCourse(stage)
    B.spec = spec; B.plan = GJ.plan_jog(spec, B.dims)
    info = B.build(spec)
    print(f"[jog] GJ1F: prepare deactivated {len(off)}; built {info.get('n_parts')} parts (published L1: 289)", flush=True)
    try:
        import course_ctrl as CC
        CC._register_parts(B.parts)
    except Exception as e:
        print("[jog] contact part registration skipped:", repr(e), flush=True)
