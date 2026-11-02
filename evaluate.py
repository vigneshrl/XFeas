"""Evaluate XFeas in closed loop (Isaac Sim, GPU 0), or check a trained model offline.

    python evaluate.py school  <job id> <amo|spot|sonic> <stock|g5b|g5bho> <trials> [SC2A_O1|SC2A_O1L|SC2A_O1N]
    python evaluate.py gym     <job id> <amo|sonic> <GC2F|CC1F> <trials>
    python evaluate.py terrain <spot|amo> <tile> <trial index> <out dir>
    python evaluate.py model   --ckpt data/gen_train/gen5b/runs/all/model.pt --device cpu    # offline, no simulator

school / gym results: data/gen5_closed/ and data/gen5_gym/ (runs_<robot>.jsonl per job, videos, decision logs).
"""

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
LAUNCHERS = {
    "school": "run_school_cell.sh",
    "gym": "run_gym_cell.sh",
    "terrain": "run_terrain_trial.sh",
}

if len(sys.argv) < 2 or sys.argv[1] not in (*LAUNCHERS, "model"):
    sys.exit(__doc__)
what, args = sys.argv[1], sys.argv[2:]
if what == "model":
    import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)

    sys.argv = ["model_server.py", "--selftest", *args]
    runpy.run_module("model_server", run_name="__main__", alter_sys=True)
else:
    os.execvp("bash", ["bash", os.path.join(ROOT, "xfeas", "closed_loop", LAUNCHERS[what]), *args])
