"""XFeas paths. Every script imports this first (two-line bootstrap at its top).

* puts every XFeas code folder on sys.path: modules import each other by bare name (import model5, import gensim4, ...)
* holds the machine-specific locations. Each one is an environment variable with a default; set them in env.sh
  (the shell launchers source it) or export them before running a script by hand.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE_DIRS = ("Scripts/model", "Scripts/data_collector", "Scripts/train", "closed_loop", "closed_loop/baselines",
             "isaac_sim", "isaac_sim/controllers", "isaac_sim/controllers/sonic", "isaac_sim/controllers/php",
             "isaac_sim/controllers/gr00t", "isaac_sim/controllers/urma", "isaac_sim/controllers/isaaclab",
             "scenes", "scenes/terrain")
for _d in reversed(CODE_DIRS):
    _p = os.path.join(ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _env(name, default):
    return os.environ.get(name) or default


# interpreters
ISAACSIM_PYTHON = _env("ISAACSIM_PYTHON", os.path.expanduser("~/isaacsim-5.1.0/python.sh"))  # Isaac Sim 5.1 kit python
XFEAS_PY = _env("XFEAS_PY", sys.executable)          # torch env: training, model server, tools
GR00T_PY = _env("GR00T_PY", sys.executable)          # GR00T-WBC / SONIC / PHP policy servers (onnxruntime)
EVORA_PY = _env("EVORA_PY", sys.executable)          # EVORA baseline sidecar (numba-cuda)
# third-party code and assets (README: Setup)
BASELINES = _env("XFEAS_BASELINES", os.path.join(ROOT, "third_party"))          # released controllers / planners
ISAAC_ASSETS = _env("XFEAS_ISAAC_ASSETS", os.path.join(ROOT, "third_party", "isaac_assets"))   # Isaac Sim asset mirror
SCENE_ASSETS = _env("XFEAS_SCENE_ASSETS", os.path.join(ROOT, "third_party", "scene_assets"))   # Abandoned School pack
PEGASUS = _env("PEGASUS_DIR", os.path.join(ROOT, "third_party", "PegasusSimulator"))           # Iris drone
COURSES = _env("XFEAS_COURSES", os.path.join(ROOT, "scenes", "courses"))         # course specs / tasks (shipped)
DATA = os.path.join(ROOT, "data")                    # datasets, probes, checkpoints, outputs (README: Data)
