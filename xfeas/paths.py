"""Machine-specific locations, each an environment variable with a default (set them in env.sh).

`ROOT` is the repo root. Datasets, probes and checkpoints live in `ROOT/data` (see README, "Data").
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name, default):
    return os.environ.get(name) or default


# interpreters
ISAACSIM_PYTHON = _env("ISAACSIM_PYTHON", os.path.expanduser("~/isaacsim-5.1.0/python.sh"))  # Isaac Sim 5.1 kit python
XFEAS_PY = _env("XFEAS_PY", sys.executable)  # torch env: training, model server, tools
GR00T_PY = _env("GR00T_PY", sys.executable)  # GR00T-WBC / SONIC / PHP policy servers (onnxruntime)
EVORA_PY = _env("EVORA_PY", sys.executable)  # EVORA baseline sidecar (numba-cuda)
# third-party code and assets (README: Setup)
BASELINES = _env("XFEAS_BASELINES", os.path.join(ROOT, "third_party"))  # released controllers / planners
ISAAC_ASSETS = _env("XFEAS_ISAAC_ASSETS", os.path.join(ROOT, "third_party", "isaac_assets"))  # Isaac Sim asset mirror
SCENE_ASSETS = _env("XFEAS_SCENE_ASSETS", os.path.join(ROOT, "third_party", "scene_assets"))  # Abandoned School pack
PEGASUS = _env("PEGASUS_DIR", os.path.join(ROOT, "third_party", "PegasusSimulator"))  # Iris drone
COURSES = _env("XFEAS_COURSES", os.path.join(ROOT, "assets", "courses"))  # course specs / tasks (shipped)
DATA = os.path.join(ROOT, "data")  # datasets, probes, checkpoints, outputs (README: Data)
