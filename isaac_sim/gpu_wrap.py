"""Patch isaacsim.SimulationApp so every config gets ISAAC_GPU_CFG (renderer active GPU / physics CUDA ordinal /
multi-GPU off), then run the target script as __main__. Used by isaac_gpu.sh."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import json, os, runpy, sys
import isaacsim
_SA = isaacsim.SimulationApp
CFG = json.loads(os.environ.get("ISAAC_GPU_CFG", "{}"))


class SimulationApp(_SA):
    def __init__(self, launch_config=None, *a, **k):
        c = dict(launch_config or {}); c.update(CFG)
        print(f"[gpu_wrap] SimulationApp config += {CFG} (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})", flush=True)
        super().__init__(c, *a, **k)


isaacsim.SimulationApp = SimulationApp
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
