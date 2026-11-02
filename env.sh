# XFeas environment. The launchers source this; source it yourself before running a script by hand:
#   source env.sh
# Edit the defaults below (or export the variables before sourcing) to match your machine.
export XFEAS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ISAACSIM_PYTHON="${ISAACSIM_PYTHON:-$HOME/isaacsim-5.1.0/python.sh}"   # Isaac Sim 5.1
export ISAACLAB_PY="${ISAACLAB_PY:-$HOME/envs/isaaclab_venv/bin/python}"     # Isaac Lab venv (terrain tiles only)
export XFEAS_PY="${XFEAS_PY:-$(command -v python)}"                          # torch env: training and the model server
export GR00T_PY="${GR00T_PY:-$XFEAS_PY}"                                     # SONIC / GR00T / PHP policy servers
export EVORA_PY="${EVORA_PY:-$XFEAS_PY}"                                     # EVORA baseline only

export XFEAS_BASELINES="${XFEAS_BASELINES:-$XFEAS_ROOT/third_party}"
export XFEAS_ISAAC_ASSETS="${XFEAS_ISAAC_ASSETS:-$XFEAS_ROOT/third_party/isaac_assets}"
export XFEAS_SCENE_ASSETS="${XFEAS_SCENE_ASSETS:-$XFEAS_ROOT/third_party/scene_assets}"
export PEGASUS_DIR="${PEGASUS_DIR:-$XFEAS_ROOT/third_party/PegasusSimulator}"

export PYTHONPATH="$XFEAS_ROOT:$PYTHONPATH"     # makes `import xfeas` work (or: pip install -e .)
