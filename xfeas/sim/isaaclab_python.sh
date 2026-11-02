#!/bin/bash
# Run a script with Isaac Sim 5.1's kit runtime + the ISOLATED Isaac Lab venv ($ISAACLAB_PY,
# created from the kit python with --system-site-packages; isaaclab 2.3.2 installed there with --no-deps).
# The base kit python (python.sh) is untouched: python.sh honours PYTHONEXE, which we point at the venv interpreter.
export CUDA_VISIBLE_DEVICES=0
export PYTHONEXE=$ISAACLAB_PY
exec $ISAACSIM_PYTHON "$@"
