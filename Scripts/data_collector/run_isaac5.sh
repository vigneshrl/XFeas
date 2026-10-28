#!/bin/bash
# gen_v5 launcher (copy of scripts/gen4/run_isaac4.sh + scripts/gen5, scripts/php, scripts/sonic on the path).
# Usage: run_isaac5.sh <gpu 0|1> <script> args...
# GPU 0 or 1 (user approval 2026-10-22 for the gen5 collection). CUDA sees only that GPU; the renderer's active GPU is set by
# isaac_sim/gpu_wrap.py (ISAAC_GPU_CFG). Foot material 1, perception MINPOOL fix, real assets only, the shared default
# camera mount. Executor flags of gen_v4 (data/gen_v4/exec_flags.env) are loaded for the probe protocol parity.
source "$(dirname "$(readlink -f "$0")")/../../env.sh"; cd "$XFEAS_ROOT"
GPU=$1; shift
if [ "$GPU" != "0" ] && [ "$GPU" != "1" ]; then echo "GPU must be 0 or 1"; exit 2; fi
export CUDA_VISIBLE_DEVICES=$GPU XFEAS_FOOT_MATERIAL=1 XFEAS_PERC_MINPOOL_FIX=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ISAAC_GPU_CFG="{\"multi_gpu\": false, \"active_gpu\": $GPU, \"physics_gpu\": 0}"
unset XFEAS_LEGACY_PRIMITIVES XFEAS_CAM_MOUNT DISPLAY XAUTHORITY
if [ -f data/gen_v4/exec_flags.env ]; then set -a; . data/gen_v4/exec_flags.env; set +a; fi
# terrain (Isaac Lab photoreal tiles) needs the Isaac Lab python (run_isaac4.sh launcher 'isaaclab')
prev=""; for x in "$@"; do if [ "$prev" = "--env" ] && [ "$x" = "terrain" ]; then export PYTHONEXE=$ISAACLAB_PY; fi; prev=$x; done
exec $ISAACSIM_PYTHON isaac_sim/gpu_wrap.py "$@"
