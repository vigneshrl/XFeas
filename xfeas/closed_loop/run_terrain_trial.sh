#!/bin/bash
# ONE gen5b terrain trial in a fresh Isaac process (2026-10-28). GPU 0 only. usage: run_terrain_trial.sh <spot|amo> <tile> <k> <outdir>
# Harness = xfeas/closed_loop/run_terrain_trial.sh tile branch (same runners, field, start, goal, strict 0.5 m rule, night_white);
# layer env = xfeas/closed_loop/run_school_cell.sh (no driver flags, trigger off, no stand-up rule, no commit hold); model gen5b "all".
CTRL=$1; TILE=$2; K=$3; D=$4
source "$(dirname "$(readlink -f "$0")")/../../env.sh"; R=$XFEAS_ROOT; cd $R; mkdir -p $D
export CUDA_VISIBLE_DEVICES=0 XFEAS_FOOT_MATERIAL=1 XFEAS_PERC_MINPOOL_FIX=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ISAAC_GPU_CFG='{"multi_gpu": false, "active_gpu": 0, "physics_gpu": 0}'
unset XFEAS_LEGACY_PRIMITIVES XFEAS_CAM_MOUNT XFEAS_GEN3_TOP_CLEAR DISPLAY XAUTHORITY WAYLAND_DISPLAY XFEAS_GEN3_COMMIT_PDROP
unset XFEAS_GEN3_DRV_LAT XFEAS_GEN3_DRV_PROBE XFEAS_GEN3_DRV_LAT_VX
export XFEAS_GEN3_SPEED_FOLLOW_V=1 XFEAS_GEN3_MODE_DEFAULT_SPEED=1 XFEAS_GEN3_TRIGGER=off XFEAS_GEN3_SAFE_STAND=0 XFEAS_GEN3_COMMIT_N=1 \
       XFEAS_GEN3_END_ROBUST=0 XFEAS_GEN3_MEM_PROG=0 XFEAS_GEN3_PERC=v04 XFEAS_GEN3_MODEL_HIST=1
export EP_FALL_RULE=mode_aware SONIC_PLANNER_ASYNC=0 XFEAS_GEN5_SCENE=none XFEAS_GEN5_ARM=gen5
export XFEAS_GEN5_CKPT=$R/data/gen_train/gen5b/runs/all/model.pt XFEAS_GEN5_DIAG=$D/diag XFEAS_GEN5_HB=$D/gen5.hb
case $CTRL in
  spot) export XFEAS_GEN5_CTRL=spotp FT_GEN5_SPEED=0.5 EVORA_BAND=local XFEAS_LABEL="Spot ReLIC + XFeas gen5b (learned only)"
        CMD="xfeas/sim/isaaclab_python.sh xfeas/sim/gpu_wrap.py xfeas/closed_loop/terrain_harness_spot.py --tile $TILE --trial $K --out $D --method terrain_method:Gen5Terrain";;
  amo)  export XFEAS_GEN5_CTRL=g1c TERRAIN_FALL_TILT=1.0 XFEAS_LABEL="G1 AMO + XFeas gen5b (learned only)"
        CMD="xfeas/sim/isaaclab_python.sh xfeas/sim/gpu_wrap.py xfeas/closed_loop/terrain_harness_g1.py --mode feel --robot g1c --seed 0 --difficulty 0.6 --speed 0.6 --tile $TILE --trial $K --method terrain_method:Gen5Terrain --method-light night_white --video 1 --vid-light night_white --vid-every 4 --out $D --vid-out $D";;
esac
echo "[trial5] $(date) $CTRL $TILE t$K" > $D/run.log; echo "$CMD" >> $D/run.log
timeout -k 60 2400 $CMD >> $D/run.log 2>&1; echo "[trial5] exit $? $(date)" >> $D/run.log
