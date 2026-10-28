#!/bin/bash
# gen5b on the gym course (2026-10-28). usage: gym5_cell.sh <jid> <amo|sonic> <GC2F|CC1F> <trials>. GPU 0 only.
# Env = scripts/gen5/p5_cell.sh (no driver flags, trigger off, no stand-up rule, no commit hold); model gen5b "all".
cd /p/cral/vignesh/ARL/Scripts
JID=$1; CTRL=$2; WHAT=$3; N=$4
export CUDA_VISIBLE_DEVICES=0 XFEAS_FOOT_MATERIAL=1 XFEAS_PERC_MINPOOL_FIX=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ISAAC_GPU_CFG='{"multi_gpu": false, "active_gpu": 0, "physics_gpu": 0}'
unset XFEAS_LEGACY_PRIMITIVES XFEAS_CAM_MOUNT XFEAS_GEN3_TOP_CLEAR DISPLAY XAUTHORITY WAYLAND_DISPLAY XFEAS_GEN3_COMMIT_PDROP SC_COURSE
unset XFEAS_GEN3_DRV_LAT XFEAS_GEN3_DRV_PROBE XFEAS_GEN3_DRV_LAT_VX
export XFEAS_GEN3_SPEED_FOLLOW_V=1 XFEAS_GEN3_MODE_DEFAULT_SPEED=1 XFEAS_GEN3_TRIGGER=off XFEAS_GEN3_SAFE_STAND=0 XFEAS_GEN3_COMMIT_N=1 \
       XFEAS_GEN3_END_ROBUST=0 XFEAS_GEN3_MEM_PROG=0 XFEAS_GEN3_PERC=v04 XFEAS_GEN3_MODEL_HIST=1
export EP_FALL_RULE=mode_aware SONIC_PLANNER_ASYNC=0 XFEAS_GEN5_SCENE=$WHAT XFEAS_GEN5_ARM=gen5
G=$PWD/data/gen5_gym; mkdir -p $G/hb $G/course $G/videos $G/diag $G/logs
export XFEAS_GEN5_HB=$G/hb/$JID.hb XFEAS_GEN5_CKPT=$PWD/data/gen_train/gen5b/runs/all/model.pt XFEAS_GEN5_DIAG=$G/diag/$JID
export PYTHONPATH=$PWD/scripts/gen5:$PWD/scripts/gen4:$PWD/scripts/common:$PWD/scripts/isaaclab_humanoid:$PWD/scripts/g1ctl:$PWD/scripts/gr00t:$PWD/scripts/sonic:$PWD/scripts/course:$PWD/scripts/stairs:$PWD/scripts/demo:$PWD/scripts/xembody/controllers:$PWD/scripts/overhang:$PWD/scripts/climb:$PWD/scripts/gen_train:$PWD/scripts/xembody:$PWD/scripts/gen:$PWD/scripts/terrain:$PWD/scripts/urma:$PWD/scripts/gen3
case $CTRL in
  amo)   ROB=g1a; export CC_BASE=sidestep_g1:make_g1c XFEAS_GEN5_CTRL=g1c; SP=0.6; LBL="AMO (g1c)";;
  sonic) ROB=g1s; export CC_BASE=g1_sonic:make_g1s XFEAS_GEN5_CTRL=g1s; SP=0.1; LBL="GEAR-SONIC (held out, zero-shot)";;
esac
export XFEAS_LABEL="$LBL + XFeas gen5b (learned only)${XFEAS_GEN5_PERIOD:+, decide every ${XFEAS_GEN5_PERIOD} s}"
case $WHAT in
  GC2F) export REC_CAM=gymcourse2;;
  CC1F) export REC_CAM=ccchase CC_COURSE=v1;;
esac
O=$G/course
export CC_CONTACT_OUT=$O/contact_$JID.jsonl CC_CONTACT_TLOG=$O/contact_tlog_$JID.jsonl CC_INFO=$O/build_info_$JID.json
echo "[g5gym] $(date +%T) $JID ctrl $CTRL scene $WHAT trials $N"
exec /p/cral/vignesh/isaacsim-5.1.0/python.sh scripts/course/gpu_wrap.py scripts/gen5/gym5_ep.py --robot $ROB --robot-factory course_ctrl:make_ct \
  --scene $WHAT --method gen5_methods:StairsXFeasGen5 --speed $SP --trials $N --lighting night_white --video 1 --tag _$JID \
  --out $O --vid-out $G/videos
