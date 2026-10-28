#!/bin/bash
# gen5 closed-loop cell (2026-10-27). Usage:
#   p5_cell.sh <jid> <amo|spot> <stock|gen5|g5b|g5bho> <trials> [scene SC2A_O1|SC2A_O1L|SC2A_O1N]
# GPU 0 ONLY. Environment = scripts/gen4/p4_cell.sh course branch (DISPLAY unset, gpu_wrap active_gpu 0, CUDA_VISIBLE_DEVICES=0,
# foot material, MINPOOL fix, shared default camera mount, probe-protocol flags SPEED_FOLLOW_V / MODE_DEFAULT_SPEED (no-ops for
# AMO / Spot), mode-aware fall rule, night_white, strict 0.5 m success, relative goal, no map) MINUS every execution flag:
# data/gen_v4/exec_flags.env (driver flags) is NOT sourced; XFEAS_GEN3_TRIGGER=off, SAFE_STAND=0, COMMIT_N=1, END_ROBUST=0,
# COMMIT_PDROP / TOP_CLEAR unset (gen5_methods asserts them). Nothing tuned.
cd /p/cral/vignesh/ARL/Scripts
JID=$1; CTRL=$2; ARM=$3; N=$4; WHAT=${5:-SC2A_O1}
export CUDA_VISIBLE_DEVICES=0 XFEAS_FOOT_MATERIAL=1 XFEAS_PERC_MINPOOL_FIX=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ISAAC_GPU_CFG='{"multi_gpu": false, "active_gpu": 0, "physics_gpu": 0}'
unset XFEAS_LEGACY_PRIMITIVES XFEAS_CAM_MOUNT XFEAS_GEN3_TOP_CLEAR DISPLAY XAUTHORITY WAYLAND_DISPLAY XFEAS_GEN3_COMMIT_PDROP
unset XFEAS_GEN3_DRV_LAT XFEAS_GEN3_DRV_PROBE XFEAS_GEN3_DRV_LAT_VX
export XFEAS_GEN3_SPEED_FOLLOW_V=1 XFEAS_GEN3_MODE_DEFAULT_SPEED=1 XFEAS_GEN3_TRIGGER=off XFEAS_GEN3_SAFE_STAND=0 XFEAS_GEN3_COMMIT_N=1 \
       XFEAS_GEN3_END_ROBUST=0 XFEAS_GEN3_MEM_PROG=0 XFEAS_GEN3_PERC=v04 XFEAS_GEN3_MODEL_HIST=1
export EP_FALL_RULE=mode_aware SONIC_PLANNER_ASYNC=0
export XFEAS_GEN5_SCENE=$WHAT
G5=$PWD/data/gen5_closed; mkdir -p $G5/hb $G5/course $G5/videos $G5/diag $G5/logs
export XFEAS_GEN5_HB=$G5/hb/$JID.hb
export PYTHONPATH=$PWD/scripts/gen4:$PWD/scripts/common:$PWD/scripts/isaaclab_humanoid:$PWD/scripts/g1ctl:$PWD/scripts/gr00t:$PWD/scripts/sonic:$PWD/scripts/course:$PWD/scripts/stairs:$PWD/scripts/demo:$PWD/scripts/xembody/controllers:$PWD/scripts/overhang:$PWD/scripts/climb:$PWD/scripts/gen_train:$PWD/scripts/xembody:$PWD/scripts/gen:$PWD/scripts/terrain:$PWD/scripts/urma:$PWD/scripts/gen3
case $CTRL in
  amo)   ROB=g1a; export CC_BASE=sidestep_g1:make_g1c XFEAS_GEN5_CTRL=g1c; SP=0.6; LBL="AMO (g1c)";;
  spot)  ROB=spotp; export CC_BASE=spot_posture:make_spotp XFEAS_GEN5_CTRL=spotp; SP=0.5; LBL="Spot ReLIC (training robot)";;
  sonic) ROB=g1s; export CC_BASE=g1_sonic:make_g1s XFEAS_GEN5_CTRL=g1s; SP=0.1; LBL="GEAR-SONIC (held out, zero-shot)";;
  *) echo "bad ctrl"; exit 2;;
esac
# arms (10-27 evening): stock | gen5 (= gen5g4 model, yesterday) | g5b (gen5b all controllers) | g5bho (gen5b, AMO held out)
R5=$PWD/data/gen_train
case $ARM in
  gen5)  export XFEAS_LABEL="$LBL + XFeas gen5 (gen4-data model)" XFEAS_GEN5_CKPT=$R5/gen5g4/runs/g1c/model.pt; A5=gen5;;
  g5b)   export XFEAS_LABEL="$LBL + XFeas gen5b (all controllers)" XFEAS_GEN5_CKPT=$R5/gen5b/runs/all/model.pt; A5=gen5;;
  g5bho) export XFEAS_LABEL="$LBL + XFeas gen5b (AMO held out)" XFEAS_GEN5_CKPT=$R5/gen5b/runs/g1c_ho/model.pt; A5=gen5;;
  ft)    export XFEAS_LABEL="$LBL + XFeas gen5b two-step (random -> side-by-side, AMO held out)" XFEAS_GEN5_CKPT=$R5/gen5b/runs/ft_g1c_ho/model.pt; A5=gen5;;
  stock) export XFEAS_LABEL="$LBL stock" XFEAS_GEN5_CKPT=$R5/gen5b/runs/all/model.pt; A5=stock;;
  *) echo "bad arm"; exit 2;;
esac
export XFEAS_GEN5_ARM=$A5
KIT=/p/cral/vignesh/isaacsim-5.1.0/python.sh
O=$G5/course
export CC_CONTACT_OUT=$O/contact_$JID.jsonl CC_CONTACT_TLOG=$O/contact_tlog_$JID.jsonl CC_INFO=$O/build_info_$JID.json
export XFEAS_GEN5_DIAG=$G5/diag/$JID EP_VIDEO_TRIALS=${EP_VIDEO_TRIALS:-}
export REC_CAM=scchase SC_COURSE=SC2A
echo "[p5] $(date +%T) $JID gpu 0 ctrl $CTRL arm $ARM scene $WHAT trials $N"
exec $KIT scripts/course/gpu_wrap.py scripts/gen5/course5_ep.py --robot $ROB --robot-factory course_ctrl:make_ct \
  --scene $WHAT --method gen5_methods:StairsXFeasGen5 --speed $SP --trials $N --lighting night_white --video 1 --tag _$JID \
  --out $O --vid-out $G5/videos
