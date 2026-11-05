#!/bin/bash
# gen5 closed-loop cell (2026-10-27). Usage:
#   run_school_cell.sh <jid> <amo|spot|sonic> <stock|gen5|g5b|g5bho|g5s|g5sr> <trials> [scene SC2A|SC2A_O1|SC2A_O1L|SC2A_O1N|D1g|HW1|GC2F]
# GPU 0 by default (P5_GPU=1 only with user OK, 11-02). Environment = scripts/gen4/p4_cell.sh course branch (DISPLAY unset, gpu_wrap active_gpu 0, CUDA_VISIBLE_DEVICES=0,
# foot material, MINPOOL fix, shared default camera mount, probe-protocol flags SPEED_FOLLOW_V / MODE_DEFAULT_SPEED (no-ops for
# AMO / Spot), mode-aware fall rule, night_white, strict 0.5 m success, relative goal, no map) MINUS every execution flag:
# data/gen_v4/exec_flags.env (driver flags) is NOT sourced; XFEAS_GEN3_TRIGGER=off, SAFE_STAND=0, COMMIT_N=1, END_ROBUST=0,
# COMMIT_PDROP / TOP_CLEAR unset (xfeas_method asserts them). Nothing tuned.
source "$(dirname "$(readlink -f "$0")")/../../env.sh"; cd "$XFEAS_ROOT"
JID=$1; CTRL=$2; ARM=$3; N=$4; WHAT=${5:-SC2A_O1}
P5_GPU=${P5_GPU:-0}; [ "$P5_GPU" = 1 ] && export XFEAS_GPU1_OK=1; export CUDA_VISIBLE_DEVICES=$P5_GPU XFEAS_FOOT_MATERIAL=1 XFEAS_PERC_MINPOOL_FIX=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ISAAC_GPU_CFG="{\"multi_gpu\": false, \"active_gpu\": $P5_GPU, \"physics_gpu\": 0}"   # 11-03: renderer GPU over ALL GPUs (isaac_gpu.sh); CUDA ordinal 0 of the visible one
unset XFEAS_LEGACY_PRIMITIVES XFEAS_CAM_MOUNT XFEAS_GEN3_TOP_CLEAR DISPLAY XAUTHORITY WAYLAND_DISPLAY XFEAS_GEN3_COMMIT_PDROP
unset XFEAS_GEN3_DRV_LAT XFEAS_GEN3_DRV_PROBE XFEAS_GEN3_DRV_LAT_VX
export XFEAS_GEN3_SPEED_FOLLOW_V=1 XFEAS_GEN3_MODE_DEFAULT_SPEED=1 XFEAS_GEN3_TRIGGER=off XFEAS_GEN3_SAFE_STAND=0 XFEAS_GEN3_COMMIT_N=1 \
       XFEAS_GEN3_END_ROBUST=0 XFEAS_GEN3_MEM_PROG=0 XFEAS_GEN3_PERC=v04 XFEAS_GEN3_MODEL_HIST=1
export EP_FALL_RULE=mode_aware SONIC_PLANNER_ASYNC=0
export XFEAS_GEN5_SCENE=$WHAT
G5=$PWD/data/gen5_closed; mkdir -p $G5/hb $G5/course $G5/videos $G5/diag $G5/logs
export XFEAS_GEN5_HB=$G5/hb/$JID.hb
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
  g5s)   export XFEAS_LABEL="${LBL/held out, zero-shot/in training} + XFeas gen5b (all controllers + SONIC)" XFEAS_GEN5_CKPT=${G5S_CKPT:-$R5/gen5b/runs/all_g1s/model.pt} XF_DROP_CLS=${G5S_DROP-latL,latR}; A5=gen5;;   # 10-31 (G5S_CKPT: v2 tilt-label fix; G5S_DROP="" for v3, sidestep fixed)
  g5sr)  export XFEAS_LABEL="${LBL/held out, zero-shot/in training} + XFeas (feasibility model + skill rules)" XFEAS_GEN5_CKPT=${G5S_CKPT:-$R5/gen5b/runs/all_g1s_v3/model.pt} XF_DROP_CLS=${G5S_DROP-}; A5=rules;;   # 10-31 skill rules (xfeas/closed_loop/skill_rules.py)
  g5sr0) export XFEAS_LABEL="${LBL/held out, zero-shot/in training} + skill rules only (no model)" XFEAS_GEN5_CKPT=${G5S_CKPT:-$R5/gen5b/runs/all_g1s_v3/model.pt} XF_DROP_CLS=${G5S_DROP-} XF_RULES_NOMODEL=1; A5=rules;;   # 10-31 ablation
  g5srL) export XFEAS_LABEL="${LBL/held out, zero-shot/in training} + XFeas (model + skill rules, TEST-ONLY stay-low ${XF_RULES_STAYLOW:-6.1} m)" XFEAS_GEN5_CKPT=${G5S_CKPT:-$R5/gen5b/runs/all_g1s_v3/model.pt} XF_DROP_CLS=${G5S_DROP-} XF_RULES_STAYLOW=${XF_RULES_STAYLOW:-6.1}; A5=rules;;   # 11-01 test-only (school oil after the doorway)
  g5sm)  export XFEAS_LABEL="${LBL/held out, zero-shot/in training} + XFeas (model picks the move; rules: timing + stand-up)" XFEAS_GEN5_CKPT=${G5S_CKPT:-$XFEAS_ROOT/checkpoints/xfeas_all_g1s_v4.pt} XF_DROP_CLS=${G5S_DROP-} XF_RULES_MODELSKILL=1; A5=rules;;   # 11-03 user: model picks, rules only time it
  g5smP) export XFEAS_LABEL="${LBL/held out, zero-shot/in training} + XFeas (model picks the move; rules: timing + stand-up; SONIC execution primitives)" XFEAS_GEN5_CKPT=${G5S_CKPT:-$XFEAS_ROOT/checkpoints/xfeas_all_g1s_v4.pt} XF_DROP_CLS=${G5S_DROP-} XF_RULES_MODELSKILL=1 XF_SONIC_PRIMS=1; A5=rules;;   # 11-03: g5sm + sonic_primitives.py execution (sidestep turn/centre/slide, crawl heading hold)
  g5smC) export XFEAS_LABEL="${LBL/held out, zero-shot/in training} + XFeas (model picks the move; rules: timing + stand-up; course-safe timing)" XFEAS_GEN5_CKPT=${G5S_CKPT:-$XFEAS_ROOT/checkpoints/xfeas_all_g1s_v4.pt} XF_DROP_CLS=${G5S_DROP-} XF_RULES_MODELSKILL=1 XF_COURSE_SAFE=1 XF_SONIC_PRIMS=${G5C_PRIMS:-0}; A5=rules;;   # 11-03: g5sm + skill_rules XF_COURSE_SAFE (C1 re-observe on rising, C2 felt overhead, C3 cautious walk after stand-up, C4 gait hold); G5C_PRIMS=1 adds sonic_primitives execution
  g5smCF) export XFEAS_LABEL="${LBL/held out, zero-shot/in training} + XFeas (model picks the move; rules: timing + stand-up; course-safe timing)" XFEAS_GEN5_CKPT=${G5S_CKPT:-$XFEAS_ROOT/checkpoints/xfeas_all_g1s_v4.pt} XF_DROP_CLS=${G5S_DROP-} XF_RULES_MODELSKILL=1 XF_COURSE_SAFE=1 XF_COURSE_C2=1 XF_SONIC_PRIMS=${G5C_PRIMS:-0}; A5=rules;;   # 11-03: g5smC + C2 (felt overhead, probe-top fit with STAND_MARGIN) = the final course-safe configuration
  stock) export XFEAS_LABEL="$LBL stock" XFEAS_GEN5_CKPT=$R5/gen5b/runs/all/model.pt; A5=stock;;
  *) echo "bad arm"; exit 2;;
esac
export XFEAS_GEN5_ARM=$A5
KIT=$ISAACSIM_PYTHON
O=$G5/course
export CC_CONTACT_OUT=$O/contact_$JID.jsonl CC_CONTACT_TLOG=$O/contact_tlog_$JID.jsonl CC_INFO=$O/build_info_$JID.json
export XFEAS_GEN5_DIAG=$G5/diag/$JID EP_VIDEO_TRIALS=${EP_VIDEO_TRIALS:-}
export SC_COURSE=SC2A; MTH=xfeas_method:StairsXFeasGen5
EP=xfeas/closed_loop/school_episode.py
case $WHAT in   # 10-31: D1g / HW1 = straight drive + the side cameras of xfeas/closed_loop/run_terrain_trial.sh (as the gen4 table cells)
  SC2A) export REC_CAM=scchase SC2_TMAX=${SC2_TMAX:-300};;   # 10-31: full school course, start to finish (as gen4 table / p4_cell)
  SC2A*) export REC_CAM=scchase;;
  GC2F) export REC_CAM=gymcourse2; unset SC_COURSE; EP=xfeas/closed_loop/gym_episode.py;;   # 10-31: gym obstacle course (as gen4gym_cell.sh)
  D1g) MTH=xfeas_method:StraightXFeasGen5; export REC_CAM=zsside ZS_CAM='{"A":[-10.05,-6.35],"yaw":0.0,"floor":5.101,"side":1,"auto":1,"lat":2.3,"z":0.72,"z_tgt":0.62,"back":-1.4,"s_lo":1.0,"s_hi":5.0,"focal":10.0}';;
  HW1) MTH=xfeas_method:StraightXFeasGen5; export REC_CAM=zsside ZS_CAM='{"A":[-26.325,8.9],"yaw":1.5707963,"floor":0.681,"side":1,"auto":0,"lat":1.25,"z":2.75,"z_tgt":0.55,"back":-2.3,"fixed_s":2.5,"focal":9.0}';;
esac
echo "[p5] $(date +%T) $JID gpu $P5_GPU ctrl $CTRL arm $ARM scene $WHAT trials $N"
exec $KIT xfeas/sim/gpu_wrap.py $EP --robot $ROB --robot-factory course_ctrl:make_ct \
  --scene $WHAT --method $MTH --speed $SP --trials $N --lighting night_white --video 1 --tag _$JID \
  --out $O --vid-out $G5/videos
