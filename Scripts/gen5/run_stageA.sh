#!/bin/bash
# Stage A supervisor (2026-10-26): train (resumes from last.pt after any crash, <= 20 launches), then eval val + test.
# usage: setsid nohup bash scripts/gen5/run_stageA.sh <holdout_base> > <out>/supervisor.log 2>&1 &
cd /p/cral/vignesh/ARL/Scripts
HB=$1; O=data/gen_train/gen5/runs/A_${HB:-none}; M=data/gen_train/gen5/manifest.jsonl
PY=/p/cral/vignesh/envs/gsam2/bin/python; export CUDA_VISIBLE_DEVICES=0
mkdir -p $O; echo $$ > $O/supervisor.pid
n=0
while [ ! -f $O/model.pt ] && [ $n -lt 20 ]; do
  n=$((n+1)); echo "$(date '+%m-%d %H:%M:%S') launch $n"
  $PY scripts/gen5/train5.py --manifest $M --out $O --holdout-base "$HB" ${EXTRA} >> $O/train.log 2>&1
  echo "$(date '+%m-%d %H:%M:%S') train exited rc $?"; [ -f $O/model.pt ] || sleep 30
done
[ -f $O/model.pt ] || { echo "GAVE UP"; exit 1; }
for S in val test; do
  [ -f $O/metrics_$S.json ] || $PY scripts/gen5/train5.py --eval-only --ckpt $O/model.pt --manifest $M --eval-split $S >> $O/eval_$S.log 2>&1
done
echo "$(date '+%m-%d %H:%M:%S') ALL DONE"
