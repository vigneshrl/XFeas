#!/bin/bash
# Stage I (IQL) supervisor (2026-10-26): round 0 = Q / V on the Monte-Carlo outcome (init = Stage B last round), rounds
# 1..R = target5.py --iql from the previous round + Q / V / chooser training. Steps are skipped when their output exists
# (re-launch resumes; train5 resumes from last.pt). Then eval val + test and the chooser readout (pick5, mode pi).
# usage: setsid nohup bash scripts/gen5/run_stageI.sh <holdout_base> > <I dir>/supervisor.log 2>&1 &
cd /p/cral/vignesh/ARL/Scripts
HB=$1; R=${ROUNDS:-3}; B=data/gen_train/gen5/runs/B_${HB}/r3; I=data/gen_train/gen5/runs/I_${HB}; M=data/gen_train/gen5/manifest.jsonl
PY=/p/cral/vignesh/envs/gsam2/bin/python; export CUDA_VISIBLE_DEVICES=0
mkdir -p $I; echo $$ > $I/supervisor.pid
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
train() {   # out init [iql npz] epochs
  local n=0
  while [ ! -f $1/model.pt ] && [ $n -lt 10 ]; do
    n=$((n+1)); log "train $1 launch $n"; mkdir -p $1
    $PY scripts/gen5/train5.py --stage I --manifest $M --out $1 --holdout-base "$HB" --init $2 ${3:+--iql $3} --epochs $4 >> $1/train.log 2>&1
    [ -f $1/model.pt ] || sleep 30
  done
  [ -f $1/model.pt ] || { log "GAVE UP $1"; exit 1; }
}
train $I/r0 $B/model.pt "" 4
for r in $(seq 1 $R); do
  p=$I/r$((r-1)); o=$I/r$r; mkdir -p $o
  n=0
  while [ ! -f $o/iql.npz ] && [ $n -lt 5 ]; do
    n=$((n+1)); log "targets r$r launch $n"; $PY scripts/gen5/target5.py --iql --ckpt $p/model.pt --manifest $M --out $o/iql.npz >> $o/target.log 2>&1
  done
  [ -f $o/iql.npz ] || { log "GAVE UP targets r$r"; exit 1; }
  train $o $p/model.pt $o/iql.npz 4
  [ -f $o/pick/pick.json ] || $PY scripts/gen5/pick5.py --ckpt $o/model.pt --manifest $M --out $o/pick > $o/pick.log 2>&1
done
L=$I/r$R
for S in val test; do
  [ -f $L/metrics_$S.json ] || $PY scripts/gen5/train5.py --eval-only --ckpt $L/model.pt --manifest $M --eval-split $S >> $L/eval_$S.log 2>&1
done
log "ALL DONE"
