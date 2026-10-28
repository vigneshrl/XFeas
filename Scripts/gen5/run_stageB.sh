#!/bin/bash
# Stage B supervisor (2026-10-26): round 0 = Monte-Carlo value (init = Stage A model), rounds 1..R = bootstrapped targets
# (target5.py from the previous round). Every step is skipped when its output exists, so a re-launch resumes; train5
# itself resumes from last.pt. Then eval val + test + timing5 + candcheck5 on the last round.
# usage: setsid nohup bash scripts/gen5/run_stageB.sh <holdout_base> > <B dir>/supervisor.log 2>&1 &
cd /p/cral/vignesh/ARL/Scripts
HB=$1; R=${ROUNDS:-3}; A=data/gen_train/gen5/runs/A_${HB}; B=data/gen_train/gen5/runs/B_${HB}; M=data/gen_train/gen5/manifest.jsonl
PY=/p/cral/vignesh/envs/gsam2/bin/python; export CUDA_VISIBLE_DEVICES=0
mkdir -p $B; echo $$ > $B/supervisor.pid
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
train() {   # out init [vtarget] epochs
  local n=0
  while [ ! -f $1/model.pt ] && [ $n -lt 10 ]; do
    n=$((n+1)); log "train $1 launch $n"; mkdir -p $1
    $PY scripts/gen5/train5.py --stage B --manifest $M --out $1 --holdout-base "$HB" --init $2 ${3:+--vtarget $3} --epochs $4 >> $1/train.log 2>&1
    [ -f $1/model.pt ] || sleep 30
  done
  [ -f $1/model.pt ] || { log "GAVE UP $1"; exit 1; }
}
train $B/r0 $A/model.pt "" 6
for r in $(seq 1 $R); do
  p=$B/r$((r-1)); o=$B/r$r; mkdir -p $o
  n=0
  while [ ! -f $o/vtarget.npz ] && [ $n -lt 5 ]; do
    n=$((n+1)); log "targets r$r launch $n"; $PY scripts/gen5/target5.py --ckpt $p/model.pt --manifest $M --out $o/vtarget.npz >> $o/target.log 2>&1
  done
  [ -f $o/vtarget.npz ] || { log "GAVE UP targets r$r"; exit 1; }
  train $o $p/model.pt $o/vtarget.npz 4
done
L=$B/r$R
for S in val test; do
  [ -f $L/metrics_$S.json ] || $PY scripts/gen5/train5.py --eval-only --ckpt $L/model.pt --manifest $M --eval-split $S >> $L/eval_$S.log 2>&1
done
[ -f $L/timing/timing.json ] || $PY scripts/gen5/timing5.py --ckpt $L/model.pt --manifest $M --out $L/timing > $L/timing.log 2>&1
[ -f $L/candcheck_test.json ] || $PY scripts/gen5/candcheck5.py --ckpt $L/model.pt --manifest $M --split test > $L/candcheck_test.log 2>&1
[ -f $B/r0/timing/timing.json ] || $PY scripts/gen5/timing5.py --ckpt $B/r0/model.pt --manifest $M --out $B/r0/timing > $B/r0/timing.log 2>&1
log "ALL DONE"
