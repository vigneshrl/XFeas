#!/bin/bash
# train5g4 supervisor (2026-10-26): train (resumes from last.pt, <= 10 launches), then selection eval on val + test.
# usage: setsid nohup bash scripts/gen5/run_g4.sh <holdout_base> > data/gen_train/gen5g4/runs/<hb>/supervisor.log 2>&1 &
cd /p/cral/vignesh/ARL/Scripts
HB=$1; O=data/gen_train/gen5g4/runs/${HB:-none}; G4=data/gen_train/gen4/full/f_${HB:-none}/ckpt/xfeas_gen4_ens.pt
PY=/p/cral/vignesh/envs/gsam2/bin/python; export CUDA_VISIBLE_DEVICES=0
mkdir -p $O; echo $$ > $O/supervisor.pid
n=0
while [ ! -f $O/model.pt ] && [ $n -lt 10 ]; do
  n=$((n+1)); echo "$(date '+%m-%d %H:%M:%S') launch $n"
  $PY scripts/gen5/train5g4.py --out $O --holdout-base "$HB" --gen4-ckpt $G4 >> $O/train.log 2>&1
  [ -f $O/model.pt ] || sleep 30
done
[ -f $O/model.pt ] || { echo GAVE UP; exit 1; }
for S in val test; do
  [ -f $O/select_$S.json ] || $PY scripts/gen5/train5g4.py --eval-only --ckpt $O/model.pt --eval-split $S > $O/eval_$S.log 2>&1
done
echo "$(date '+%m-%d %H:%M:%S') ALL DONE"
