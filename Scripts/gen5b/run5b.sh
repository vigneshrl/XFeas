#!/bin/bash
# train5b supervisor (2026-10-27): AMO-held-out fold (g1c_ho) -> eval val + test, the old gen5g4 g1c model on the SAME test rows
# (reference), then the all-controllers model (all) for the closed loop -> eval val. Each training resumes from last.pt, <= 10 launches.
# usage: setsid nohup bash scripts/gen5b/run5b.sh > data/gen_train/gen5b/runs/supervisor.log 2>&1 &
cd /p/cral/vignesh/ARL/Scripts
PY=/p/cral/vignesh/envs/gsam2/bin/python; export CUDA_VISIBLE_DEVICES=0
R=data/gen_train/gen5b/runs; mkdir -p $R; echo $$ > $R/supervisor.pid
fit() {  # out holdout gen4ckpt
  local O=$1 HB=$2 G4=$3 n=0; mkdir -p $O
  while [ ! -f $O/model.pt ] && [ $n -lt 10 ]; do
    n=$((n+1)); echo "$(date '+%m-%d %H:%M:%S') $O launch $n"
    $PY scripts/gen5b/train5b.py --out $O --holdout-base "$HB" --gen4-ckpt $G4 >> $O/train.log 2>&1
    [ -f $O/model.pt ] || sleep 30
  done
  [ -f $O/model.pt ] || { echo "GAVE UP $O"; return 1; }
}
ev() { [ -f $1/select_$2.json ] || $PY scripts/gen5b/train5b.py --eval-only --ckpt $1/model.pt --eval-split $2 > $1/eval_$2.log 2>&1; echo "$(date '+%m-%d %H:%M:%S') eval $1 $2 rc $?"; }
fit $R/g1c_ho g1c data/gen_train/gen4/full/f_g1c/ckpt/xfeas_gen4_ens.pt && { ev $R/g1c_ho val; ev $R/g1c_ho test; }
mkdir -p $R/ref_gen5g4_g1c; ln -sf $(realpath data/gen_train/gen5g4/runs/g1c/model.pt) $R/ref_gen5g4_g1c/model.pt; ev $R/ref_gen5g4_g1c test
fit $R/all "" data/gen_train/gen4/full/f_none/ckpt/xfeas_gen4_ens.pt && ev $R/all val
echo "$(date '+%m-%d %H:%M:%S') ALL DONE"
