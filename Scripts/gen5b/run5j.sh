#!/bin/bash
# joint / ft supervisor (2026-10-27): AMO-held-out fold. joint_f50 / joint_f25 = gen5b + non-school gen4 + gen5 random per-tick
# single-candidate rows (half / quarter of each batch); ft = gen5 random Stage-B model fine-tuned on gen5b + gen4. Each resumes from
# last.pt (<= 10 launches), then evals val (mix ratio chosen on val only).
# usage: setsid nohup bash scripts/gen5b/run5j.sh > data/gen_train/gen5b/runs/supervisor_j.log 2>&1 &
cd /p/cral/vignesh/ARL/Scripts
PY=/p/cral/vignesh/envs/gsam2/bin/python; export CUDA_VISIBLE_DEVICES=0
R=data/gen_train/gen5b/runs; echo $$ > $R/supervisor_j.pid
G4=data/gen_train/gen4/full/f_g1c/ckpt/xfeas_gen4_ens.pt
fit() {  # out extra-args...
  local O=$1 n=0; shift; mkdir -p $O
  while [ ! -f $O/model.pt ] && [ $n -lt 10 ]; do
    n=$((n+1)); echo "$(date '+%m-%d %H:%M:%S') $O launch $n"
    $PY scripts/gen5b/train5b.py --out $O --holdout-base g1c --workers 8 "$@" >> $O/train.log 2>&1
    [ -f $O/model.pt ] || sleep 30
  done
  [ -f $O/model.pt ] || { echo "GAVE UP $O"; return 1; }
  [ -f $O/select_val.json ] || $PY scripts/gen5b/train5b.py --eval-only --ckpt $O/model.pt --eval-split val --workers 8 > $O/eval_val.log 2>&1
  echo "$(date '+%m-%d %H:%M:%S') $O done (val rc $?)"
}
fit $R/joint_f50_g1c_ho --gen4-ckpt $G4 --g5-manifest data/gen_train/gen5/manifest.jsonl --g5-frac 0.5 &
fit $R/joint_f25_g1c_ho --gen4-ckpt $G4 --g5-manifest data/gen_train/gen5/manifest.jsonl --g5-frac 0.25 &
fit $R/ft_g1c_ho --init data/gen_train/gen5/runs/B_g1c/r0/model.pt --norm-from-init &
wait
echo "$(date '+%m-%d %H:%M:%S') ALL DONE"
