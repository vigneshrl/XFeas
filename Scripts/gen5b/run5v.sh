#!/bin/bash
# verification supervisor (2026-10-27). The gen5 random-data model B_g1c was initialised from the gen4 RELEASE weights, which
# trained on 342 AMO gen4 situations (most of the G4 test rows) and its gen5 data includes school. Clean chain: gen5 Stage A +
# Stage B r0 on the no-school manifest from the AMO-free gen4 fold f_g1c, then the two-step fine-tune (ft_clean). Seed repeats
# for noise. ft_all (AMO included, clean chain init) for the closed loop. Every step skips when its model.pt exists.
# usage: setsid nohup bash scripts/gen5b/run5v.sh > data/gen_train/gen5b/runs/supervisor_v.log 2>&1 &
cd /p/cral/vignesh/ARL/Scripts
PY=/p/cral/vignesh/envs/gsam2/bin/python; export CUDA_VISIBLE_DEVICES=1   # 10-27 20:05 user approved GPU 1 for this training
R=data/gen_train/gen5b/runs; G=data/gen_train/gen5/runs; echo $$ > $R/supervisor_v.pid
M=data/gen_train/gen5/manifest_noschool.jsonl; F=data/gen_train/gen4/full/f_g1c/ckpt/xfeas_gen4_ens.pt
F0=data/gen_train/gen4/full/f_none/ckpt/xfeas_gen4_ens.pt
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
loop() {  # out cmd...
  local O=$1 n=0; shift; mkdir -p $O
  while [ ! -f $O/model.pt ] && [ $n -lt 10 ]; do n=$((n+1)); log "$O launch $n"; "$@" >> $O/train.log 2>&1; [ -f $O/model.pt ] || sleep 30; done
  [ -f $O/model.pt ] && log "$O done" || { log "GAVE UP $O"; return 1; }
}
t5b() { local O=$1; shift; loop $O $PY scripts/gen5b/train5b.py --out $O --workers 6 "$@"; }
clean() {  # holdout gen4ckpt tag
  local HB=$1 W=$2 T=$3
  loop $G/A_${T} $PY scripts/gen5/train5.py --manifest $M --out $G/A_${T} --holdout-base "$HB" --gen4-ckpt $W || return 1
  loop $G/B_${T}/r0 $PY scripts/gen5/train5.py --stage B --manifest $M --out $G/B_${T}/r0 --holdout-base "$HB" --init $G/A_${T}/model.pt --epochs 6 || return 1
}
( clean g1c $F g1c_clean && t5b $R/ft_clean_g1c_ho --holdout-base g1c --init $G/B_g1c_clean/r0/model.pt --norm-from-init \
  && t5b $R/ft_clean_g1c_ho_s1 --holdout-base g1c --init $G/B_g1c_clean/r0/model.pt --norm-from-init --seed 1 ) &
( t5b $R/g1c_ho_s1 --holdout-base g1c --gen4-ckpt $F --seed 1 && t5b $R/g1c_ho_s2 --holdout-base g1c --gen4-ckpt $F --seed 2 ) &
( clean "" $F0 none_clean && t5b $R/ft_all --holdout-base "" --init $G/B_none_clean/r0/model.pt --norm-from-init ) &
wait
log "ALL DONE"
