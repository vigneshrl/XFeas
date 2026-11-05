#!/bin/bash
# 11-01 overnight chain (user go): s5 collection (job_queue, staged stand-up option stockS) -> manifest v4 + data-QA gate ->
# retrain all_g1s_v4 (as v3) -> closed loop GC2F rules + model v4, 10 trials. GPU 0 only. QA FAIL -> no training.
# usage: setsid nohup bash xfeas/train/run_sonic_retrain.sh > data/gen_train/gen5b/runs/overnight_v4.log 2>&1 &
source "$(dirname "$(readlink -f "$0")")/../../env.sh"; cd "$XFEAS_ROOT"
export CUDA_VISIBLE_DEVICES=0
PY=$XFEAS_PY
G=data/gen_train/gen5b; O=$G/runs/all_g1s_v4; M=$G/manifest_g1s_v4.jsonl
echo $$ > $G/runs/overnight_v4.pid
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
left() { $PY -c "import json;st=json.load(open('data/gen_v5b/queue/state.json'));print(sum(v['status'] in ('pending','running') for k,v in st.items() if '_s5_' in k))"; }
log "waiting for s5 collection"
while [ "$(left)" != "0" ]; do sleep 300; done
log "s5 collection finished: $(grep -c '_s5_.*-> done' data/gen_v5b/queue/queue.log) done; $(grep '_s5_' data/gen_v5b/queue/QUARANTINE.txt 2>/dev/null | wc -l) quarantined"
$PY xfeas/train/branch_dataset.py build --out $M.raw
python3 xfeas/train/build_manifest_v4.py $M.raw $M || { log "QA FAIL -> no training (see $M.qa.json)"; exit 1; }
log "QA PASS -> training $O"
mkdir -p $O; n=0
while [ ! -f $O/model.pt ] && [ $n -lt 10 ]; do
  n=$((n+1)); log "train launch $n"
  $PY xfeas/train/trainer.py --out $O --holdout-base "" --gen4-ckpt data/gen_train/gen4/full/f_none/ckpt/xfeas_gen4_ens.pt \
      --manifest $M --workers 8 >> $O/train.log 2>&1
  [ -f $O/model.pt ] || sleep 30
done
[ -f $O/model.pt ] || { log "GAVE UP training"; exit 1; }
log "trained; closed loop GC2F rules + model v4, 10 trials"
G5S_CKPT=$(realpath $O/model.pt) bash xfeas/closed_loop/run_school_cell.sh c5_sonic_g5sr4_GC2F_n10 sonic g5sr 10 GC2F > data/gen5_g5sr/logs/ev_g5sr4_GC2F_n10.log 2>&1
log "ALL DONE: $(grep -E '^\[ep\].*trial' data/gen5_g5sr/logs/ev_g5sr4_GC2F_n10.log | grep -c success)/10 success"
