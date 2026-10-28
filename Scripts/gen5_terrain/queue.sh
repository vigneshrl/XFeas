#!/bin/bash
# gen5b terrain table queue (2026-10-28): Spot + G1 AMO x 7 tiles x 3 trials, up to SLOTS trials in parallel on GPU 0.
R=/p/cral/vignesh/ARL/Scripts; O=$R/data/gen5_terrain; mkdir -p $O; LOG=$O/queue.log; SLOTS=${SLOTS:-8}
echo $$ > $O/queue.pid; echo "$(date '+%m-%d %T') START pid $$ slots $SLOTS" >> $LOG
JOBS=()
for k in 0 1 2; do for c in spot amo; do for t in ice oil_spill sand_dunes rubble_rocky roots_leaves wet_tile sticky_clay; do JOBS+=("$c $t $k"); done; done; done
for j in "${JOBS[@]}"; do
  set -- $j; D=$O/${1}_${2}/t$3
  grep -q "RESULT\|: success\|: fall\|: stuck\|: timeout" $D/run.log 2>/dev/null && continue
  while true; do
    n=$(jobs -rp | wc -l); nth=$(ps -L -u $(id -un) --no-headers | wc -l)
    [ $n -lt $SLOTS ] && [ $nth -lt 1500 ] && break; sleep 10
  done
  echo "$(date '+%m-%d %T') RUN $1 $2 t$3" >> $LOG
  bash $R/scripts/gen5_terrain/trial.sh $1 $2 $3 $D &
  sleep 20
done
wait; echo "$(date '+%m-%d %T') ALL DONE" >> $LOG
