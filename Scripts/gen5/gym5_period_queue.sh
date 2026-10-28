#!/bin/bash
# decision-interval sweep (2026-10-28): AMO + SONIC x GC2F + CC1F x period 1 / 2.5 / 3 / 5 s, 1 trial each, GPU 0, <= SLOTS at once
R=/p/cral/vignesh/ARL/Scripts; G=$R/data/gen5_gym; LOG=$G/period_queue.log; SLOTS=${SLOTS:-8}
echo $$ > $G/period_queue.pid; echo "$(date '+%m-%d %T') START pid $$" >> $LOG
for P in 1 2.5 3 5; do for c in sonic amo; do for s in GC2F CC1F; do
  J=gy_${c}_${s}_p${P/./}
  [ -f $G/videos/*_${s}_${J}_t0.mp4 ] 2>/dev/null && continue
  while [ $(jobs -rp | wc -l) -ge $SLOTS ] || [ $(ps -L -u $(id -un) --no-headers | wc -l) -gt 1500 ]; do sleep 10; done
  echo "$(date '+%m-%d %T') RUN $J" >> $LOG
  XFEAS_GEN5_PERIOD=$P bash $R/scripts/gen5/gym5_cell.sh $J $c $s 1 > $G/logs/$J.log 2>&1 &
  sleep 20
done; done; done
wait; echo "$(date '+%m-%d %T') ALL DONE" >> $LOG
