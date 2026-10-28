#!/bin/bash
# restarts run_g5b_closed.sh (restart-safe) if its supervisor died before ALL DONE; logs a status line every 10 min
cd /p/cral/vignesh/ARL/Scripts
Q=data/gen5_closed/queue5b; echo $$ > $Q/watchdog.pid
while ! grep -q "ALL DONE" $Q/queue.log 2>/dev/null; do
  if ! kill -0 $(cat $Q/supervisor.pid 2>/dev/null) 2>/dev/null; then
    echo "$(date '+%m-%d %T') watchdog: supervisor gone -> restart" >> $Q/queue.log
    SLOTS=5 setsid nohup bash scripts/gen5/run_g5b_closed.sh >> data/gen5_closed/logs/run_g5b_closed.out 2>&1 < /dev/null &
    sleep 30
  fi
  echo "$(date '+%m-%d %T') done $(ls $Q/*.done 2>/dev/null | wc -l) failed $(ls $Q/*.failed 2>/dev/null | wc -l)" > $Q/status.txt
  sleep 120
done
