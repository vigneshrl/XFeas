#!/bin/bash
# gen5b closed-loop matrix (10-27 evening). Restart-safe queue: a cell is DONE when its runs_<robot>.jsonl has N rows.
# Up to SLOTS cells at once, GPU 0 only (p5_cell.sh). One retry per cell. Never kills anything it did not start.
source "$(dirname "$(readlink -f "$0")")/../env.sh"; cd "$XFEAS_ROOT"
G5=data/gen5_closed; L=$G5/logs; Q=$G5/queue5b; mkdir -p $L $Q
N=3; SLOTS=${SLOTS:-6}
CELLS=(
 "c5_amo_g5b_O1L amo g5b SC2A_O1L"   "c5_amo_g5bho_O1L amo g5bho SC2A_O1L" "c5_amo_stock_O1L amo stock SC2A_O1L"
 "c5_amo_g5b_O1 amo g5b SC2A_O1"     "c5_amo_g5bho_O1 amo g5bho SC2A_O1"   "c5_amo_stock_O1 amo stock SC2A_O1"
 "c5_amo_g5b_O1N amo g5b SC2A_O1N"   "c5_amo_g5bho_O1N amo g5bho SC2A_O1N" "c5_amo_stock_O1N amo stock SC2A_O1N"
 "c5_amo_gen5_O1L amo gen5 SC2A_O1L" "c5_amo_gen5_O1 amo gen5 SC2A_O1"
 "c5_spot_g5b_O1L spot g5b SC2A_O1L" "c5_spot_stock_O1L spot stock SC2A_O1L" "c5_spot_gen5_O1L spot gen5 SC2A_O1L"
 "c5_spot_g5b_O1 spot g5b SC2A_O1"   "c5_spot_stock_O1 spot stock SC2A_O1"   "c5_spot_gen5_O1 spot gen5 SC2A_O1"
 "c5_spot_g5b_O1N spot g5b SC2A_O1N" "c5_spot_stock_O1N spot stock SC2A_O1N"
)
rob_of() { [ "$1" = amo ] && echo g1a || echo spotp; }
nrows() { local f=$G5/course/$1/runs_$(rob_of $2).jsonl; [ -f $f ] && grep -c . $f || echo 0; }
alive() { [ -f $Q/$1.pid ] && kill -0 $(cat $Q/$1.pid) 2>/dev/null; }
log() { echo "$(date '+%m-%d %T') $*" >> $Q/queue.log; }
echo $$ > $Q/supervisor.pid
log "supervisor start pid $$ slots $SLOTS"
while true; do
  running=0; pending=0
  for c in "${CELLS[@]}"; do
    set -- $c; jid=$1
    if alive $jid; then running=$((running+1)); continue; fi
    n=$(nrows $jid $2)
    if [ "$n" -ge $N ]; then [ -f $Q/$jid.done ] || { touch $Q/$jid.done; log "DONE $jid ($n runs)"; }; continue; fi
    tries=$(cat $Q/$jid.tries 2>/dev/null || echo 0)
    if [ "$tries" -ge 2 ]; then [ -f $Q/$jid.failed ] || { touch $Q/$jid.failed; log "FAILED $jid after $tries tries ($n runs)"; }; continue; fi
    pending=$((pending+1))
  done
  [ $running -eq 0 ] && [ $pending -eq 0 ] && { log "ALL DONE"; break; }
  for c in "${CELLS[@]}"; do
    [ $running -ge $SLOTS ] && break
    nth=$(ps -L -u $(id -un) --no-headers | wc -l)          # RLIMIT_NPROC 2000 per user: ~230 threads per cell
    [ $nth -gt 1500 ] && { log "hold: $nth user threads"; break; }
    set -- $c; jid=$1
    alive $jid && continue
    [ -f $Q/$jid.done ] || [ -f $Q/$jid.failed ] && continue
    [ "$(nrows $jid $2)" -ge $N ] && continue
    tries=$(cat $Q/$jid.tries 2>/dev/null || echo 0); [ "$tries" -ge 2 ] && continue
    echo $((tries+1)) > $Q/$jid.tries
    # a retry re-runs all N trials of the cell into a fresh directory (the runner appends; keep the partial run aside)
    [ -d $G5/course/$jid ] && mv $G5/course/$jid $G5/course/${jid}_try$tries
    [ -f $L/$jid.log ] && mv $L/$jid.log $L/$jid.try$tries.log
    [ -d $G5/diag/$jid ] && mv $G5/diag/$jid $G5/diag/${jid}_try$tries
    rm -f $Q/$jid.pid
    setsid nohup bash -c "echo \$\$ > $Q/$jid.pid; exec bash closed_loop/p5_cell.sh $jid $2 $3 $N $4" > $L/$jid.log 2>&1 < /dev/null &
    for _ in 1 2 3 4 5 6 7 8 9 10; do [ -s $Q/$jid.pid ] && break; sleep 1; done
    running=$((running+1)); log "START $jid try $((tries+1)) pid $(cat $Q/$jid.pid 2>/dev/null)"
    sleep 45
  done
  sleep 60
done
