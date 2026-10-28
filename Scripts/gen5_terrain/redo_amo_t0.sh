#!/bin/bash
# AMO t0 trials ran before TERRAIN_FALL_TILT (01:45:37): wait for the main queue, set them aside, re-run via queue.sh
R=/p/cral/vignesh/ARL/Scripts; O=$R/data/gen5_terrain
while kill -0 $(cat $O/queue.pid) 2>/dev/null; do sleep 20; done
for d in $O/amo_*/t0; do [ -d $d ] && mv $d ${d}_oldrule; done
echo "$(date '+%m-%d %T') REDO amo t0 (old fall rule set aside)" >> $O/queue.log
exec bash $R/scripts/gen5_terrain/queue.sh
