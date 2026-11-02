"""Collect side-by-side training data in Isaac Sim (one job = one controller, a chunk of one suite).

    python collect.py --env school --base g1c --suite data/gen_v5b/suites/school_g1c_b1.json \
        --chunk 0:4 --part school_g1c_b1_0000 [--smoke]

Runs xfeas/data_collection/branch_collector.py through the Isaac launcher on GPU 0. For a full production run use the
queue instead:  python xfeas/data_collection/make_suites.py --groups a,b,c --write-jobs
                python xfeas/data_collection/job_queue.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
launcher = os.path.join(ROOT, "xfeas", "data_collection", "run_isaac.sh")
script = os.path.join(ROOT, "xfeas", "data_collection", "branch_collector.py")
os.execvp("bash", ["bash", launcher, "0", script, *sys.argv[1:]])
