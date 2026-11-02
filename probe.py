"""Run the controller self-test (probe) in Isaac Sim: what each of the controller's commands physically does.

    python probe.py --base g1c --ctrls g1c          # base controller + the variants to probe

Runs xfeas/data_collection/controller_selftest.py through the Isaac launcher on GPU 0. Results go to data/probes_v5/.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
launcher = os.path.join(ROOT, "xfeas", "data_collection", "run_isaac.sh")
script = os.path.join(ROOT, "xfeas", "data_collection", "controller_selftest.py")
os.execvp("bash", ["bash", launcher, "0", script, *sys.argv[1:]])
