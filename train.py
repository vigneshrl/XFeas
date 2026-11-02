"""Train XFeas, or evaluate a trained model offline. All arguments are passed to xfeas/train/trainer.py.

    python train.py --out data/gen_train/gen5b/runs/all \
        --gen4-ckpt data/gen_train/gen4/full/f_none/ckpt/xfeas_gen4_ens.pt
    python train.py --out data/gen_train/gen5b/runs/g1c_ho --holdout-base g1c \
        --gen4-ckpt data/gen_train/gen4/full/f_g1c/ckpt/xfeas_gen4_ens.pt      # leave AMO out of training
    python train.py --eval-only --ckpt data/gen_train/gen5b/runs/all/model.pt --eval-split val
"""

import runpy
import sys

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)

sys.argv[0] = "trainer.py"
runpy.run_module("trainer", run_name="__main__", alter_sys=True)
