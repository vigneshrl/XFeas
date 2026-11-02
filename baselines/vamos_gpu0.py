"""
VAMOS (released) exactly as baselines/vamos/arl_adapter/vamos_plugin.VAMOS, with ONE change: its VLM sidecar is
started on GPU 0 instead of the hard-coded CUDA_VISIBLE_DEVICES="1" (machine rule: GPU 0 only). Same code, weights,
config, sensors and controller; the device does not change the method.
    PYTHONPATH=...:$XFEAS_BASELINES/vamos/arl_adapter  ... --method vamos_gpu0:VAMOSGPU0
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os, subprocess, types
import vamos_plugin as VP


class VAMOSGPU0(VP.VAMOS):
    def setup(self, ctx):
        real = subprocess.Popen

        def popen(*a, **kw):
            env = dict(kw.get("env") or os.environ)
            env["CUDA_VISIBLE_DEVICES"] = "0"
            kw["env"] = env
            return real(*a, **kw)

        VP.subprocess = types.SimpleNamespace(Popen=popen, PIPE=subprocess.PIPE)
        try:
            super().setup(ctx)
        finally:
            VP.subprocess = subprocess
        print("[vamos_gpu0] sidecar on CUDA_VISIBLE_DEVICES=0", flush=True)
