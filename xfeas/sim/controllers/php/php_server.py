"""stdin/stdout policy server for php_ref.PhpPolicy (run in the gr00t_wbc venv, which has onnxruntime).
Framing as xfeas/sim/controllers/sonic/sonic_server.py: uint32 LE length N, then N bytes = UTF-8 JSON header, b"\\0", raw LE float64.

Requests (header["op"]):
  reset : -> {"ok":1}                           (clears last action / depth frame; direction -> stand)
  dir   : header {"dir": name} | {"vx","vy","wz"} (release quantiser) -> {"ok":1, "dir": name}
  depth : array = metric z-depth (60*106, row-major; inf = no hit) -> {"ok":1}   (preprocessed as DepthShmPlugin)
  tick  : array = q(29) dq(29) pelvis_quat_wxyz(4) gyro_body(3) -> array q_target(29), header {"dir", "ms"}
  frame : -> array = the current preprocessed (58*87) frame
  quit
"""

import os, sys  # noqa: E401  (standalone process, possibly in an isolated env: find the repo root itself)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
import xfeas  # noqa: E402, F401  (puts every XFeas code folder on sys.path)
import json
import os
import struct
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np  # noqa: E402


def read_msg(f):
    h = f.read(4)
    if len(h) < 4:
        return None, None
    (n,) = struct.unpack("<I", h)
    buf = f.read(n)
    i = buf.index(b"\0")
    return json.loads(buf[:i].decode()), np.frombuffer(buf[i + 1 :], dtype="<f8")


def write_msg(f, header, arr=None):
    b = json.dumps(header).encode() + b"\0" + (np.asarray(arr, dtype="<f8").tobytes() if arr is not None else b"")
    f.write(struct.pack("<I", len(b)) + b)
    f.flush()


def main():
    fin, fout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr
    import php_ref as P

    pol = P.PhpPolicy(threads=int(os.environ.get("PHP_THREADS", "2")))
    write_msg(fout, {"ready": 1})
    while True:
        h, arr = read_msg(fin)
        if h is None or h.get("op") == "quit":
            break
        op = h["op"]
        try:
            if op == "tick":
                t = time.perf_counter()
                qt = pol.tick(arr[0:29], arr[29:58], arr[58:62], arr[62:65])
                write_msg(fout, {"dir": pol.direction, "ms": (time.perf_counter() - t) * 1e3}, qt)
            elif op == "depth":
                pol.set_depth(P.preprocess_depth(arr.reshape(P.CAM_H, P.CAM_W)))
                write_msg(fout, {"ok": 1})
            elif op == "dir":
                name = h.get("dir") or P.quantise(h.get("vx", 0.0), h.get("vy", 0.0), h.get("wz", 0.0), pol.speed_mode)
                pol.set_direction(name)
                write_msg(fout, {"ok": 1, "dir": name})
            elif op == "reset":
                pol.reset()
                write_msg(fout, {"ok": 1})
            elif op == "frame":
                write_msg(fout, {"ok": 1}, pol.frame if pol.frame is not None else np.zeros(0))
            else:
                write_msg(fout, {"error": f"unknown op {op}"})
        except Exception as e:
            write_msg(fout, {"error": repr(e)})


if __name__ == "__main__":
    main()
