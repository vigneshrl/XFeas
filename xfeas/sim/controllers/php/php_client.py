"""numpy-only client for php_server.py (usable from Isaac kit python, which lacks onnxruntime). Lock-step: one tick per
50 Hz control step. Joint order everywhere: MJCF G1 29-DoF order (php_ref.JOINT_NAMES)."""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json
import os
import struct
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = xfeas.paths.GR00T_PY


class PhpClient:
    def __init__(self, python=VENV_PY, threads=2):
        env = {k: os.environ[k] for k in ("PATH", "HOME", "USER", "TMPDIR", "LANG") if k in os.environ}
        env.update(CUDA_VISIBLE_DEVICES="0", PHP_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
        self.p = subprocess.Popen(
            [python, "-I", os.path.join(HERE, "php_server.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
            bufsize=0,
            cwd=HERE,
        )
        h, _ = self._recv()
        assert h.get("ready"), h

    def _send(self, header, arr=None):
        b = json.dumps(header).encode() + b"\0" + (np.asarray(arr, dtype="<f8").tobytes() if arr is not None else b"")
        self.p.stdin.write(struct.pack("<I", len(b)) + b)
        self.p.stdin.flush()

    def _recv(self):
        (n,) = struct.unpack("<I", self._read(4))
        buf = self._read(n)
        i = buf.index(b"\0")
        h = json.loads(buf[:i].decode())
        if "error" in h:
            raise RuntimeError(h["error"])
        return h, np.frombuffer(buf[i + 1 :], dtype="<f8")

    def _read(self, n):
        out = b""
        while len(out) < n:
            chunk = self.p.stdout.read(n - len(out))
            if not chunk:
                raise EOFError("php_server exited")
            out += chunk
        return out

    def reset(self):
        self._send({"op": "reset"})
        return self._recv()[0]

    def direction(self, name=None, vx=0.0, vy=0.0, wz=0.0):
        self._send({"op": "dir", "dir": name, "vx": float(vx), "vy": float(vy), "wz": float(wz)})
        return self._recv()[0]["dir"]

    def depth(self, z_metric):
        self._send({"op": "depth"}, np.asarray(z_metric, float).ravel())
        return self._recv()[0]

    def frame(self):
        return self._recv_after({"op": "frame"})

    def _recv_after(self, h):
        self._send(h)
        return self._recv()[1]

    def tick(self, q, dq, quat, gyro):
        self._send({"op": "tick"}, np.concatenate([q, dq, quat, gyro]))
        h, a = self._recv()
        return a.copy(), h

    def close(self):
        try:
            self._send({"op": "quit"})
            self.p.wait(timeout=10)
        except Exception:
            self.p.kill()
