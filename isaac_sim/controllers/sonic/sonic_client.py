"""numpy-only client for sonic_server.py (usable from Isaac kit python, which lacks onnxruntime).

    from sonic_client import SonicClient
    cli = SonicClient()                          # spawns the server in the gr00t_wbc venv
    cli.reset(q_mj, quat_wxyz)
    cli.cmd("slow_walk", vx=0.5)
    q_target, info = cli.tick(q_mj, dq_mj, quat_wxyz, gyro_body)   # every 20 ms
    # apply tau = kp*(q_target-q) - kd*dq at the physics rate (gains: cli.info()["kp"/"kd"])
Joint order everywhere: MuJoCo/URDF G1 29-DoF order (see sonic_ref.MJ_JOINT_NAMES).

Run as a script to measure round-trip latency.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import json
import os
import struct
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PY = xfeas_paths.GR00T_PY


class SonicClient:
    def __init__(self, python=VENV_PY, planner_async=True, threads=4, extra_args=()):
        args = [python, os.path.join(HERE, "sonic_server.py"), "--threads", str(threads), *extra_args]
        if planner_async:
            args.append("--planner_async")
        # minimal environment: don't leak Isaac kit's PYTHONPATH / PYTHONHOME / LD_LIBRARY_PATH / OMP settings
        env = {k: os.environ[k] for k in ("PATH", "HOME", "USER", "TMPDIR", "LANG") if k in os.environ}
        env["CUDA_VISIBLE_DEVICES"] = "0"
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env, bufsize=0)
        h, _ = self._recv()
        assert h.get("ready"), h

    def _send(self, header, arr=None):
        b = json.dumps(header).encode() + b"\0" + (np.asarray(arr, dtype="<f8").tobytes() if arr is not None else b"")
        self.p.stdin.write(struct.pack("<I", len(b)) + b); self.p.stdin.flush()

    def _recv(self):
        hd = self._read(4)
        (n,) = struct.unpack("<I", hd)
        buf = self._read(n); i = buf.index(b"\0")
        h = json.loads(buf[:i].decode())
        if "error" in h:
            raise RuntimeError(h["error"])
        return h, np.frombuffer(buf[i + 1:], dtype="<f8")

    def _read(self, n):
        out = b""
        while len(out) < n:
            chunk = self.p.stdout.read(n - len(out))
            if not chunk:
                raise EOFError("sonic_server exited")
            out += chunk
        return out

    def reset(self, q, quat):
        self._send({"op": "reset"}, np.concatenate([q, quat])); return self._recv()[0]

    def cmd(self, mode, vx=0.0, vy=0.0, wz=None, heading=None, height=-1.0, speed=None):
        self._send({"op": "cmd", "mode": mode, "vx": vx, "vy": vy, "wz": wz, "heading": heading, "height": height, "speed": speed})
        return self._recv()[0]

    def raw(self, mode, movement=(0, 0, 0), facing=(1, 0, 0), speed=-1.0, height=-1.0):
        self._send({"op": "raw", "mode": mode, "movement": list(movement), "facing": list(facing), "speed": speed, "height": height})
        return self._recv()[0]

    def tick(self, q, dq, quat, gyro):
        self._send({"op": "tick"}, np.concatenate([q, dq, quat, gyro]))
        h, a = self._recv()
        return a.copy(), h

    def info(self):
        self._send({"op": "info"}); return self._recv()[0]

    def close(self):
        try:
            self._send({"op": "quit"}); self.p.wait(timeout=10)
        except Exception:
            self.p.kill()


if __name__ == "__main__":
    # latency benchmark inside a real MuJoCo loop (sim stepped here, policy in the server)
    import argparse
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--ticks", type=int, default=750)
    a = ap.parse_args()
    sys.path.insert(0, HERE)
    from sonic_mujoco_ref import Sim, DECIM, ground_start
    sim = Sim(); d = sim.d
    ground_start(sim)
    cli = SonicClient(planner_async=not a.sync)
    q, dq, quat, gyro = sim.state(); cli.reset(q, quat)
    rt = []
    for k in range(a.ticks):
        if k == 50: cli.cmd("slow_walk", vx=0.5)
        if k == 350: cli.cmd("slow_walk", vx=0.3, wz=0.3)
        q, dq, quat, gyro = sim.state()
        t = time.perf_counter(); qt, h = cli.tick(q, dq, quat, gyro); rt.append((time.perf_counter() - t) * 1e3)
        sim.q_target = qt
        for _ in range(DECIM):
            sim.step()
    info = cli.info(); cli.close()
    rt = np.array(rt)
    print(json.dumps(dict(mode="sync" if a.sync else "async", ticks=a.ticks, rt_ms_median=float(np.median(rt)),
                          rt_ms_p90=float(np.percentile(rt, 90)), rt_ms_p99=float(np.percentile(rt, 99)),
                          rt_ms_max=float(rt.max()), frac_over_20ms=float((rt > 20).mean()),
                          final_pelvis_z=float(d.qpos[2]), final_xy=[float(d.qpos[0]), float(d.qpos[1])],
                          server_policy_ms_median=info["policy_ms_median"], server_planner_ms_median=info["planner_ms_median"],
                          n_planner_calls=info["n_planner_calls"])))
