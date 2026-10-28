"""Tiny stdin/stdout policy server for SonicController (run in the venv that has onnxruntime).

Framing (both directions): uint32 little-endian length N, then N bytes of payload.
Payload: UTF-8 JSON header, a single b"\\0", then raw little-endian float64 array bytes (may be empty).

Requests (header["op"]):
  reset : array = q_mj(29) + base_quat_wxyz(4)                       -> reply {"ok":1}
  cmd   : header {"mode","vx","vy","wz","heading","height","speed"}  -> reply {"ok":1}
  raw   : header {"mode","movement","facing","speed","height"} (planner frame, ZMQManager semantics)
  tick  : array = q(29) dq(29) quat_wxyz(4) gyro_body(3)              -> array q_target(29),
          header {"frame","planner_yaw_offset","ref_root_z"}
  info  : -> header with gains/default pose/timing
  quit
Client: sonic_client.SonicClient (numpy only; works from Isaac kit python).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import argparse
import json
import os
import struct
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_msg(f):
    h = f.read(4)
    if len(h) < 4:
        return None, None
    (n,) = struct.unpack("<I", h)
    buf = f.read(n)
    i = buf.index(b"\0")
    return json.loads(buf[:i].decode()), np.frombuffer(buf[i + 1:], dtype="<f8")


def write_msg(f, header, arr=None):
    b = json.dumps(header).encode() + b"\0" + (np.asarray(arr, dtype="<f8").tobytes() if arr is not None else b"")
    f.write(struct.pack("<I", len(b)) + b)
    f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=xfeas_paths.BASELINES + "/GEAR-SONIC")
    ap.add_argument("--variant", default="default")
    ap.add_argument("--planner_async", action="store_true", help="run planner in a background thread (C++-like)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    fin, fout = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr  # keep the binary channel clean
    from sonic_ref import SonicController, KPS, KDS, DEFAULT_ANGLES, G1_ACTION_SCALE
    c = SonicController(a.model_dir, a.variant, planner_async=a.planner_async, intra_op_threads=a.threads)
    write_msg(fout, {"ready": 1})
    while True:
        h, arr = read_msg(fin)
        if h is None or h.get("op") == "quit":
            break
        op = h["op"]
        try:
            if op == "tick":
                qt = c.control_tick(arr[0:29], arr[29:58], arr[58:62], arr[62:65])
                write_msg(fout, {"frame": c.current_frame, "planner_yaw_offset": c.planner_to_world_yaw(),
                                 "ref_root_z": float(c.motion.pos[c.current_frame][2])}, qt)
            elif op == "reset":
                c.reset(arr[0:29], arr[29:33]); write_msg(fout, {"ok": 1})
            elif op == "cmd":
                c.set_command(h["mode"], h.get("vx", 0.0), h.get("vy", 0.0), h.get("wz"), h.get("heading"),
                              h.get("height", -1.0), h.get("speed")); write_msg(fout, {"ok": 1})
            elif op == "raw":
                c.set_planner_command(h["mode"], h.get("movement", (0, 0, 0)), h.get("facing", (1, 0, 0)),
                                      h.get("speed", -1.0), h.get("height", -1.0)); write_msg(fout, {"ok": 1})
            elif op == "info":
                t = c.timing
                write_msg(fout, {"kp": KPS.tolist(), "kd": KDS.tolist(), "default": DEFAULT_ANGLES.tolist(),
                                 "action_scale": G1_ACTION_SCALE.tolist(),
                                 "planner_ms_median": float(np.median(t["planner_ms"])) if t["planner_ms"] else None,
                                 "policy_ms_median": float(np.median(t["policy_ms"])) if t["policy_ms"] else None,
                                 "n_planner_calls": len(t["planner_ms"])})
            else:
                write_msg(fout, {"error": f"unknown op {op}"})
        except Exception as e:  # report, keep serving
            write_msg(fout, {"error": repr(e)})


if __name__ == "__main__":
    main()
