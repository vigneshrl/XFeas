#!/usr/bin/env python
"""
Closed-loop pieces of the gen3 layer that the MODEL side owns (gen3 model agent, 2026-10-17). NOT RUN here (no Isaac).
The episode harness / executor (scripts/gen3/{probe,driver,execute,feas}.py) belongs to the data/infra agent; the arm that
wires these together for SONIC is listed in scripts/gen_train/gen3_sonic_eval.sh.

  HistoryRecorder   h(t) exactly as GEN3_SPEC section 6 stores it at a decision: 60 frames at 10 Hz, oldest first,
                    hist_cmd (60, 8) BY ROLE (lin_vel_x, lin_vel_y, yaw_rate, pelvis_height, torso_pitch, torso_roll,
                    mode index, speed; NaN = role absent in this spec), hist_env (60, 4), hist_ach (60, 12), hist_q / hist_dq,
                    hist_valid, hist_t. Feed it every control tick; it decimates to 10 Hz.
  Gen3Client        length-prefixed npz client of `select_gen3.py --serve` (the selector runs in the gsam2 env as a
                    sidecar, like select_gen2); request = obs keys + spec_json + probe_json + lat_room + gap_e + rule.
  StallWatch        gen2 F4: a running program with < 0.2 m lane progress in 5 s is ended -> re-decide (stock is
                    always a candidate).
  role_command      spec-level command dict -> the 8 role channels of hist_cmd (mode name -> index in spec['modes']).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import io, json, math, os, struct, subprocess, sys
from collections import deque
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
import gen3_common as G

PY = xfeas.paths.XFEAS_PY


def role_command(spec, cmd):
    """spec-level command dict (names of THAT spec) -> (8,) by HIST_CMD_ROLES; NaN = role absent in the spec, 0 = present
    but not sent this tick"""
    spec = G.load_spec(spec)
    out = np.full(8, np.nan, np.float32)
    by_role = {c["role"]: c for c in spec.get("commands", [])}
    for j, role in enumerate(G.HIST_CMD_ROLES):
        c = by_role.get(role)
        if c is None:
            continue
        v = cmd.get(c.get("name", role), None)
        if role == "mode":
            names = [m.get("name") for m in spec.get("modes", [])]
            out[j] = float(names.index(v)) if v in names else -1.0
        else:
            out[j] = (
                float(v) if v is not None else float(c.get("nominal", 0.0) if c.get("type") == "continuous" else 0.0)
            )
    return out


class HistoryRecorder:
    def __init__(self, joint_names, hz=10.0, n=G.T_HIST):
        self.jn, self.dt, self.n = list(joint_names), 1.0 / hz, n
        self.buf = deque(maxlen=n)
        self.t_last = -1e9

    def push(self, t, cmd_roles, env_intent, ach12, q, dq):
        """t (s); cmd_roles (8,) from role_command; env_intent (4,) v, top, wid, pitch in force (approach = stock:
        nominal v, 9.9, 9.9, 0); ach12 (12,) GEN3_SPEC 6 hist_ach channels; q / dq (J,)"""
        if t - self.t_last < self.dt - 1e-6:
            return
        self.t_last = t
        self.buf.append(
            (
                float(t),
                np.asarray(cmd_roles, np.float32),
                np.asarray(env_intent, np.float32),
                np.asarray(ach12, np.float32),
                np.asarray(q, np.float32),
                np.asarray(dq, np.float32),
            )
        )

    def arrays(self, t_now):
        """-> dict of the spec-6 history keys (frames before the robot's spawn are zeros, hist_valid 0)"""
        n, J = self.n, len(self.jn)
        out = dict(
            hist_t=np.zeros(n, np.float32),
            hist_valid=np.zeros(n, np.uint8),
            hist_cmd=np.full((n, 8), np.nan, np.float32),
            hist_env=np.zeros((n, 4), np.float32),
            hist_ach=np.zeros((n, 12), np.float32),
            hist_q=np.zeros((n, J), np.float32),
            hist_dq=np.zeros((n, J), np.float32),
            joint_names=json.dumps(self.jn),
        )
        k0 = n - len(self.buf)
        for k, (t, c, e, a, q, dq) in enumerate(self.buf):
            i = k0 + k
            out["hist_t"][i] = t - t_now
            out["hist_valid"][i] = 1
            out["hist_cmd"][i] = c
            out["hist_env"][i] = e
            out["hist_ach"][i] = a
            out["hist_q"][i] = q
            out["hist_dq"][i] = dq
        out["hist_t"][:k0] = (np.arange(k0) - n + 1) * self.dt
        return out


class StallWatch:
    def __init__(self, window_s=5.0, min_prog=0.2):
        self.w, self.mp = window_s, min_prog
        self.hist = deque()

    def reset(self):
        self.hist.clear()

    def update(self, t, s):
        """t (s), s lane progress (m) -> True when the running program has stalled"""
        self.hist.append((t, s))
        while self.hist and t - self.hist[0][0] > self.w:
            self.hist.popleft()
        return bool(self.hist and t - self.hist[0][0] >= self.w - 0.15 and s - self.hist[0][1] < self.mp)


class Gen3Client:
    def __init__(self, ckpt, env=None, agree_ckpt=""):
        e = dict(os.environ, CUDA_VISIBLE_DEVICES="0", **(env or {}))
        cmd = [PY, os.path.join(THIS, "select_gen3.py"), "--ckpt", ckpt, "--serve"] + (
            ["--agree-ckpt", agree_ckpt] if agree_ckpt else []
        )
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=e)

    def ask(self, obs, spec, probe, lat_room=(1.0, 1.0), gap_e=0.0, rule=False):
        b = io.BytesIO()
        np.savez(
            b,
            spec_json=json.dumps(spec),
            probe_json=json.dumps(probe),
            lat_room=np.asarray(lat_room, np.float32),
            gap_e=np.float32(gap_e),
            rule=np.int32(rule),
            **obs,
        )
        raw = b.getvalue()
        self.p.stdin.write(struct.pack("<I", len(raw)))
        self.p.stdin.write(raw)
        self.p.stdin.flush()
        n = struct.unpack("<I", self.p.stdout.read(4))[0]
        z = np.load(io.BytesIO(self.p.stdout.read(n)), allow_pickle=False)
        return json.loads(str(z["choice"]))

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(5)
        except Exception:
            self.p.kill()


def selftest():
    spec = G.g1c_fallback_spec()
    for c in spec["commands"]:
        c["name"] = "x_" + c["role"]
    r = role_command(spec, {"x_lin_vel_x": 0.3, "x_pelvis_height": 0.5})
    assert r[0] == np.float32(0.3) and r[3] == np.float32(0.5) and np.isnan(r[6]) and np.isnan(r[7]) and r[1] == 0.0
    H = HistoryRecorder(G.G1_JOINTS[:23])
    for k in range(200):
        H.push(k * 0.02, r, [0.3, 9.9, 9.9, 0.0], np.r_[0.25, np.zeros(11)], np.zeros(23), np.zeros(23))
    A = H.arrays(199 * 0.02)
    assert A["hist_valid"].sum() == 40 and A["hist_valid"][-1] == 1 and A["hist_valid"][0] == 0
    X, v = G.history_feats(A, spec)
    assert v.sum() == 40 and np.isfinite(X).all()
    sw = StallWatch()
    assert not any(sw.update(t * 0.1, 0.01 * t) for t in range(30)) and sw.update(5.1, 0.15)
    print("[gen3_closed_loop] selftest OK")


if __name__ == "__main__":
    selftest()
