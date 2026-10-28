#!/usr/bin/env python
"""
GEN_V2_SPEC section 3 action encoding, model side (gen2 agent, 2026-10-14): build act_v2 vectors for candidate skills
and RENDER their (10, 7) chunks exactly as the collector does (section 3.3 as implemented in scripts/gen/run_gen_v2.render_chunk; checked on real gen_v2 rollouts), so the selector scores candidates in the
training encoding. Pure numpy.

  go       act_v2 idx 0-3 (= 4-7)                                  chunk rows (v, 0, 0, 0, dh, p, r)
  go2      A = idx 4-7, B = idx 0-3, d_commit 8, d_level 9, yaw_off 11, d_yaw 15
           row k: A if 0.3 k < t_c else B, t_c = max(0, d_dec - d_commit) / v_A; nominal posture at speed v_B for rows with
           0.3 k >= (d_dec + d_level) / v_B when d_level > 0; yaw_off: wz = yaw_off / 0.3 in the commit row and
           -yaw_off / 0.3 in the row where it ends (d_yaw past the front edge)
  detour   v, posture idx 0-3, lat_off 10, head_off 11 = atan2(lat_off, L_ramp), L_ramp 13, s_back 14:
           e_ref(s) ramps 0 -> lat_off over L_ramp from the decision, holds, ramps back after s_end + s_back;
           wz_k = (psi(0.3 (k+1)) - psi(0.3 k)) / 0.3, psi(t) = atan(de_ref/ds) at s = v t
  sidestep v_nom 0, d_trig 8, e_gap 10, -side pi/2 11, side vy 12; rows 0-2 turn, 3-8 sideways, 9 turn back
SKILL ids: go 0, go2 3, sidestep 4, detour 7.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math
import numpy as np

DT, K = 0.3, 10
SKILL = dict(go=0, go2=3, sidestep=4, detour=7)
MASK = dict(go=[0, 1, 2, 3], go2=list(range(10)), detour=[0, 1, 2, 3, 10, 11, 13, 14], sidestep=[0, 8, 10, 11, 12])


def make(skill, **kw):
    """-> (act_v2 (16,), mask (16,)); posture values: dh (m, relative to nominal), pitch, roll"""
    A = np.zeros(16, np.float32)
    if skill == "go":
        A[0:4] = [kw["v"], kw.get("dh", 0.0), kw.get("pitch", 0.0), kw.get("roll", 0.0)]; A[4:8] = A[0:4]
    elif skill == "go2":
        A[0:4] = [kw["v_b"], kw.get("dh_b", 0.0), kw.get("pitch_b", 0.0), kw.get("roll_b", 0.0)]
        A[4:8] = [kw["v_a"], kw.get("dh_a", 0.0), kw.get("pitch_a", 0.0), kw.get("roll_a", 0.0)]
        A[8], A[9], A[11], A[15] = kw.get("d_commit", 0.0), kw.get("d_level", 0.0), kw.get("yaw_off", 0.0), kw.get("d_yaw", 0.0)
    elif skill == "detour":
        A[0:4] = [kw["v"], kw.get("dh", 0.0), kw.get("pitch", 0.0), kw.get("roll", 0.0)]; A[4:8] = A[0:4]
        A[10], A[13], A[14] = kw["lat_off"], kw.get("L_ramp", 1.2), kw.get("s_back", 0.5)
        A[11] = math.atan2(A[10], A[13])
    elif skill == "sidestep":
        A[0], A[8], A[10] = kw["v_nom"], kw["d_trig"], kw.get("e_gap", 0.0)
        A[11] = -kw["side"] * math.pi / 2; A[12] = kw["side"] * kw["vy"]
    M = np.zeros(16, np.float32); M[MASK[skill]] = 1.0
    if skill == "go2" and abs(A[11]) > 1e-9:
        M[[11, 15]] = 1.0
    return A, M


def render(skill, A, d_dec=1.5, s_len=0.5, g1=False, wz_max=1.0):
    """-> (10, 7) chunk (vx, vy, vz, wz, dh, pitch, roll); d_dec = base -> hazard front (m), s_len = hazard length along
    the lane (detour: s_end = s_dec + d_dec + s_len)"""
    ch = np.zeros((K, 7), np.float32)
    if skill == "go":
        ch[:, 0] = A[0]; ch[:, 4:7] = A[1:4]
    elif skill == "go2":                                    # = scripts/gen/run_gen_v2.render_chunk (the data's own rule)
        vA, vB = max(float(A[4]), 0.05), max(float(A[0]), 0.05)
        tc = max(0.0, d_dec - float(A[8])) / vA
        for k in range(K):
            t = DT * k
            if t < tc:
                ch[k, 0] = A[4]; ch[k, 4:7] = A[5:8]
            else:
                ch[k, 0] = A[0]; ch[k, 4:7] = A[1:4]
                if A[9] > 0 and t >= tc + (float(A[8]) + float(A[9])) / vB:
                    ch[k, 4:7] = 0.0
        if abs(float(A[11])) > 1e-9:
            kc = int(tc / DT)
            ke = int((tc + (float(A[8]) + float(A[15])) / vB) / DT)
            if kc < K:
                ch[kc, 3] += A[11] / DT
            if ke < K:
                ch[ke, 3] -= A[11] / DT
    elif skill == "detour":                                 # = run_gen_v2.render_chunk: heading offset while 0.1 <= v t <= 0.1 + L
        v, lat, L = float(A[0]), float(A[10]), max(float(A[13]), 1e-3)
        psi = lambda t: math.atan(lat / L) if 0.1 <= v * t <= 0.1 + L else 0.0
        ch[:, 0] = v; ch[:, 4:7] = A[1:4]
        for k in range(K):
            ch[k, 3] = (psi(DT * (k + 1)) - psi(DT * k)) / DT
    elif skill == "sidestep":
        side = -math.copysign(1.0, float(A[11])) if abs(float(A[11])) > 1e-9 else 1.0
        vy = abs(float(A[12]))
        vxm = 0.1 if g1 else 0.0
        ch[0:3, 0] = vxm; ch[0:3, 3] = -side * wz_max
        ch[3:9, 1] = side * vy
        ch[9, 0] = vxm; ch[9, 3] = side * wz_max
    return ch


def selftest():
    A, M = make("sidestep", v_nom=0.3, d_trig=0.6, e_gap=0.06, side=1.0, vy=0.25)
    ch = render("sidestep", A, g1=True)
    assert ch[0, 3] == -1 and np.isclose(ch[3, 1], 0.25) and ch[9, 3] == 1 and np.isclose(ch[0, 0], 0.1)
    A, M = make("go2", v_a=0.3, v_b=0.9, pitch_b=-0.35, dh_b=-0.12, d_commit=0.3)
    ch = render("go2", A, d_dec=0.6)                  # t_c = (0.6 - 0.3) / 0.3 = 1.0 s -> rows 4.. phase B
    assert np.isclose(ch[3, 0], 0.3) and np.isclose(ch[4, 0], 0.9) and np.isclose(ch[4, 5], -0.35), ch
    A, M = make("detour", v=0.6, lat_off=0.6, L_ramp=1.2, s_back=0.5)
    ch = render("detour", A, d_dec=1.5, s_len=0.5)
    assert np.isclose(ch[0, 3], math.atan(0.5) / DT) and np.isclose(ch[7, 3], -math.atan(0.5) / DT), ch[:, 3]
    print("[act_v2] selftest OK")


if __name__ == "__main__":
    selftest()
