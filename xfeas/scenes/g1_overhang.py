"""
G1 overhang study (humanoid under the same overhang types Spot crouches through).

  build_d1g  scene D1g (episode_runner SCENES['D1g']): a primitive slab with D1's door footprint (0.75 m deep,
             0.12 m thick, 2.4 m wide; in memory, assets/school.usd untouched) across a G1-sized lane, underside at
             SCENES['D1g']['u'] -- inside G1's own crouch band measured on a training lane (g1_overhang_probe.py).
             Lane: upper floor, (-10.05, -6.35) -> (-4.55, -6.35); >= 6.3 m from every training/validation lane
             and every same-floor test scene (T1/T0/H1/H2/V*/W2/S*/N*), inside the D1 exclusion rectangle (no data
             was ever collected there); D1 itself is on the floor below (4.4 m lower).
  StraightCrouchPitchG1  ORACLE arm: lowest body height + a fixed torso pitch (env G1_ORACLE_PITCH, chosen on the
             TRAINING lane oh0003 by g1_overhang_probe.py --phase pass, never on D1/T1/D1g), straight drive.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math, os
import numpy as np

D1_LEAF = "BP_school_door_5/SM_school_door_3_3/InstancedSM_school_door_3_04"


def build_d1g(stage, sid, sc):
    """REAL (default since 2026-10-10): a real panelled school door leaf (BP_school_door_13 ... InstancedSM_school_door_4_2,
    2.58 x 0.73 x 0.14 m; it lies diagonally in the school, aligned by its minimum-area frame) laid flat ACROSS the
    lane, uniform scale 1.026 -> 0.75 m deep (D1's door span on its lane) x 2.65 m wide, lowest point at SCENES['D1g']['u']
    above the floor, front edge at s_front. Legacy primitive slab: XFEAS_LEGACY_PRIMITIVES=1."""
    import real_assets as RA

    if RA.LEGACY:
        return build_d1g_primitive(stage, sid, sc)
    lib = RA.RealLib(stage, root=f"/World/G1Scene_{sid}")
    P0 = np.asarray(sc["start"], float)
    g = np.asarray(sc["goal"], float)
    yaw = math.atan2(*(g - P0)[::-1])
    d = np.array([math.cos(yaw), math.sin(yaw)])
    Lw, Dw, Tw = lib.flat_dims("door_leaf")
    s = float(np.clip(0.75 / Dw, *RA.SCALE_LIM))
    s0 = sc.get("s_front", 3.0)
    xy = P0 + d * (s0 + 0.5 * Dw * s)
    rec = lib.place(
        "door_leaf", xy[0], xy[1], sc["floor"] + sc["u"], "flat", yaw + math.pi / 2 + lib.flat_yaw("door_leaf"), s
    )
    # NO FLOATING OBJECTS: the leaf rests on two real support columns beside the lane (tops at the underside height)
    sup = RA.supports_under(lib, rec["copy"]["prim"], P0, yaw, sc["floor"], room=(0.6, 0.6))  # tops = the leaf's lowest
    if any(x is None for x in sup):  # vertex over each column
        raise RuntimeError("D1g: door leaf does not reach both support sites")
    geo = dict(
        supports=sup,
        kind="door_leaf",
        scale=round(s, 4),
        depth=round(Dw * s, 3),
        width=round(Lw * s, 3),
        thick=round(Tw * s, 3),
        u=sc["u"],
        start=P0.tolist(),
        yaw=round(yaw, 4),
        aabb=[np.round(rec["mn"], 3).tolist(), np.round(rec["mx"], 3).tolist()],
    )
    print(f"[g1_overhang] {sid}: REAL door leaf, underside {sc['u']:.2f} m, {geo}", flush=True)
    return geo


def build_d1g_primitive(stage, sid, sc):
    """Primitive slab (overhang_lib 'prim' family, analytic box collider) with D1's door footprint on the lane:
    depth 0.75 m along the lane (D1 door span on its lane: y -2.45..-1.70), 0.12 m thick, 2.4 m wide, wood colour.
    (A COPY of D1's door leaf was tried first and composed to an object with no effective collider/visual on the
    lane -- stock G1 walked through it -- so the slab type is used, as for V3/H2.)"""
    import obstacle_lib as OB

    P0 = np.asarray(sc["start"], float)
    g = np.asarray(sc["goal"], float)
    yaw = math.atan2(*(g - P0)[::-1])
    pool = OB.BoxPool(stage, n=1, root=f"/World/G1Scene_{sid}")
    boxes = OB.overhang(sc.get("s_front", 3.0), sc["u"], depth=0.75, thick=0.12, width=2.4)
    pool.place(boxes, P0, yaw, sc["floor"], color=(0.62, 0.47, 0.30))
    geo = dict(boxes=boxes, start=P0.tolist(), yaw=round(yaw, 4))
    print(f"[g1_overhang] {sid}: slab (D1 door footprint), underside {sc['u']:.2f} m, {geo}", flush=True)
    return geo


import local_nav as LN
from door_methods import straight_act


class StraightCrouchPitchG1(LN.LocalNav):
    """ORACLE arm (not a method): lowest commanded body height + fixed torso pitch (training-lane choice)."""

    name = "crouchpitch_straight"
    label = "ORACLE: lowest body height + torso pitch (straight drive)"

    def reset(self, ep):
        super().reset(ep)
        self.sp_set = False
        rob = self.ctx["rob"]
        if hasattr(rob, "reset_setpoints"):
            rob.reset_setpoints()

    def act(self, obs):
        if not self.sp_set:
            lo = self.spec.extra["setpoint_ranges"]["body_height"][0]
            p = float(os.environ.get("G1_ORACLE_PITCH", "0.0"))
            obs["rob"].command(dict(body_height=lo, pitch=p, roll=0.0))
            self.sp_set = True
            self.info = f"oracle: body_height {lo:.2f} m, torso pitch {p:+.2f} rad"
        r = straight_act(self, obs)
        r["info"] = self.info
        return r
