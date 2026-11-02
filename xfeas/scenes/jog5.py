"""
gen5 'jog' family (school, real assets, in memory): a fenced lane with ONE jog, the gym course L1 fence recipe
(baselines/gym_course_spot_jog.py: post-and-rail fence of school slats, lane width 1.344 m) on a TRAINING
school lane (>= 3 m from every test scene; the gym course itself is never rebuilt or approached).

Layout in the lane frame (s along the lane, l lateral, left +):
   A: (0, 0) -> (s_bend, 0)                       heading 0
   J: -> (s_bend + Lj cos th, side Lj sin th)     heading side * th       (th = 45 or 90 deg)
   C: -> (L - 0.15, side Lj sin th)               heading 0
Fences on both sides of every segment, offset LANE_HALF + rail; each side is trimmed / extended to the intersection of the
neighbouring offset lines (mitre), the first post of a segment is dropped where it coincides with the previous segment's
last post, spans shorter than one legal rail bay (0.85 x slat) are left open. Verified: every post stands on the floor
(PhysX ray), the lane centre line is free (no part within 0.45 m of the path at 0.15-1.4 m height), no primitives.
Import after SimulationApp exists.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
import seq_course as SQ
import gensim as GS

JOG_ROOT = "/World/Climb/JogCourse"  # under gensim.OURS -> hazard for contacts / ray skipping / audits
LANE_HALF = SQ.LANE_HALF


def jog_polyline(L, s_bend, th_deg, Lj, side):
    th = math.radians(th_deg) * side
    p0 = (0.0, 0.0)
    p1 = (s_bend, 0.0)
    p2 = (s_bend + Lj * math.cos(th), Lj * math.sin(th))
    p3 = (L - 0.15, p2[1])
    return [p0, p1, p2, p3]


class JogCourse(SQ.SeqCourse):
    def __init__(self, stage):
        SQ.SeqCourse.__init__(self, stage, root=JOG_ROOT)

    def fence_side(self, F0, h, a, b, side, skip_first, rails=(0.30, 0.58, 0.86)):
        """one side of a fence along the segment frame (origin F0 (lane s, l), heading h rel. lane), from a to b"""
        SLs, SHs, STs = self.dims["slat"]
        if b - a < 0.85 * SLs:
            return None
        nb = max(1, int(round((b - a) / SLs)))
        while (b - a) / nb > SLs * 1.15:
            nb += 1
        while (b - a) / nb < SLs * 0.85 and nb > 1:
            nb -= 1
        bay = (b - a) / nb
        rsc = bay / SLs
        if not 0.85 <= rsc <= 1.15:
            return None
        rail_t = STs * rsc
        l_post = LANE_HALF + rail_t + 0.5 * STs
        dl = np.array([math.cos(h), math.sin(h)])
        nl = np.array([-dl[1], dl[0]])
        W = lambda s, l, z: self.W(F0[0] + dl[0] * s + nl[0] * l, F0[1] + dl[1] * s + nl[1] * l, z)
        yaw_seg = self.yaw + h
        sy = self.lib.size("slat")
        yaw_al = yaw_seg + (0.0 if sy[0] >= sy[1] else math.pi / 2)
        n_world = self.d * nl[0] + self.n * nl[1]
        posts = []
        for i in range(nb + 1):
            if i == 0 and skip_first:
                continue
            w = W(a + i * bay, side * l_post, 0.0)
            posts.append(
                self.put(
                    f"jog_post{'RL'[side > 0]}",
                    "slat",
                    1.0,
                    yaw_al,
                    pitch=math.pi / 2,
                    axis=tuple(n_world),
                    center=w[:2],
                    z_bottom=w[2],
                    contacts=[("down", "flat")],
                )
            )
        for i in range(nb):
            for hz in rails:
                w = W(a + (i + 0.5) * bay, side * (LANE_HALF + 0.5 * rail_t), hz)
                self.put(
                    f"jog_rail{'RL'[side > 0]}",
                    "slat",
                    rsc,
                    yaw_al,
                    center=w[:2],
                    z_bottom=w[2] - 0.5 * SHs * rsc,
                    contacts=[("lat", side)],
                )
        return dict(a=round(a, 3), b=round(b, 3), bays=nb, rail_scale=round(rsc, 3), posts=posts)


class JogScene:
    def __init__(self, sim):
        self.sim = sim
        sim.world.stop()
        self.C = JogCourse(sim.stage)
        self.built = False

    def clear(self):
        if self.built:
            self.sim.world.stop()
            self.C.reset()
            self.built = False

    def build(self, sit):
        G = self.sim
        J = sit["jog"]
        lane = sit["lane"]
        P0, yaw, d, n, fz = G.lane_frame(lane)
        L = float(lane["len"])
        th, Lj, side = float(J["theta_deg"]), float(J["Lj"]), float(J["side"])
        poly = jog_polyline(L, float(J["s_bend"]), th, Lj, side)
        off = abs(poly[2][1])
        ss = np.arange(0.4, L - 0.2, 0.4)
        room = G.free_lat(P0, d, n, fz, tuple(ss), zs=(0.15, 0.6, 1.0, 1.4))
        need_side = off + LANE_HALF + 0.12
        need_other = LANE_HALF + 0.12
        ok = (
            (room[0] >= need_side and room[1] >= need_other)
            if side > 0
            else (room[1] >= need_side and room[0] >= need_other)
        )
        if not ok:
            raise GS.Reject(
                f"jog_no_room {tuple(round(r, 2) for r in room)} need {need_side:.2f}/{need_other:.2f} side {side:+.0f}"
            )
        G.world.stop()
        G.OBJ.park_all()
        G.B.park()
        if G.SUP is not None:
            G.SUP.park()
        C = self.C
        C.reset()
        C.frame(P0, yaw, fz)
        self.built = True
        SLs, SHs, STs = C.dims["slat"]
        l_post = LANE_HALF + STs + 0.5 * STs
        segs = []
        for i in range(3):
            a_, b_ = np.array(poly[i]), np.array(poly[i + 1])
            h = math.atan2(b_[1] - a_[1], b_[0] - a_[0])
            segs.append((a_, h, float(np.linalg.norm(b_ - a_))))
        fence = []
        for i, (A, h, Ls) in enumerate(segs):
            for sd in (-1, 1):
                a0, b0 = (0.3 if i == 0 else 0.0), Ls
                # mitre trims: turn angle at the start / end bend, inner side shortens, outer side extends
                if i > 0:
                    phi = segs[i][1] - segs[i - 1][1]
                    a0 += sd * l_post * math.tan(phi / 2.0)
                if i < 2:
                    phi = segs[i + 1][1] - segs[i][1]
                    b0 -= sd * l_post * math.tan(phi / 2.0)
                b0 = min(b0, Ls - (0.0 if i < 2 else 0.0))
                r = C.fence_side(A, h, a0, b0, sd, skip_first=i > 0)
                fence.append(
                    dict(
                        seg=i,
                        side=sd,
                        span=[round(a0, 3), round(b0, 3)],
                        built=r is not None,
                        bays=None if r is None else r["bays"],
                        rail_scale=None if r is None else r["rail_scale"],
                    )
                )
        G._restart()
        # ---- verification: posts on the floor, centre line free, no primitives
        bad = []
        for p in C.parts:
            if "post" not in p["role"]:
                continue
            xy = p["corners"][:, :2].mean(0)
            zb = float(p["corners"][:, 2].min())
            dist, who, _ = G.ray(np.r_[xy, zb + 0.02], (0, 0, -1), 0.3)
            if dist is None or who.startswith(SQ.SEQ_ROOT) or who.startswith(JOG_ROOT) or abs(dist - 0.02) > 0.02:
                bad.append(dict(role=p["role"], dist=None if dist is None else round(dist, 3)))
        if bad:
            raise GS.Reject(f"jog_post_unsupported {bad[:3]}")
        Pw = [P0 + d * s + n * l for (s, l) in poly]
        import path_follower as ST

        pf = ST.PathFollower(Pw)
        clear_fail = []
        for sarc in np.arange(0.4, pf.total - 0.2, 0.1):
            q = pf.point(sarc)
            for z in (0.15, 0.6, 1.0, 1.4):
                for ang in np.linspace(0, 2 * math.pi, 12, endpoint=False):
                    v = (math.cos(ang), math.sin(ang), 0.0)
                    dist, who, _ = G.ray((q[0], q[1], fz + z), v, 0.45)
                    if dist is not None and who.startswith(JOG_ROOT):
                        clear_fail.append((round(float(sarc), 2), z, round(dist, 3)))
                        break
        if clear_fail:
            raise GS.Reject(f"jog_lane_blocked {clear_fail[:3]}")
        import real_assets as RA

        npc = RA.audit_no_primitives(G.stage, (JOG_ROOT,))
        if not npc["ok"]:
            raise GS.Reject(f"jog_primitive {npc.get('bad', [])[:2]}")
        s_b1 = float(J["s_bend"])
        arc_b1 = s_b1
        arc_b2 = s_b1 + Lj
        arc_end = pf.total
        geo = dict(
            P0=P0,
            yaw=yaw,
            d=d,
            n=n,
            fz=fz,
            s_front=round(s_b1, 3),
            s_end=round(s_b1 + Lj * math.cos(math.radians(th)), 3),
            s_goal=round(L - 0.4, 3),
            z_top=None,
            u_track=False,
            family="jog",
            family_oracle="null",
            on_top=False,
            value=th,
            path_lane=[[round(s, 4), round(l, 4)] for s, l in poly],
            path_world=[[float(p[0]), float(p[1])] for p in Pw],
            hazards=[dict(kind=f"bend{int(th)}", arc0=round(arc_b1, 3), arc1=round(arc_b2, 3))],
            arc_goal=round(arc_end - 0.25, 3),
            aim=None,
        )
        lc = G.lane_check(P0, d, n, fz, min(L, s_b1 + 0.2))
        geo["checks"] = dict(
            fence=fence,
            room=room,
            lane=lc,
            no_primitive=dict(ok=npc["ok"], n_gprims=npc.get("n_gprims")),
            n_parts=len(C.parts),
        )
        geo["hazard_band"] = (s_b1 - 1.0, s_b1 + Lj + 0.3)
        return geo
