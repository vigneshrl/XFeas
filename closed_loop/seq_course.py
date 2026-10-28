"""
gen_v4 'seq' family: COURSE-LIKE fenced-lane sequences built from REAL school assets, IN MEMORY (scenes/assets/school.usd is never
written), on TRAINING school lanes (>= 3 m from every test scene and from the gym box; the gym course is a TEST scene and
is never rebuilt or approached). Import after SimulationApp exists.

Assets (the gym course's own recipe, real_assets.RealLib copies: exact sub-prim, own material, exact colliders, uniform
scale inside 0.85-1.15):
  board = BP_school_desk_26/SM_school_desk_2_2 (desk top 1.20 x 0.65 x 0.026)   slat = .../SM_school_desk_2_3 (apron board)
  plank = SM_literature_props_1 (3.00 x 0.50 x 0.04)   door_leaf = school door leaf (2.58 x 0.73 x 0.14)
  desk  = SM_school_desk_2 (1.20 x 0.65 x 0.757)

Elements (lane frame: s along the lane, l lateral, z above the lane floor; nothing floats):
  fence      post-and-rail (posts = slats on end, 3 rails = slats on edge on the posts' lane faces), lane width 1.344 m,
             interrupted where a beam stands (the beam's own supports bound the lane there)
  beam       G1 band: 2-4 planks laid ACROSS the lane, each on two slat pillars standing on the floor at the lane edges
             (underside u = pillar height, 0.96-1.30 m)
  doorbeam   low band (Spot crouch / G1 too-low): 1-2 door leaves laid flat across the lane on two school desks standing
             outside the lane (underside = desk height, 0.64-0.87 m)
  funnel     V-funnel: two desk tops upright (1.38 m), angled from the rails to a throat of width g, each fixed to two
             slat posts behind it (gym course v2/v3 funnel geometry)
  hump       pitch hump: a ridge slat on edge across the lane + two desk tops leaning on it (gym course hump on the floor)
  deck       flat deck: 2-5 layers of desk tops (step-up 0.057-0.14 m), entry ramp (a desk top resting on the deck's front
             edge), flat run of 2-3 boards, step down at its end
One situation = one FOCUS element (the decision is made before it); the other elements of the course are the context
(behind the spawn, or downstream beyond the goal). Verified with PhysX rays after the build (supports, underside, throat).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
for _p in (os.path.join(ROOT, "scenes"), os.path.join(ROOT, "closed_loop"), os.path.join(ROOT, "Scripts", "data_collector")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gym_course as GC
import gensim as GS

SEQ_ROOT = "/World/Climb/SeqCourse"        # under gensim.OURS -> hazard for contacts / ray skipping / audits
CAT = dict(GC.CATALOG)
FOCUS_ORACLE = dict(beam="overhang", doorbeam="overhang", funnel="gap", hump="step", deck="step")
LANE_HALF = 0.672                          # gym course: 0.5 * board length at scale 1.1 + 12 mm


class SeqCourse(GC.GymCourse):
    def __init__(self, stage, root=SEQ_ROOT):
        GC.GymCourse.__init__(self, stage, root=root)

    def frame(self, P0, yaw, fz):
        self.P0 = np.asarray(P0, float); self.yaw = float(yaw); self.fz = float(fz)
        self.d = np.array([math.cos(yaw), math.sin(yaw)]); self.n = np.array([-self.d[1], self.d[0]])

    def W(self, s, l, z=0.0):
        xy = self.P0 + self.d * s + self.n * l
        return np.array([xy[0], xy[1], self.fz + z])

    def reset(self):
        self.lib.park(); self.parts = []

    # ---------------------------------------------------------------- elements
    def _yaw_al(self):
        sy = self.lib.size("slat")
        return self.yaw + (0.0 if sy[0] >= sy[1] else math.pi / 2)

    def fence(self, segs, rails=(0.30, 0.58, 0.86)):
        SLs, SHs, STs = self.dims["slat"]
        yaw_al = self._yaw_al()
        out = []
        for (a, b) in segs:
            if b - a < 0.5:
                continue
            nb = max(1, int(round((b - a) / SLs)))
            while (b - a) / nb > SLs * 1.15:
                nb += 1
            while (b - a) / nb < SLs * 0.85 and nb > 1:
                nb -= 1
            bay = (b - a) / nb; rsc = bay / SLs
            if not 0.85 <= rsc <= 1.15:
                continue                                   # a stub shorter than one legal rail: leave it open
            rail_t = STs * rsc
            l_post = LANE_HALF + rail_t + 0.5 * STs
            for side in (-1, 1):
                for i in range(nb + 1):
                    w = self.W(a + i * bay, side * l_post, 0.0)
                    self.put(f"fence{'RL'[side > 0]}_post", "slat", 1.0, yaw_al, pitch=math.pi / 2, axis=tuple(self.n),
                             center=w[:2], z_bottom=w[2], contacts=[("down", "flat")])
                for i in range(nb):
                    for hz in rails:
                        w = self.W(a + (i + 0.5) * bay, side * (LANE_HALF + 0.5 * rail_t), hz)
                        self.put(f"fence{'RL'[side > 0]}_rail", "slat", rsc, yaw_al, center=w[:2], z_bottom=w[2] - 0.5 * SHs * rsc,
                                 contacts=[("lat", side)])
            out.append(dict(s=[round(a, 3), round(b, 3)], bays=nb, rail_scale=round(rsc, 3)))
        return out

    def beam(self, s0, u, n_planks, room=(9.9, 9.9)):
        SLs, SHs, STs = self.dims["slat"]
        psc = u / SLs
        assert 0.85 <= psc <= 1.15, psc
        Pl, Pw, Pt = self.lib.flat_dims("plank"); bsc = 0.85; pw = Pw * bsc; gap = 0.002
        D = n_planks * pw + (n_planks - 1) * gap
        pyaw = self.yaw + math.pi / 2 + self.lib.flat_yaw("plank")
        yaw_al = self._yaw_al()
        l_pil = LANE_HALF - 0.5 * STs * psc - 0.004 - 0.012
        # the plank may slide sideways on its two pillars (still resting on both, >= 0.25 m of plank beyond each pillar)
        # towards the side with more free room: room = (left, right) lateral free distance of the lane
        half = 0.5 * Pl * bsc
        shift = float(np.clip(0.5 * (room[0] - room[1]), -(half - l_pil - 0.25), half - l_pil - 0.25))
        pairs = []
        for k in range(n_planks):
            sk = s0 + k * (pw + gap) + 0.5 * pw
            tops = []; pil = []
            for side in (-1, 1):
                wp = self.W(sk, side * l_pil, 0.0)
                p = self.put("beam_pillar", "slat", psc, yaw_al, pitch=math.pi / 2, axis=tuple(self.n), center=wp[:2], z_bottom=wp[2],
                             contacts=[("down", "flat")])
                tops.append(float(p["corners"][:, 2].max())); pil.append(p)
            w = self.W(sk, shift, 0.0); zb = max(tops)
            pk = self.put("beam_plank", "plank", bsc, pyaw, center=w[:2], z_bottom=zb, orient="flat")
            pairs += [(pk, q) for q in pil]
        return dict(kind="beam", s_front=s0, s_end=s0 + D, u=u, depth=D, width=Pl * bsc, supports=pairs, shift=round(shift, 3),
                    need_lr=(half + shift + 0.05, half - shift + 0.05))

    def doorbeam(self, s0, desk_scale, n_doors):
        dsz = np.sort(self.lib.size("desk"))[::-1]          # (1.20, 0.65, 0.757)? height = the smallest vertical extent
        dL, dW = float(dsz[0]), float(dsz[1])
        Ls, Ws, Hs = (x * desk_scale for x in self.lib.size("desk"))
        Dl, Dw, Dt = self.lib.flat_dims("door_leaf")
        gap = 0.003
        D = n_doors * Dw + (n_doors - 1) * gap
        dyaw = self.yaw + math.pi / 2 + self.lib.flat_yaw("door_leaf")
        sz = self.lib.size("desk") * desk_scale
        long_ax = int(np.argmax(sz[:2]))
        desk_yaw = self.yaw + (0.0 if long_ax == 0 else math.pi / 2)        # desk long side ALONG the lane
        w_across = float(min(sz[:2]))
        l_desk = 0.65 + 0.5 * w_across                                     # inner face 0.65 m from the lane centre (RealLib L_IN)
        s_c = s0 + 0.5 * D
        desks, tops = [], []
        for side in (-1, 1):
            w = self.W(s_c, side * l_desk, 0.0)
            p = self.put("door_desk", "desk", desk_scale, desk_yaw, center=w[:2], z_bottom=w[2], contacts=[("down", "flat")])
            desks.append(p); tops.append(float(p["corners"][:, 2].max()))
        pairs = []
        for k in range(n_doors):
            sk = s0 + k * (Dw + gap) + 0.5 * Dw
            w = self.W(sk, 0.0, 0.0)
            pk = self.put("door_leaf", "door_leaf", 1.0, dyaw, center=w[:2], z_bottom=max(tops), orient="flat")
            pairs += [(pk, q) for q in desks]
        u = max(tops) - self.fz
        return dict(kind="doorbeam", s_front=s0, s_end=s0 + D, u=u, depth=D, width=Dl, supports=pairs,
                    half_room=max(0.5 * Dl, l_desk + 0.5 * w_across) + 0.05)

    def funnel(self, s_o, g, board_scale=1.15, post_scale=1.15, post_at=(0.35, 0.75)):
        import real_assets as RA
        BL1, BD1, BT1 = self.dims["board"]
        Hb, Wb, tb = BL1 * board_scale, BD1 * board_scale, BT1 * board_scale
        SLs, SHs, STs = self.dims["slat"]
        l_lim = LANE_HALF - 0.002
        th = math.asin((l_lim - 0.5 * g) / Wb)
        for _ in range(50):
            th = math.asin((l_lim - 0.5 * g - tb * math.cos(th)) / Wb)
        Ls = Wb * math.cos(th) + tb * math.sin(th)
        s_t = s_o + Wb * math.cos(th)
        wl = lambda s, l: np.r_[self.d * s + self.n * l]
        yaw_al = self._yaw_al()
        lp = lambda P: (P[:, :2] - self.P0) @ self.n
        sup = []
        for side in (-1, 1):
            v = np.array([-math.cos(th), side * math.sin(th)])
            nb = np.array([math.sin(th), side * math.cos(th)])
            Ft = np.array([s_t, side * 0.5 * g])
            Cc = Ft + v * (0.5 * Wb) + nb * (0.5 * tb)
            a_w, v_w = wl(*nb), wl(*v)
            byaw = math.atan2(a_w[1], a_w[0]) + self.lib.flat_yaw("board")
            wc = self.W(Cc[0], Cc[1], 0.0)
            part = self.put("funnel_wing", "board", board_scale, byaw, pitch=math.pi / 2, axis=(float(v_w[0]), float(v_w[1])),
                            center=wc[:2], z_bottom=wc[2], orient="flat", contacts=[("down", "flat")])
            Pw = RA.world_points(self.stage.GetPrimAtPath(part["path"]))
            delta = float((side * lp(Pw)).min()) - 0.5 * g
            dxy = -side * delta * self.n
            self.lib.shift([self.lib.placed[-1]], dxy)
            part["corners"] = part["corners"] + np.r_[dxy, 0.0]
            Ft = Ft + np.array([0.0, -side * delta])
            pyaw = math.atan2(v_w[1], v_w[0]) + (yaw_al - self.yaw)
            for fr in post_at:
                Q = Ft + v * (Wb * (1.0 - fr)) + nb * (tb + 0.5 * STs * post_scale + 0.0005)
                wq = self.W(Q[0], Q[1], 0.0)
                pp = self.put("funnel_post", "slat", post_scale, pyaw, pitch=math.pi / 2, axis=(float(a_w[0]), float(a_w[1])),
                              center=wq[:2], z_bottom=wq[2], contacts=[("down", "flat")])
                sup.append((part, pp, nb.copy(), Q.copy()))
        return dict(kind="funnel", s_front=s_o, s_end=s_o + Ls, s_throat=s_t, g=g, posts=sup, half_room=LANE_HALF + 0.1)

    def hump(self, s0, support_scale=0.85, hump_scale=1.10):
        SLs, SHs, STs = self.dims["slat"]
        Hs, Ts = SHs * support_scale, STs * support_scale
        PL, PD, PT = (x * hump_scale for x in self.dims["board"])
        apex = Hs + 0.003
        for _ in range(30):
            th = math.asin(apex / PD); apex = Hs + 0.5 * Ts * math.tan(th) + 0.0005
        th = math.asin(apex / PD)
        Hh = PD * math.cos(th) + PT * math.sin(th)
        c = s0 + Hh
        yaw_al = self._yaw_al()
        byaw = self.yaw + math.pi / 2 + self.lib.flat_yaw("board")
        C = self.W(c, 0.0, 0.0)
        ridge = self.put("hump_ridge", "slat", support_scale, yaw_al + math.pi / 2, center=C[:2], z_bottom=C[2], contacts=[("down", "flat")])

        def upper_anchor(P, signs, size):
            B = self.bottom_face(P, signs, size); return B[np.argsort(B[:, 2])[2:]].mean(0)
        u_sup = (0.25 * Ts) / (PD * math.cos(th)); v_sup = min(0.3, 0.4 * SLs * support_scale) / PL
        for side in (-1, 1):
            axis = tuple(self.n)
            for sg in (1, -1):
                P, signs, size = self._local("board", "flat", byaw, hump_scale, sg * th, axis)
                Bf = self.bottom_face(P, signs, size); hi = Bf[np.argsort(Bf[:, 2])[2:]]
                if ((hi.mean(0)[:2] - Bf.mean(0)[:2]) @ self.d) * (-side) > 0:
                    break
            self.put("hump_panel", "board", hump_scale, byaw, pitch=sg * th, axis=axis, anchor=upper_anchor, target=C + np.r_[0, 0, apex],
                     orient="flat", contacts=[("down", ("ridge", u_sup, v_sup))])
        top = apex + PT / math.cos(th)
        return dict(kind="hump", s_front=s0, s_end=c + Hh, c=c, deg=math.degrees(th), top=top, half_room=0.5 * PL + 0.05)

    def deck(self, s0, layers, n_boards, board_scale=1.10):
        BL, BD, BT = (x * board_scale for x in self.dims["board"])
        t = BT; gap = GC.GAP
        byaw = self.yaw + math.pi / 2 + self.lib.flat_yaw("board")
        head = (layers - 1) * t
        th_r = math.asin(head / BD); s_d = s0 + BD * math.cos(th_r)
        for jl in range(layers):
            s_k0 = s_d + (GC.HEAD_SEAT if jl == layers - 1 else 0.0)
            for ia in range(n_boards):
                sc = s_k0 + ia * (BD + gap) + BD / 2
                w = self.W(sc, 0.0, jl * t)
                self.put("deck", "board", board_scale, byaw, center=w[:2], z_bottom=w[2], orient="flat", contacts=[("down", "flat")])
        s_e = s_d + n_boards * (BD + gap) - gap
        ramp = None
        for sg in (1, -1):
            P, signs, size = self._local("board", "flat", byaw, board_scale, sg * th_r, tuple(self.n))
            B = self.bottom_face(P, signs, size)
            hi = B[np.argsort(B[:, 2])[2:]]
            if (hi.mean(0)[:2] - B.mean(0)[:2]) @ self.d > 0:
                ramp = sg * th_r; break

        def head_anchor(P, signs, size):
            B = self.bottom_face(P, signs, size); return B[np.argsort(B[:, 2])[2:]].mean(0)
        self.put("deck_ramp", "board", board_scale, byaw, pitch=ramp, axis=tuple(self.n), anchor=head_anchor,
                 target=self.W(s_d + 0.5 * GC.HEAD_SEAT, 0.0, head), orient="flat", contacts=[("down", "edges")])
        return dict(kind="deck", s_front=s0, s_end=s_e, s_deck=s_d, top=layers * t, ramp_deg=math.degrees(th_r), half_room=0.5 * BL + 0.05)


class SeqScene:
    """the collector's handle: build(sit) -> geo (gensim build contract) or raises gensim.Reject"""

    def __init__(self, sim):
        self.sim = sim
        sim.world.stop()
        self.C = SeqCourse(sim.stage)
        self.built = False

    def clear(self):
        if self.built:
            self.sim.world.stop()
            self.C.reset()
            self.built = False

    # ---------------------------------------------------------------- PhysX checks
    def _ray_skip(self, o, v, L, skip):
        G = self.sim; acc = 0.0; v = np.asarray(v, float)
        for _ in range(10):
            dist, who, pos = G.ray(np.asarray(o, float) + v * acc, v, L - acc)
            if dist is None:
                return None, "", None
            if who.startswith(skip) or who.startswith(G.root):
                acc += dist + 1e-3; continue
            return acc + dist, who, pos
        return None, "", None

    def build(self, sit):
        G = self.sim
        Q = sit["seq"]; lane = sit["lane"]
        P0, yaw, d, n, fz = G.lane_frame(lane)
        L = float(lane["len"])
        # lateral room BEFORE building (scene only): fence everywhere, beams / desks need more
        ss = np.arange(0.4, L - 0.2, 0.4)
        room = G.free_lat(P0, d, n, fz, tuple(ss), zs=(0.15, 0.6, 1.0, 1.4))
        G.world.stop()
        G.OBJ.park_all(); G.B.park()
        if G.SUP is not None:
            G.SUP.park()
        C = self.C; C.reset(); C.frame(P0, yaw, fz)
        self.built = True                                  # from here on, parts may be placed: clear() must park them
        els = []; s = float(Q["s_first"])
        for i, e in enumerate(Q["elements"]):
            k = e["kind"]
            if k == "beam":
                r = C.beam(s, float(e["u"]), int(e["n"]), room=room)
            elif k == "doorbeam":
                r = C.doorbeam(s, float(e["desk_scale"]), int(e["n"]))
            elif k == "funnel":
                r = C.funnel(s, float(e["g"]))
            elif k == "hump":
                r = C.hump(s)
            elif k == "deck":
                r = C.deck(s, int(e["layers"]), int(e["n"]))
            else:
                raise GS.Reject(f"seq_unknown_element {k}")
            r["i"] = i; els.append(r)
            s = r["s_end"] + float(e.get("gap_after", 0.0))
        if s > L + 0.05:
            raise GS.Reject(f"seq_too_long {s:.2f} > {L:.2f}")
        # fence over the course, interrupted at beams (their pillars / desks bound the lane there)
        cut = [(e["s_front"] - 0.15, e["s_end"] + 0.15) for e in els if e["kind"] in ("beam", "doorbeam")]
        a0, b0 = 0.3, L - 0.15; segs = []; cur = a0
        for (x0, x1) in sorted(cut):
            if x0 > cur:
                segs.append((cur, min(x0, b0)))
            cur = max(cur, x1)
        if cur < b0:
            segs.append((cur, b0))
        fence = C.fence(segs)
        need_l = max([LANE_HALF + 0.12] + [e.get("need_lr", (e.get("half_room", 0), 0))[0] for e in els])
        need_r = max([LANE_HALF + 0.12] + [e.get("need_lr", (0, e.get("half_room", 0)))[1] for e in els])
        if room[0] < need_l or room[1] < need_r:
            raise GS.Reject(f"seq_no_room {room} need ({need_l:.2f}, {need_r:.2f})")
        G._restart()
        # ---- verification (PhysX rays): supports, underside, throat, lane clear of the scene
        chk = dict(fence=fence, room=room, elements=[], support_fail=[])
        for e in els:
            if "supports" in e:
                for (pc, sp) in e["supports"]:
                    Pp, Ps = pc["corners"], sp["corners"]
                    zb = float(Pp[:, 2].min())
                    xy = Ps[:, :2].mean(0)
                    dist, who, pos = self._ray_skip(np.r_[xy, zb + 0.02], (0, 0, -1), 0.2, pc["path"])
                    ok = dist is not None and who.startswith(sp["path"]) and abs(dist - 0.02) < 0.006
                    if not ok:
                        chk["support_fail"].append(dict(piece=pc["role"], support=sp["role"], dist=None if dist is None else round(dist - 0.02, 4),
                                                        who=who[-50:]))
            if e["kind"] in ("beam", "doorbeam"):
                U = []
                for s_ in np.arange(e["s_front"] + 0.03, e["s_end"] - 0.02, 0.03):
                    for l_ in (-0.4, -0.2, 0.0, 0.2, 0.4):
                        dist, who, pos = G.ray(C.W(s_, l_, 0.05), (0, 0, 1), 2.5)
                        if dist is not None and who.startswith(SEQ_ROOT):
                            U.append(dist + 0.05)
                if len(U) < 10:
                    raise GS.Reject(f"seq_underside_unseen {e['kind']} n={len(U)}")
                e["u_meas"] = round(float(np.min(U)), 3)
                if abs(e["u_meas"] - e["u"]) > 0.03:
                    raise GS.Reject(f"seq_underside_mismatch {e['u_meas']} vs {e['u']:.3f}")
            if e["kind"] == "funnel":
                W_ = []
                for s_ in np.arange(e["s_front"], e["s_end"] + 0.01, 0.01):
                    for hz in (0.1, 0.4, 0.8, 1.2):
                        o = C.W(s_, 0.0, hz)
                        a_, wa, _ = G.ray(o, np.r_[n, 0.0], 1.5); b_, wb, _ = G.ray(o, np.r_[-n, 0.0], 1.5)
                        if a_ is not None and b_ is not None and wa.startswith(SEQ_ROOT) and wb.startswith(SEQ_ROOT):
                            W_.append(a_ + b_)
                if len(W_) < 10:
                    raise GS.Reject("seq_throat_unseen")
                e["g_meas"] = round(float(np.min(W_)), 3)
                if abs(e["g_meas"] - e["g"]) > 0.02:
                    raise GS.Reject(f"seq_throat_mismatch {e['g_meas']} vs {e['g']:.3f}")
            chk["elements"].append({k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items() if k not in ("supports", "posts")})
        if chk["support_fail"]:
            raise GS.Reject(f"seq_unsupported {chk['support_fail'][:3]}")
        import real_assets as RA
        npc = RA.audit_no_primitives(G.stage, (SEQ_ROOT,))
        chk["no_primitive"] = dict(ok=npc["ok"], n_gprims=npc.get("n_gprims"), bad=npc.get("bad", [])[:3])
        if not npc["ok"]:
            raise GS.Reject(f"seq_primitive {npc.get('bad', [])[:2]}")
        # ---- the focus element -> geo
        f = els[int(Q["focus"])]
        nxt = els[int(Q["focus"]) + 1]["s_front"] if int(Q["focus"]) + 1 < len(els) else L
        s_goal = min(f["s_end"] + 0.6, nxt - 0.3, L - 0.2)
        if s_goal < f["s_end"] + 0.25:
            raise GS.Reject(f"seq_no_goal_room {s_goal:.2f} end {f['s_end']:.2f}")
        prev_end = els[int(Q["focus"]) - 1]["s_end"] if int(Q["focus"]) > 0 else 0.0
        geo = dict(P0=P0, yaw=yaw, d=d, n=n, fz=fz, s_front=round(f["s_front"], 3), s_end=round(f["s_end"], 3), s_goal=round(s_goal, 3),
                   z_top=None, u_track=f["kind"] in ("beam", "doorbeam"), family="seq", family_oracle=FOCUS_ORACLE[f["kind"]],
                   seq_prev_end=round(prev_end, 3), on_top=False)
        if f["kind"] in ("beam", "doorbeam"):
            geo["value"] = f["u_meas"]; geo["aim"] = (0.5 * (f["s_front"] + f["s_end"]), 0.0, f["u_meas"] + 0.02)
            geo["u_track"] = True
            geo["S"] = np.arange(f["s_front"], f["s_end"] + 1e-6, 0.05); geo["U"] = np.full(len(geo["S"]), f["u_meas"])
        elif f["kind"] == "funnel":
            geo["value"] = f["g_meas"]; geo["aim"] = (f["s_throat"], 0.0, 0.6)
        elif f["kind"] == "hump":
            geo["value"] = round(f["top"], 3); geo["aim"] = (f["c"], 0.0, 0.5 * f["top"])
        else:
            geo["value"] = round(f["top"], 3); geo["aim"] = (f["s_deck"], 0.0, 0.5 * f["top"])
        lc = G.lane_check(P0, d, n, fz, min(s_goal + 0.3, L))
        chk["lane"] = lc
        if lc["n_low_head"] > 0 and f["kind"] not in ("beam", "doorbeam"):
            low = [x for x in lc["low_head"] if not (any(e["s_front"] - 0.05 <= x[0] <= e["s_end"] + 0.05 for e in els if e["kind"] in ("beam", "doorbeam")))]
            if low:
                raise GS.Reject(f"seq_low_headroom {low[:2]}")
        geo["checks"] = chk
        geo["hazard_band"] = (f["s_front"] - 1.0, f["s_end"] + 0.3)
        return geo
