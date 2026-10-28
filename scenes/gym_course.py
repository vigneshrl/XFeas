#!/usr/bin/env python
"""
GYM OBSTACLE COURSE -- mobility only -- in the school gym (scenes/assets/school.usd, built IN MEMORY, the file is never written).
Modelled on Goal_scenes/obstacle_course.png, keeping ONLY its mobility hurdles, in order along one straight lane:

  1. FENCED LANE     post-and-rail timber fence on both sides (posts = school desk apron boards standing on end,
                     rails = the same boards on edge, three rails per bay), from the ramp foot to the stairs.
  2. ENTRY RAMP      one school desk top tilted ~11-12 deg, foot on the gym floor, head resting on the deck's front edge.
     + DECK          the raised course floor: a laminated stack of school desk tops (light plywood-look wood).
  3. PITCH/ROLL      "crossing ramps": humps of two desk tops leaning on a ridge board (an apron board on edge), ridges
     RAMPS           at +-45 deg to the lane and alternating (chevron), panels ~14 deg -> ~10 deg pitch AND ~10 deg roll.
  4. STAIRS          desk-top terrace (riser = 6 boards = ~0.17 m, tread ~0.30 m) up to a RAISED PLATFORM, with
     + PLATFORM      sloped handrails on both sides and a rail enclosure on the platform's sides and far end.
EXCLUDED from the reference (manipulation): the gantry with the hook, the door with a handle, the wall valves/handles,
the sink/counter with objects and bins, the swing gate, and anything to pick, push or turn. The reference has no open
doorway without a door, so no doorway passage is built.

Every part is a duplicate of a real school asset (hazard_copy.CopyLib recipe via real_assets.RealLib: exact sub-prim,
full composed pose, instancing broken, exact triangle colliders, the asset's own material MI_school_desk_2):
  board  BP_school_desk_26/SM_school_desk_2_2   desk top   1.200 x 0.650 x 0.0257 m (OBB at scale 1)
  slat   BP_school_desk_26/SM_school_desk_2_3   apron board 1.129 x 0.187 x 0.026 m
Uniform scales only, inside RealLib's 0.85-1.15 band. Nothing floats: boards lie on boards / the floor, the ramp rests on
the floor and on the deck's front edge, hump panels rest on the deck and on their ridge board, posts stand on the floor /
treads / platform, rails are fixed to the faces of their posts (contacts verified with PhysX rays, verify()).

  spec = make_spec(seed)                       # pure python, JSON-able; seed 0 = the canonical course
  python scenes/gym_course.py --seed 0 --out spec.json            # spec only (no Isaac)
  # inside a SimulationApp with the school open:
  off = prepare_gym(stage); GC = GymCourse(stage); info = GC.build(spec); world.reset(); meas = GC.verify(spec, pxq)
Robots: spec["start"] (world xy, yaw) and spec["goal"] (on the platform) + spec["path"] (3-D polyline of the walking
surface) are written for later episode runs.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import argparse, json, math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
FLOOR_Z = 0.008976                      # gym floor (SM_floor_element_1), ray grid data/gym_course/survey/floor_grid.npz
GYM = dict(x=(-24.445, -6.946), y=(-2.62, 7.849))     # inner wall faces, horizontal rays at 0.5 m (gym_survey.json)
G = "/World/School/Geom/"
ROOT_PRIM = "/World/GymCourse"
# loose floor props of the gym, deactivated IN MEMORY (as scripts/demo/two_robot_gym2.py) + the drill press in the middle
LOOSE = ["SM_gym_props_12", "SM_gym_props_13", "SM_gym_props_14", "SM_gym_props_15", "SM_gym_props_16", "SM_gym_props_17",
         "SM_gym_props_23", "SM_school_lighting_6", "SM_school_lighting_7", "SM_school_lighting_8", "SM_school_lighting_9",
         "Oil_surface_01", "Oil_surface_02", "SM_school_desk_1_02", "InstancedSM_school_door_6_2",
         "BP_school_window_11/SM_school_window_1_3/InstancedSM_school_window_1_3",
         "BP_drilling_machine_1/SM_drilling_machine_02"]
BAKED_IN_GYM = ("G1", "G2")             # baked climb scenes on the gym floor (ClimbScene_G1 mats, ClimbScene_G2 planks)
CATALOG = dict(board=dict(src="BP_school_desk_26/SM_school_desk_2_2"),
               slat=dict(src="BP_school_desk_26/SM_school_desk_2_3"))
GAP = 0.002                             # gap between neighbouring boards of one layer
HEAD_SEAT = 0.008                       # the ramp head bears 4 mm deep on the deck's front edge (top layer set back 8 mm)


def prepare_gym(stage):
    """clean school + clear the gym floor in memory (scenes/assets/school.usd is never written). -> deactivated paths"""
    sys.path.insert(0, os.path.join(ROOT, "closed_loop"))
    import scene_clean, baked_scenes as BK
    for p in ("/World/Spot", "/World/Surfaces"):
        if stage.GetPrimAtPath(p).IsValid():
            stage.GetPrimAtPath(p).SetActive(False)
    scene_clean.apply(stage)
    off = []
    for n in LOOSE:
        pr = stage.GetPrimAtPath(G + n)
        if pr.IsValid() and pr.IsActive():
            pr.SetActive(False); off.append(G + n)
    for sid in BAKED_IN_GYM:
        for r in BK.baked(stage).get(sid, []):
            pr = stage.GetPrimAtPath(r)
            if pr.IsValid() and pr.IsActive():
                pr.SetActive(False); off.append(r)
    return off


# ------------------------------------------------------------------ spec (pure python)
def make_spec(seed=0):
    """seed 0 = canonical course; other seeds jitter the layout inside the same mobility ranges"""
    rng = np.random.default_rng(seed)
    j = (lambda a, b, c: c) if seed == 0 else (lambda a, b, c: float(rng.uniform(a, b)))
    n_el = 4 if seed == 0 else int(rng.integers(3, 6))
    first = 0 if seed == 0 else int(rng.integers(0, 2))
    kinds = [("hump", "valley")[(k + first) % 2] for k in range(n_el)]
    spec = dict(
        seed=int(seed), version=1,
        board_scale=1.10,                 # deck / ramp / stairs desk tops: 1.32 across x 0.715 along x 0.0283 thick
        riser_layers=6,                   # deck height = one riser = 6 boards
        flat_a=round(j(1.6, 2.6, 2.0), 3),          # flat deck between the ramp head and the first hump
        crossing=[dict(type=t_, support_scale=0.85) for t_ in kinds],   # pitch humps and cross-slope valleys
        hump_scale=1.10,                  # hump panels: desk tops 1.32 (across the lane) x 0.715 (fall line)
        valley_scale=1.00,                # valley panels: desk tops 1.20 (along the lane) x 0.65 (fall line, across)
        valley_inset=0.026,               # valley support slat fully under the panel's outer edge
        cross_gap=round(j(0.10, 0.30, 0.15), 3),    # flat deck between consecutive crossing elements
        flat_b=round(j(1.6, 2.6, 2.0), 3),          # flat deck between the last hump and the stairs
        n_steps=3 if seed == 0 else int(rng.integers(3, 5)),   # treads below the platform (risers = n_steps + 1)
        tread=round(j(0.28, 0.32, 0.30), 3),
        platform_len=round(j(1.9, 2.3, 2.0), 3),
        fence=dict(post_scale=1.0, rail_scale_max=1.15, rails_above_deck=[0.30, 0.58, 0.86], bay=1.0),
        handrail=dict(post_scale=0.85, rails_above=[0.45, 0.85]),
        lane_y=round(j(2.2, 3.0, 2.6), 3), yaw=0.0)
    return spec


def layout(spec, lib_dims):
    """lane-frame layout (s along the lane from the ramp foot, l lateral, z above the gym floor) from the spec and the
    measured asset dimensions lib_dims = dict(board=(L, D, t) at scale 1, slat=(L, H, t) at scale 1)"""
    bs = spec["board_scale"]
    BL, BD, BT = (x * bs for x in lib_dims["board"])
    t = BT
    deck_top = spec["riser_layers"] * t
    head = (spec["riser_layers"] - 1) * t                     # the ramp head rests on the top of layer riser_layers-2
    th_r = math.asin(head / BD)
    s_d = BD * math.cos(th_r)                                 # deck front edge (= ramp head, horizontally)
    # crossing elements on the deck (all panels = desk tops, supports = apron slats on edge)
    SLs, SHs, STs = lib_dims["slat"]
    els, s = [], s_d + spec["flat_a"]
    for k, el in enumerate(spec["crossing"]):
        ss_ = el["support_scale"]; Hs, Ts = SHs * ss_, STs * ss_
        if el["type"] == "hump":
            PL, PD, PT = (x * spec["hump_scale"] for x in lib_dims["board"])
            apex = Hs + 0.003
            for _ in range(30):                                   # panels rest on the ridge slat's top edges
                th = math.asin(apex / PD); apex = Hs + 0.5 * Ts * math.tan(th) + 0.0005
            th = math.asin(apex / PD)
            Hh = PD * math.cos(th) + PT * math.sin(th)
            c = s + Hh
            els.append(dict(type="hump", c=c, extent=(s, c + Hh), deg=math.degrees(th), apex=apex, panel=(PL, PD, PT),
                            support=(SLs * ss_, Hs, Ts), lat=0.5 * PL))
            s = c + Hh
        else:
            PL, PD, PT = (x * spec["valley_scale"] for x in lib_dims["board"])
            ins = spec["valley_inset"]
            lo_, hi_ = math.radians(5), math.radians(30)            # PD sin(th) - ins tan(th) = Hs  (inner slat top edge)
            for _ in range(60):
                th = 0.5 * (lo_ + hi_)
                if PD * math.sin(th) - ins * math.tan(th) > Hs:
                    hi_ = th
                else:
                    lo_ = th
            l0 = 0.002 + PT * math.sin(th)                        # inner TOP edges 2 mm off the centre line (no overlap)
            l_out = l0 + PD * math.cos(th)
            c = s + 0.5 * PL
            els.append(dict(type="valley", c=c, extent=(s, s + PL), deg=math.degrees(th), panel=(PL, PD, PT),
                            support=(SLs * ss_, Hs, Ts), l0=l0, l_out=l_out, l_sup=l_out - ins,
                            lat=l_out + PT * math.sin(th)))
            s = s + PL
        s += spec["cross_gap"] if k + 1 < len(spec["crossing"]) else 0.0
    s_st = s + spec["flat_b"]
    starts = [s_d, s_st] + [s_st + k * spec["tread"] for k in range(1, spec["n_steps"] + 1)]
    s_p0 = starts[-1]
    s_end = s_p0 + spec["platform_len"]
    return dict(BL=BL, BD=BD, t=t, deck_top=deck_top, head=head, ramp_deg=math.degrees(th_r), s_d=s_d,
                crossing=els,
                s_st=s_st, step_starts=starts, s_p0=s_p0, s_end_req=s_end)


def _poly_dist(A, B):
    """smallest distance between two convex quads given as 4 corners each (any order); 0 if they overlap"""
    def hull(P):
        c = P.mean(0); return P[np.argsort(np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0]))]
    A, B = hull(A), hull(B)

    def sep(P, Q):
        best = -1e9
        for i in range(len(P)):
            e = P[(i + 1) % len(P)] - P[i]; nrm = np.array([e[1], -e[0]]) / np.linalg.norm(e)
            if np.dot(P.mean(0) - P[i], nrm) > 0:
                nrm = -nrm
            best = max(best, float(np.min((Q - P[i]) @ nrm)))
        return best
    return max(0.0, sep(A, B), sep(B, A))


# ------------------------------------------------------------------ builder (needs a stage)
class GymCourse:
    def __init__(self, stage, root=ROOT_PRIM):
        sys.path.insert(0, os.path.join(ROOT, "closed_loop"))
        import real_assets as RA
        self.RA, self.stage = RA, stage
        self.lib = RA.RealLib(stage, root=root, catalog=CATALOG)
        self.root = self.lib.root
        self.parts = []
        sb, ss = self.lib.size("board"), self.lib.size("slat")
        self.dims = dict(board=(float(max(sb[:2])), float(min(sb[:2])), float(sb[2])),
                         slat=(float(max(ss[:2])), float(ss[2]), float(min(ss[:2]))))

    # lane frame
    def W(self, s, l, z=0.0):
        P0, d, n = self.P0, self.d, self.n
        xy = P0 + d * s + n * l
        return np.array([xy[0], xy[1], FLOOR_Z + z])

    def _local(self, kind, orient, yaw, scale, pitch, axis):
        R, P = self.lib.orient_matrix(kind, orient, yaw, scale, pitch, axis)
        size = self.lib.size(kind)
        signs = np.array([(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        return P, signs, size

    def put(self, role, kind, scale, yaw, pitch=0.0, axis=(0.0, 1.0), anchor=None, target=None, center=None, z_bottom=None,
            orient="as_is", contacts=None):
        """place one copy. Either center=(x, y) + z_bottom, or anchor(P, signs, size)->local point + its world target."""
        P, signs, size = self._local(kind, orient, yaw, scale, pitch, axis)
        if anchor is not None:
            a = anchor(P, signs, size)
            c = np.asarray(target, float) - a
            center, z_bottom = c[:2], c[2] + P[:, 2].min()
        rec = self.lib.place(kind, float(center[0]), float(center[1]), float(z_bottom), orient, yaw, scale, pitch, axis)
        zc = z_bottom - P[:, 2].min()
        Pw = P + np.array([center[0], center[1], zc])
        part = dict(role=role, kind=kind, scale=round(float(scale), 4), path=str(rec["copy"]["prim"].GetPath()), corners=Pw,
                    signs=signs, size=size, contacts=contacts or [])
        self.parts.append(part)
        return part

    # ---- helpers for the corner sets
    @staticmethod
    def thin_axis(size):
        return int(np.argmin(size))

    @staticmethod
    def face(part_or_P, signs, axis_i, sign):
        P = part_or_P["corners"] if isinstance(part_or_P, dict) else part_or_P
        return P[signs[:, axis_i] == sign]

    def bottom_face(self, P, signs, size):
        """the 4 corners of the face whose outward normal points most downwards"""
        best, c0 = None, P.mean(0)
        for i in range(3):
            for sg in (-1, 1):
                F = P[signs[:, i] == sg]
                v = F.mean(0) - c0; v = v / max(1e-12, np.linalg.norm(v))       # outward normal of the face
                if best is None or v[2] < best[0]:
                    best = (v[2], F)
        return best[1]

    # ------------------------------------------------------------ build
    def build(self, spec):
        s_ = spec
        yaw = s_["yaw"]
        self.d = np.array([math.cos(yaw), math.sin(yaw)]); self.n = np.array([-self.d[1], self.d[0]])
        L = layout(s_, self.dims)
        total = L["s_end_req"] + 0.8                                   # + the enclosure / terrace overrun
        x_mid = 0.5 * (GYM["x"][0] + GYM["x"][1])
        self.P0 = np.array([x_mid - 0.5 * total, s_["lane_y"]]) if abs(yaw) < 1e-9 else np.array(s_.get("P0"))
        self.L = L
        bs, t = s_["board_scale"], L["t"]
        BL, BD = L["BL"], L["BD"]
        byaw = yaw + math.pi / 2 + self.lib.flat_yaw("board")          # desk top long side ACROSS the lane
        info = dict(layout={k: v for k, v in L.items() if k != "crossing"}, P0=self.P0.tolist())

        # ---- terrace: deck (riser_layers layers from s_d) + stairs + platform, all desk tops across the lane
        RL = s_["riser_layers"]
        starts = L["step_starts"]                                       # step k: layers [k*RL, (k+1)*RL)
        nsteps = len(starts)
        cnt = [0] * nsteps; req = L["s_end_req"]
        for k in range(nsteps - 1, -1, -1):
            cnt[k] = max(1, int(math.ceil((req - starts[k]) / (BD + GAP) - 1e-6)))
            req = starts[k] + cnt[k] * (BD + GAP) - GAP
        ends = [starts[k] + cnt[k] * (BD + GAP) - GAP for k in range(nsteps)]
        layer = 0
        for k in range(nsteps):
            for jl in range(RL):
                zb = layer * t
                s0k = starts[k] + (HEAD_SEAT if (k == 0 and jl == RL - 1) else 0.0)   # top deck layer leaves the ramp seat
                for ia in range(cnt[k]):
                    sc = s0k + ia * (BD + GAP) + BD / 2
                    w = self.W(sc, 0.0, zb)
                    role = "deck" if k == 0 else ("platform" if k == nsteps - 1 else f"step{k}")
                    self.put(f"{role}_L{layer}", "board", bs, byaw, center=w[:2], z_bottom=w[2], orient="flat",
                             contacts=[("down", "flat")])
                layer += 1
        info["terrace"] = dict(steps=nsteps, layers=layer, boards_per_step=cnt, starts=[round(x, 4) for x in starts],
                               ends=[round(x, 4) for x in ends], tops=[round((k + 1) * RL * t, 4) for k in range(nsteps)])
        self.s_end = ends[-1]

        # ---- entry ramp: desk top pitched about the lateral axis, head on the deck's front edge (top of layer RL-2)
        th = math.radians(L["ramp_deg"])
        ramp = None
        for sg in (1, -1):
            P, signs, size = self._local("board", "flat", byaw, bs, sg * th, tuple(self.n))
            B = self.bottom_face(P, signs, size)
            hi = B[np.argsort(B[:, 2])[2:]]                             # the 2 upper bottom corners
            if (hi.mean(0)[:2] - B.mean(0)[:2]) @ self.d > 0:           # head towards +s
                ramp = (sg * th, B, hi)
                break
        pitch_r = ramp[0]

        def head_anchor(P, signs, size):
            B = self.bottom_face(P, signs, size); return B[np.argsort(B[:, 2])[2:]].mean(0)
        self.put("ramp", "board", bs, byaw, pitch=pitch_r, axis=tuple(self.n), anchor=head_anchor,
                 target=self.W(L["s_d"] + 0.5 * HEAD_SEAT, 0.0, L["head"]), orient="flat", contacts=[("down", "edges")])
        info["ramp"] = dict(deg=round(L["ramp_deg"], 3), head_z=round(L["head"], 4), foot_s=0.0, head_s=round(L["s_d"], 4))

        # ---- crossing ramps on the deck: pitch humps (ridge across the lane) and cross-slope valleys (V along it)
        sy0 = self.lib.size("slat")
        yaw_al = yaw + (0.0 if sy0[0] >= sy0[1] else math.pi / 2)        # slat long axis along the lane
        info["crossing"] = []
        dk = L["deck_top"]

        def upper_anchor(P, signs, size):
            B = self.bottom_face(P, signs, size); return B[np.argsort(B[:, 2])[2:]].mean(0)
        for i, el in enumerate(L["crossing"]):
            PL, PD, PT = el["panel"]; SL_, SH_, ST_ = el["support"]
            th = math.radians(el["deg"]); ss_ = s_["crossing"][i]["support_scale"]
            if el["type"] == "hump":
                C = self.W(el["c"], 0.0, dk)
                self.put(f"x{i}_hump_ridge", "slat", ss_, yaw_al + math.pi / 2, center=C[:2], z_bottom=C[2],
                         contacts=[("down", "flat")])
                u_sup = (0.25 * ST_) / (PD * math.cos(th)); v_sup = min(0.3, 0.4 * SL_) / PL
                for side in (-1, 1):                                       # -1: up panel (before the ridge)
                    axis = tuple(self.n)
                    for sg in (1, -1):
                        P, signs, size = self._local("board", "flat", byaw, s_["hump_scale"], sg * th, axis)
                        Bf = self.bottom_face(P, signs, size); hi = Bf[np.argsort(Bf[:, 2])[2:]]
                        if ((hi.mean(0)[:2] - Bf.mean(0)[:2]) @ self.d) * (-side) > 0:
                            break
                    self.put(f"x{i}_hump_panel{'ab'[side > 0]}", "board", s_["hump_scale"], byaw, pitch=sg * th, axis=axis,
                             anchor=upper_anchor, target=C + np.r_[0, 0, el["apex"]], orient="flat",
                             contacts=[("down", ("ridge", u_sup, v_sup))])
                info["crossing"].append(dict(type="hump", c_s=round(el["c"], 4), extent_s=[round(x, 4) for x in el["extent"]],
                                             panel_deg=round(el["deg"], 3), apex_bottom_above_deck=round(el["apex"], 4),
                                             ridge_slat=[round(x, 4) for x in el["support"]]))
            else:
                vyaw = yaw + self.lib.flat_yaw("board")                    # desk top long side ALONG the lane
                for side in (-1, 1):
                    # support slat on edge along the lane, its inner face at l_sup (under the panel's outer edge)
                    lc = side * (el["l_sup"] + 0.5 * ST_)
                    w = self.W(el["c"], lc, dk)
                    self.put(f"x{i}_valley_slat{'RL'[side > 0]}", "slat", ss_, yaw_al, center=w[:2], z_bottom=w[2],
                             contacts=[("down", "flat")])
                    axis = tuple(self.d)
                    for sg in (1, -1):
                        P, signs, size = self._local("board", "flat", vyaw, s_["valley_scale"], sg * th, axis)
                        Bf = self.bottom_face(P, signs, size); hi = Bf[np.argsort(Bf[:, 2])[2:]]
                        if ((hi.mean(0)[:2] - Bf.mean(0)[:2]) @ self.n) * side > 0:     # outer edge up
                            break
                    tgt = self.W(el["c"], side * el["l_out"], dk + PD * math.sin(th))
                    u_sup = (s_["valley_inset"] - 0.003) / (PD * math.cos(th)); v_sup = min(0.35, 0.4 * SL_) / PL
                    self.put(f"x{i}_valley_panel{'RL'[side > 0]}", "board", s_["valley_scale"], vyaw, pitch=sg * th, axis=axis,
                             anchor=upper_anchor, target=tgt, orient="flat", contacts=[("down", ("ridge", u_sup, v_sup))])
                info["crossing"].append(dict(type="valley", c_s=round(el["c"], 4), extent_s=[round(x, 4) for x in el["extent"]],
                                             panel_deg=round(el["deg"], 3), outer_edge_l=round(el["l_out"], 4),
                                             support_slat=[round(x, 4) for x in el["support"]]))

        # ---- lane fence (floor posts outside the deck, rails on the posts' lane-side faces)
        fs = s_["fence"]
        SLs, SHs, STs = self.dims["slat"]                               # (1.129, 0.187, 0.026) at scale 1
        ps = fs["post_scale"]
        Ppost = (SLs * ps, SHs * ps, STs * ps)                          # height, along-lane width, lateral thickness
        f_len = L["s_st"]
        nb = max(1, int(round(f_len / (SLs * fs["bay"]))))
        while f_len / nb > SLs * fs["rail_scale_max"]:
            nb += 1
        bay = f_len / nb; rsc = bay / SLs
        rail_t = STs * rsc
        l_rail_in = 0.5 * L["BL"] + 0.012
        l_post = l_rail_in + rail_t + 0.5 * Ppost[2]
        sy = self.lib.size("slat")
        yaw_al = yaw + (0.0 if sy[0] >= sy[1] else math.pi / 2)         # slat long axis along the lane
        for side in (-1, 1):
            for i in range(nb + 1):
                w = self.W(i * bay, side * l_post, 0.0)
                self.put(f"fence{'RL'[side > 0]}_post{i}", "slat", ps, yaw_al, pitch=math.pi / 2, axis=tuple(self.n),
                         center=w[:2], z_bottom=w[2], contacts=[("down", "flat")])
            for i in range(nb):
                for k, hz in enumerate(fs["rails_above_deck"]):
                    zc = L["deck_top"] + hz
                    w = self.W((i + 0.5) * bay, side * (l_rail_in + 0.5 * rail_t), zc)
                    self.put(f"fence{'RL'[side > 0]}_rail{i}_{k}", "slat", rsc, yaw_al, center=w[:2],
                             z_bottom=zc - 0.5 * SHs * rsc + FLOOR_Z, contacts=[("lat", side)])
        info["fence"] = dict(bays=nb, bay=round(bay, 4), rail_scale=round(rsc, 4), post_h=round(Ppost[0], 4),
                             post_top_above_deck=round(Ppost[0] - L["deck_top"], 4), rail_inner_l=round(l_rail_in, 4),
                             lane_width=round(2 * l_rail_in, 4), rails_above_deck=fs["rails_above_deck"], length=round(f_len, 4))

        # ---- stair handrails + platform enclosure (posts stand ON the treads / platform at the board edges)
        hr = s_["handrail"]; hp = hr["post_scale"]
        Hh = (SLs * hp, SHs * hp, STs * hp)
        tops = info["terrace"]["tops"]
        l_hp = 0.5 * L["BL"] - 0.5 * Hh[2] - 0.004                       # post centre, 4 mm inside the board edge
        s_t1 = 0.5 * (starts[1] + starts[2]) if nsteps > 2 else starts[1] + 0.15
        s_pf = L["s_p0"] + 0.5 * Hh[1] + 0.02
        z_t1, z_pf = tops[1], tops[-1]
        pe = ends[-1]
        n_side = max(1, int(round((pe - 0.5 * Hh[1] - 0.02 - s_pf) / 1.0)))
        side_posts = list(np.linspace(s_pf, pe - 0.5 * Hh[1] - 0.02, n_side + 1))
        for side in (-1, 1):
            tag = 'RL'[side > 0]
            self.put(f"hand{tag}_post_t1", "slat", hp, yaw_al, pitch=math.pi / 2, axis=tuple(self.n),
                     center=self.W(s_t1, side * l_hp)[:2], z_bottom=FLOOR_Z + z_t1, contacts=[("down", "flat")])
            for j_, sp in enumerate(side_posts):
                self.put(f"plat{tag}_post{j_}", "slat", hp, yaw_al, pitch=math.pi / 2, axis=tuple(self.n),
                         center=self.W(sp, side * l_hp)[:2], z_bottom=FLOOR_Z + z_pf, contacts=[("down", "flat")])
            # sloped handrails between the tread post and the first platform post (centre to centre)
            run = s_pf - s_t1; rise = z_pf - z_t1
            phi = math.atan2(rise, run); Lr = math.hypot(run, rise)
            rs_ = Lr / SLs
            rs_ = min(max(rs_, 0.85), 1.15)
            l_r = side * (l_hp - 0.5 * Hh[2] - 0.5 * STs * rs_)
            for k, hz in enumerate(hr["rails_above"]):
                mid = self.W(0.5 * (s_t1 + s_pf), l_r, 0.5 * (z_t1 + z_pf) + hz)

                def mid_anchor(P, signs, size):
                    return P.mean(0)
                for sg in (1, -1):
                    P, signs, size = self._local("slat", "as_is", yaw_al, rs_, sg * phi, tuple(self.n))
                    i_long = int(np.argmax(size))
                    hiend = P[signs[:, i_long] == 1].mean(0); loend = P[signs[:, i_long] == -1].mean(0)
                    if (hiend[2] - loend[2]) * ((hiend[:2] - loend[:2]) @ self.d) > 0:
                        break
                self.put(f"hand{tag}_rail{k}", "slat", rs_, yaw_al, pitch=sg * phi, axis=tuple(self.n), anchor=mid_anchor,
                         target=mid, contacts=[("lat", side)])
            # platform side rails between consecutive side posts
            for j_ in range(len(side_posts) - 1):
                a_, b_ = side_posts[j_], side_posts[j_ + 1]
                rsc2 = (b_ - a_) / SLs
                for k, hz in enumerate(hr["rails_above"]):
                    zc = z_pf + hz
                    w = self.W(0.5 * (a_ + b_), side * (l_hp - 0.5 * Hh[2] - 0.5 * STs * rsc2), zc)
                    self.put(f"plat{tag}_rail{j_}_{k}", "slat", rsc2, yaw_al, center=w[:2],
                             z_bottom=FLOOR_Z + zc - 0.5 * SHs * rsc2, contacts=[("lat", side)])
        # far-end rails across the lane, wedged between the two last side posts (end faces on the posts' broad faces)
        span = 2 * (l_hp - 0.5 * Hh[2])
        rs_e = span / SLs
        yaw_ac = yaw_al + math.pi / 2
        for k, hz in enumerate(hr["rails_above"]):
            zc = z_pf + hz
            w = self.W(side_posts[-1] + 0.5 * Hh[1] - 0.5 * STs * rs_e - 0.003, 0.0, zc)   # beyond the side rails' ends
            self.put(f"plat_end_rail{k}", "slat", rs_e, yaw_ac, center=w[:2], z_bottom=FLOOR_Z + zc - 0.5 * SHs * rs_e,
                     contacts=[("ends", None)])
        info["handrail"] = dict(post_h=round(Hh[0], 4), stair_rail_deg=round(math.degrees(phi), 3), stair_rail_scale=round(rs_, 4),
                                rails_above=hr["rails_above"], platform_posts_s=[round(x, 4) for x in side_posts],
                                end_rail_scale=round(rs_e, 4), stair_clear_width=round(2 * (l_hp - 0.5 * Hh[2] - STs * rs_), 4))
        info["n_parts"] = len(self.parts)
        info["s_end"] = round(self.s_end, 4)
        # robot task: start on the gym floor 1.2 m before the ramp foot, goal at the platform centre
        st_w = self.W(-1.2, 0.0); gl = self.W(0.5 * (L["s_p0"] + side_posts[-1]), 0.0, tops[-1])
        path = [self.W(-1.2, 0, 0), self.W(0.0, 0, 0), self.W(L["s_d"], 0, L["deck_top"]), self.W(L["s_st"], 0, L["deck_top"])]
        for k in range(1, nsteps):
            path.append(self.W(starts[k], 0, tops[k]))
        path.append(gl)
        info["robot"] = dict(start=[round(st_w[0], 3), round(st_w[1], 3), round(yaw, 4)], goal=np.round(gl, 3).tolist(),
                             path=[np.round(p, 3).tolist() for p in path])
        self.info = info
        return info

    # ------------------------------------------------------------ verification (PhysX rays)
    def verify(self, pxq, carb):
        def ray(o, v, L, skip=None):
            """closest hit, skipping colliders under the prim path `skip` (restart just past them)"""
            o = np.asarray(o, float); v = np.asarray(v, float); v = v / np.linalg.norm(v); done = 0.0
            for _ in range(12):
                h = pxq.raycast_closest(carb.Float3(*map(float, o)), carb.Float3(*map(float, v)), float(L - done))
                if not (h and h.get("hit")):
                    return None
                c = str(h.get("collision", "")); dist = float(h["distance"])
                if skip and c.startswith(skip):
                    o = o + v * (dist + 1e-4); done += dist + 1e-4; continue
                return dict(d=done + dist, c=c, p=np.asarray(h["position"], float))
            return None
        L, info = self.L, self.info
        res = dict(parts=len(self.parts), checks=0, fail=[], max_gap=0.0)
        for p in self.parts:
            P, sg, size = p["corners"], p["signs"], p["size"]
            for kind_, mode in p["contacts"]:
                pts, dirs = [], []
                if kind_ == "down":
                    B = self.bottom_face(P, sg, size)
                    cB = B.mean(0)
                    if mode == "flat":
                        pts = [cB + 0.7 * (b - cB) for b in B] + [cB]
                    elif isinstance(mode, tuple):                        # hump panel: lower edge on the deck + book top edge
                        o = np.argsort(B[:, 2]); lo2, hi2 = B[o[:2]], B[o[2:]]
                        u, v = mode[1], mode[2]
                        lm, hm_ = lo2.mean(0), hi2.mean(0); ev = hi2[1] - hi2[0]
                        pts = [lo2[0] + 0.25 * (lo2[1] - lo2[0]), lo2[1] + 0.25 * (lo2[0] - lo2[1])]
                        for sv in (-1, 1):
                            pts.append(hm_ + u * (lm - hm_) + sv * v * ev)
                    else:                                                # sloped: lower + upper edge points
                        o = np.argsort(B[:, 2])
                        lo2, hi2 = B[o[:2]], B[o[2:]]
                        k_ = 0.25 if mode == "edges" else 0.30
                        pts = [lo2[0] + k_ * (lo2[1] - lo2[0]), lo2[1] + k_ * (lo2[0] - lo2[1]),
                               hi2[0] + k_ * (hi2[1] - hi2[0]) + 0.002 * (cB - hi2.mean(0)) / max(1e-9, np.linalg.norm(cB - hi2.mean(0))),
                               hi2[1] + k_ * (hi2[0] - hi2[1]) + 0.002 * (cB - hi2.mean(0)) / max(1e-9, np.linalg.norm(cB - hi2.mean(0)))]
                    dirs = [np.array([0, 0, -1.0])] * len(pts)
                elif kind_ == "lat":                                     # rail on the lane side of its posts
                    i_long = int(np.argmax(size))
                    n_out = np.r_[self.n * mode, 0.0]
                    for e in (-1, 1):
                        F = P[sg[:, i_long] == e]
                        ctr = 0.5 * (P.mean(0) + F.mean(0))              # 3/4 of the way towards this end
                        ctr = P.mean(0) + 0.85 * (F.mean(0) - P.mean(0))
                        half_t = 0.5 * float(np.min(size)) * p["scale"]
                        pts.append(ctr + n_out * (half_t - 0.001)); dirs.append(n_out)
                elif kind_ == "ends":
                    i_long = int(np.argmax(size))
                    for e in (-1, 1):
                        F = P[sg[:, i_long] == e]; c0 = P.mean(0)
                        v = F.mean(0) - c0; v = v / np.linalg.norm(v)
                        pts.append(F.mean(0) - v * 0.001); dirs.append(v)
                for q, v in zip(pts, dirs):
                    res["checks"] += 1
                    h = ray(q - v * 0.02, v, 0.08, skip=p["path"])
                    gap = None if h is None else h["d"] - 0.02
                    if gap is None or gap > 0.005 or gap < -0.01:
                        res["fail"].append(dict(part=p["role"], mode=f"{kind_}:{mode}", at=np.round(q, 3).tolist(),
                                                gap=None if gap is None else round(gap, 4), hit=None if h is None else h["c"][-60:]))
                    else:
                        res["max_gap"] = max(res["max_gap"], round(gap, 4))
        # walking-surface profiles (lane centre and +-0.35 m)
        prof = {}
        s_hi = self.s_end + 0.3
        for l in (-0.35, 0.0, 0.35):
            rows = []
            for s in np.arange(-0.5, s_hi, 0.02):
                o = self.W(s, l, 3.0)
                h = ray(o, (0, 0, -1), 4.0)
                rows.append((round(float(s), 3), None if h is None else round(float(h["p"][2] - FLOOR_Z), 4)))
            prof[str(l)] = rows
        res["profile"] = prof
        meas = {}
        c = np.array([(s, z) for s, z in prof["0.0"] if z is not None])
        sd = L["s_d"]
        m = (c[:, 0] > 0.12) & (c[:, 0] < sd - 0.12)
        meas["ramp_deg"] = round(math.degrees(math.atan(np.polyfit(c[m, 0], c[m, 1], 1)[0])), 2)
        m = (c[:, 0] > sd + 0.2) & (c[:, 0] < sd + self.info["layout"]["s_d"] + 1.0)
        meas["deck_top_z"] = round(float(np.median(c[m, 1])), 4)
        # stairs: tread heights from the profile between step starts
        st = self.info["terrace"]["starts"]; tops = []
        for k in range(1, len(st)):
            a = st[k] + 0.05; b = (st[k + 1] - 0.05) if k + 1 < len(st) else st[k] + 0.6
            mm = (c[:, 0] > a) & (c[:, 0] < b)
            tops.append(round(float(np.median(c[mm, 1])), 4))
        meas["tread_tops_z"] = tops
        meas["risers"] = [round(b - a, 4) for a, b in zip([meas["deck_top_z"]] + tops[:-1], tops)]
        meas["platform_z"] = tops[-1]
        # humps: panel planes from 3 down-rays inside each panel -> slope, pitch, roll w.r.t. the lane
        hm = []
        for p in self.parts:
            if "_panel" not in p["role"]:
                continue
            P, sg, size = p["corners"], p["signs"], p["size"]
            B = self.bottom_face(P, sg, size); cB = B.mean(0)
            pts = []
            for k_ in range(3):
                q = cB + 0.5 * (B[k_] - cB)
                h = ray(np.r_[q[:2], q[2] + 1.0], (0, 0, -1), 1.5)
                pts.append(h["p"])
            A = np.array(pts); nrm = np.cross(A[1] - A[0], A[2] - A[0]); nrm = nrm / np.linalg.norm(nrm)
            if nrm[2] < 0:
                nrm = -nrm
            slope = math.degrees(math.acos(nrm[2]))
            g = -nrm[:2] / nrm[2]                                       # surface gradient dz/dx, dz/dy
            pitch = math.degrees(math.atan(abs(g @ self.d))); roll = math.degrees(math.atan(abs(g @ self.n)))
            hm.append(dict(part=p["role"], slope_deg=round(slope, 2), pitch_deg=round(pitch, 2), roll_deg=round(roll, 2)))
        meas["hump_panels"] = hm
        tops_x = []
        for el in self.info["crossing"]:
            if el["type"] == "hump":
                hit = ray(self.W(el["c_s"], 0.0, 3.0), (0, 0, -1), 4.0)
                tops_x.append(dict(type="hump", top_above_deck=round(float(hit["p"][2] - FLOOR_Z) - meas["deck_top_z"], 4)))
            else:
                hc = ray(self.W(el["c_s"], 0.0, 3.0), (0, 0, -1), 4.0)
                ho = ray(self.W(el["c_s"], 0.55, 3.0), (0, 0, -1), 4.0)
                tops_x.append(dict(type="valley", centre_above_deck=round(float(hc["p"][2] - FLOOR_Z) - meas["deck_top_z"], 4),
                                   at_l055_above_deck=round(float(ho["p"][2] - FLOOR_Z) - meas["deck_top_z"], 4)))
        meas["crossing_heights"] = tops_x
        # lane width between the rails at mid-rail height, every bay
        widths = []
        f = self.info["fence"]
        for i in range(f["bays"]):
            s = (i + 0.5) * f["bay"] + 0.2
            o = self.W(s, 0.0, self.L["deck_top"] + 0.58)
            hp_ = ray(o, np.r_[self.n, 0], 1.5); hm_ = ray(o, np.r_[-self.n, 0], 1.5)
            if hp_ and hm_:
                widths.append(round(hp_["d"] + hm_["d"], 4))
        meas["lane_width"] = dict(min=min(widths), max=max(widths), n=len(widths)) if widths else None
        # fence / handrail heights: top of the highest rail above the walking surface beneath it
        def top_height(prefix):
            out = []
            for p in self.parts:
                if not p["role"].startswith(prefix) or "_rail" not in p["role"]:
                    continue
                cP = p["corners"].mean(0)
                h = ray(np.r_[cP[:2], cP[2] + 1.0], (0, 0, -1), 1.2)
                if h is None or not h["c"].startswith(p["path"]):
                    continue
                # surface 0.3 m inside the lane at the same s
                s_ = float((cP[:2] - self.P0) @ self.d)
                g = ray(self.W(s_, 0.0, 3.0), (0, 0, -1), 4.0)
                out.append(round(float(h["p"][2] - g["p"][2]), 4))
            return out
        fh = top_height("fence"); hh = top_height("hand"); ph = top_height("plat")
        meas["fence_top_rail_above_walk"] = max(fh) if fh else None
        meas["stair_handrail_top_above_tread_centre_line"] = max(hh) if hh else None
        meas["platform_rail_top_above_platform"] = max(ph) if ph else None
        res["measured"] = meas
        res["ok"] = not res["fail"]
        return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sp = make_spec(a.seed)
    js = json.dumps(sp, indent=1)
    if a.out:
        open(a.out, "w").write(js)
    print(js)
