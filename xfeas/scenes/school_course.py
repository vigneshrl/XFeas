#!/usr/bin/env python
"""
SCHOOL COURSE (experiments/3_school_course; additive 2026-10-20, no existing script is changed). One straight lane on the
ground floor of the Abandoned School, built IN MEMORY in assets/school.usd (the file is never written). Sections in
travel order, ~1.6 m of clear floor between them:

  1. SPEED STRAIGHT  >= 5.5 m of clear school floor from the start.
  2. HURDLE          a real gym bench (SM_gym_props_5, the school's own) across the lane, standing on its own legs;
                     uniform scale 0.85 -> top ~0.255 m.
  3. LOG             round step-over across the lane resting on the floor (see LOG_KIND below; no log / trunk asset
                     exists in the mirrored NVIDIA Vegetation library, which only has Debris leaf litter, Leaves, Rocks).
  4. SMALL JUMP      two raised decks of laminated school desk tops (BP_school_desk_26/SM_school_desk_2_2, as the gym
                     course deck) with a ditch (the school floor) between them: step up, jump the ditch, step down.
  5. CROUCH          a real panelled school door leaf (InstancedSM_school_door_4_2) laid flat across the lane, resting on
                     two upright wooden cloakroom partitions (SM_cloakroom_2) beside the lane, underside ~1.05 m.
  6. SIDESTEP        a cross wall of real blue school lockers (BP_school_closet_1) closing the corridor except one gap of
                     ~0.38 m at the lane centre (fronts facing the robot; outer pieces reach into the corridor walls).

Every part is a copy of a real school asset made by real_assets.RealLib (hazard_copy recipe: exact sub-prim, composed
pose, instancing broken, exact triangle colliders, the asset's own materials, uniform scale 0.85-1.15).
Nothing floats: verify() casts PhysX rays down from the lowest vertex band of every part (skipping the part itself) and
requires (a) supported vertices with gap <= 5 mm and (b) the part's centroid inside the convex hull of its supported
vertices (static stability), and checks the walking surface along the lane is a VISIBLE school floor mesh or a course
part (no invisible collision plane).

  spec = make_spec(loc)                          # pure python
  B = SchoolCourse(stage, spec); off = B.prepare(); info = B.build(); world.reset(); ver = B.verify(pxq, carb)
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, math, os, sys
import numpy as np

ROOT = xfeas.paths.ROOT
ROOT_PRIM = "/World/SchoolCourse"
G = "/World/School/Geom/"
GAP = 0.002
CATALOG = dict(
    board=dict(src="BP_school_desk_26/SM_school_desk_2_2"),
    slat=dict(src="BP_school_desk_26/SM_school_desk_2_3"),
    pipe=dict(src=os.environ.get("SC_PIPE_SRC", "SM_sewer_pipes_8")),
)
LOG_KIND = os.environ.get("SC_LOG_KIND", "pipe")

# candidate locations (lane frame: P0 = start, yaw = travel direction, floor = walking surface z)
LOCATIONS = {
    # ground-floor south hall (z 0.681), lane along +x from the west corridor junction (survey 2026-10-20)
    "south_hall": dict(P0=[-26.9, -5.8], yaw=0.0, floor=0.681, half=1.35),
    # ground-floor west corridor (z 0.681), lane along +y from the south end
    "west_corr": dict(P0=[-26.4, -8.3], yaw=math.pi / 2, floor=0.681, half=1.35),
}


def make_spec(loc="south_hall", **over):
    L = dict(LOCATIONS[loc]) if isinstance(loc, str) else dict(loc)
    L.update({k: v for k, v in over.items() if k in ("P0", "yaw", "floor", "half")})
    sp = dict(
        version=1,
        location=loc if isinstance(loc, str) else "custom",
        lane=L,
        straight=5.5,  # start .. hurdle
        clear=1.5,  # clear floor between sections
        hurdle=dict(kind="bench", scale=0.85),
        log=dict(kind=LOG_KIND, scale=1.15),
        jump=dict(layers=6, board_scale=1.10, boards_a=2, boards_b=2, gap=0.40),
        crouch=dict(
            u=1.05,
            leaf_scale=1.0,
            support="partition",
            l_in=1.0,
            n_leaves=2,
            side_walls=[k for k in os.environ.get("SC_CROUCH_WALLS", "wardrobe,wardrobe1").split(",") if k],
            wall_depth=0.58,
        ),
        sidestep=dict(gap=0.34, gap_l=-0.35, kinds=["locker"], srange=[0.95, 1.05]),
        clear_sidestep=2.6,  # crouch .. sidestep: open floor seen through the crouch opening
        goal_after=1.2,
    )
    for k, v in over.items():
        if k in sp and isinstance(sp[k], dict) and isinstance(v, dict):
            sp[k].update(v)
        elif k in sp and k != "lane":
            sp[k] = v
    return sp


def _hull(P):
    P = sorted(map(tuple, np.round(np.asarray(P, float), 5)))
    P = list(dict.fromkeys(P))
    if len(P) < 3:
        return np.array(P)
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def _dist_to_hull(H, q):
    """0 if q is inside the convex polygon H (ccw), else the distance to it (segments / points handled)"""
    q = np.asarray(q, float)
    if len(H) == 0:
        return 1e9
    if len(H) == 1:
        return float(np.linalg.norm(H[0] - q))
    inside = len(H) >= 3 and all(
        ((H[(i + 1) % len(H)] - H[i])[0] * (q - H[i])[1] - (H[(i + 1) % len(H)] - H[i])[1] * (q - H[i])[0]) >= -1e-9
        for i in range(len(H))
    )
    if inside:
        return 0.0
    best = 1e9
    for i in range(len(H)):
        a, b = H[i], H[(i + 1) % len(H)]
        ab = b - a
        t = float(np.clip((q - a) @ ab / max(ab @ ab, 1e-12), 0, 1))
        best = min(best, float(np.linalg.norm(a + t * ab - q)))
    return best


class SchoolCourse:
    def __init__(self, stage, spec, root=ROOT_PRIM):
        import real_assets as RA

        self.RA, self.stage, self.spec = RA, stage, spec
        self.root_req = root
        L = spec["lane"]
        self.P0 = np.asarray(L["P0"], float)
        self.yaw = float(L["yaw"])
        self.F = float(L["floor"])
        self.d = np.array([math.cos(self.yaw), math.sin(self.yaw)])
        self.n = np.array([-self.d[1], self.d[0]])
        self.parts = []
        self.lib = None

    # ---------------------------------------------------------------- frame
    def W(self, s, l=0.0, z=0.0):
        xy = self.P0 + self.d * s + self.n * l
        return np.array([xy[0], xy[1], self.F + z])

    def lane_of(self, xy):
        v = np.asarray(xy, float)[:2] - self.P0
        return float(v @ self.d), float(v @ self.n)

    # ---------------------------------------------------------------- scene prep (in memory)
    def prepare(self, pxq=None, carb=None, s_hi=None, half=None, dry=False):
        """clean school (scene_clean) + deactivate, IN MEMORY, loose props whose box touches the lane volume
        (s -1 .. s_hi, |l| <= half + 0.25, floor+0.02 .. floor+2.3). Structure (any extent > 3.5 m) is never touched."""
        import scene_clean, baked_scenes as BK
        from pxr import Usd, UsdGeom, UsdLux

        st = self.stage
        self.init_lib()
        for p in ("/World/Spot", "/World/Surfaces"):
            if st.GetPrimAtPath(p).IsValid():
                st.GetPrimAtPath(p).SetActive(False)
        scene_clean.apply(st)
        s_hi = s_hi if s_hi is not None else self.length_estimate() + 1.0
        half = half if half is not None else self.spec["lane"]["half"] + 0.25
        bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        cand = []
        geom = st.GetPrimAtPath("/World/School/Geom")
        roots = [c for c in geom.GetChildren()]
        for sid, rr in BK.baked(st).items():
            for r in rr:
                p = st.GetPrimAtPath(r)
                if p.IsValid():
                    roots.append(p)
        rect = [self.W(-1.0, -half)[:2], self.W(-1.0, half)[:2], self.W(s_hi, half)[:2], self.W(s_hi, -half)[:2]]
        rect = np.array(rect)
        hw = half + 1.4
        wide = np.array([self.W(-0.5, -hw)[:2], self.W(-0.5, hw)[:2], self.W(s_hi, hw)[:2], self.W(s_hi, -hw)[:2]])
        off, struct = [], []

        def visit(ch, depth):
            if not ch.IsActive() or ch.IsA(UsdLux.BoundableLightBase) or ch.IsA(UsdLux.NonboundableLightBase):
                return
            r = bbc.ComputeWorldBound(ch).ComputeAlignedRange()
            if r.IsEmpty():
                return
            mn, mx = np.array(r.GetMin()), np.array(r.GetMax())
            name = ch.GetName()
            decal = name.startswith(("Oil_surface", "Rubble_surface"))  # slippery-hazard decals of S1/S2
            if (mx[2] < self.F + 0.004 and not decal) or mn[2] > self.F + 2.3 or mx[2] < self.F - 0.05:
                return
            B = np.array([[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]])
            sz = mx - mn
            # lane box; plus a wider DECLUTTER box for small props lying / leaning on the floor (boards, panels)
            small_floor = (
                mn[2] < self.F + 0.10
                and max(sz) < 3.0
                and not name.startswith(
                    ("SM_heating", "BP_school_window", "SM_skirting", "SM_school_window", "BP_school_door")
                )
            )
            for R_ in (rect, wide) if small_floor else (rect,):
                hit = True
                for ax in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), self.d, self.n):
                    a_, b_ = B @ ax, R_ @ ax
                    if a_.max() < b_.min() or b_.max() < a_.min():
                        hit = False
                        break
                if hit:
                    break
            if not hit:
                return
            rec = dict(path=str(ch.GetPath()), min=np.round(mn, 3).tolist(), max=np.round(mx, 3).tolist())
            if max(sz[:2]) > 3.5 and not decal:
                kids = [q for q in ch.GetChildren()]
                if sz[2] < 3.0 and mn[2] >= self.F - 0.05 and not kids:  # one loose free-standing prop (a
                    off.append(rec)  # gym wall-bar frame across the hall)
                    if not dry:
                        ch.SetActive(False)
                    return
                if depth < 3 and kids and not ch.IsInstance():
                    for q in kids:  # loose parts of big blueprints (a door
                        visit(q, depth + 1)  # leaf lying on the floor, ...)
                else:
                    struct.append(rec)
                return
            if sz[2] > 3.0:  # columns / wall pieces: never touched
                struct.append(rec)
                return
            off.append(rec)
            if not dry:
                ch.SetActive(False)

        for ch in roots:
            visit(ch, 0)
        self.prep = dict(deactivated=off, structure_in_lane=struct, rect=rect.round(3).tolist())
        return self.prep

    def init_lib(self):
        """create the RealLib and resolve every source (frames from the school's own prims) BEFORE loose props in
        the lane are deactivated (a deactivated source cannot be measured; copies reference the asset file)"""
        if self.lib is None:
            self.lib = self.RA.RealLib(self.stage, root=self.root_req, catalog=CATALOG)
            self.root = self.lib.root
            for k in ("board", "pipe", "bench", "door_leaf", "partition", "locker", "wardrobe", "wardrobe1"):
                self.lib.size(k)
        return self.lib

    def length_estimate(self):
        sp = self.spec
        return (
            sp["straight"]
            + 0.3
            + sp["clear"]
            + 0.3
            + sp["clear"]
            + 3.4
            + sp["clear"]
            + 1.9
            + sp.get("clear_sidestep", sp["clear"])
            + 0.8
            + sp["goal_after"]
        )

    # ---------------------------------------------------------------- placement helpers
    def put(
        self, role, kind, x, y, z_bottom, orient="as_is", yaw=0.0, scale=1.0, pitch=0.0, axis=(0.0, 1.0), section=""
    ):
        rec = self.lib.place(
            kind, float(x), float(y), float(z_bottom), orient, float(yaw), float(scale), float(pitch), axis
        )
        R, P = self.lib.orient_matrix(kind, orient, yaw, scale, pitch, axis)
        zc = z_bottom - P[:, 2].min()
        part = dict(
            role=role,
            section=section,
            kind=kind,
            src=self.lib.catalog[kind]["src"],
            scale=round(float(scale), 4),
            orient=orient,
            yaw=round(float(yaw), 5),
            path=str(rec["copy"]["prim"].GetPath()),
            prim=rec["copy"]["prim"],
            corners=P + np.array([x, y, zc]),
            size=self.lib.size(kind) * scale,
            center=np.array([x, y, zc]),
        )
        self.parts.append(part)
        return part

    def footprint(self, part, na=12, nb=4, margin=0.01):
        """grid of xy points on the TOP face of a box-like part (its OBB top rectangle, inset by margin)"""
        P = part["corners"]
        top = P[np.argsort(P[:, 2])[4:], :2]
        o = top[0]
        dd = sorted(top[1:], key=lambda q: np.linalg.norm(q - o))
        e1, e2 = dd[0] - o, dd[1] - o  # the two edges from corner o (the diagonal is longest)
        L1, L2 = np.linalg.norm(e1), np.linalg.norm(e2)
        out = []
        for i in np.linspace(margin / L1, 1 - margin / L1, nb):
            for j in np.linspace(margin / L2, 1 - margin / L2, na):
                out.append(o + e1 * i + e2 * j)
        return out

    def lat_extent(self, kind, orient, yaw, scale=1.0):
        R, P = self.lib.orient_matrix(kind, orient, yaw, scale)
        return float(np.ptp(P[:, :2] @ self.n)), float(np.ptp(P[:, :2] @ self.d)), float(np.ptp(P[:, 2]))

    def yaw_across(self, kind, orient):
        """yaw making the object's longest horizontal side ACROSS the lane"""
        best = None
        for extra in (0.0, math.pi / 2):
            lat, al, H = self.lat_extent(kind, orient, self.yaw + extra)
            if best is None or lat > best[0]:
                best = (lat, extra)
        return self.yaw + best[1]

    def yaw_along(self, kind, orient):
        best = None
        for extra in (0.0, math.pi / 2):
            lat, al, H = self.lat_extent(kind, orient, self.yaw + extra)
            if best is None or lat < best[0]:
                best = (lat, extra)
        return self.yaw + best[1]

    def wall_dist(self, pxq, carb, s, z=1.0, side=1, skip=None):
        o = self.W(s, 0.0, z)
        v = np.r_[self.n * side, 0.0]
        h = self.ray(pxq, carb, o, v, 6.0, skip=skip)
        return None if h is None else h["d"]

    @staticmethod
    def ray(pxq, carb, o, v, L, skip=None):
        o = np.asarray(o, float)
        v = np.asarray(v, float)
        v = v / np.linalg.norm(v)
        done = 0.0
        for _ in range(16):
            h = pxq.raycast_closest(carb.Float3(*map(float, o)), carb.Float3(*map(float, v)), float(L - done))
            if not (h and h.get("hit")):
                return None
            c = str(h.get("collision", ""))
            dist = float(h["distance"])
            if skip and any(c.startswith(s_) for s_ in (skip if isinstance(skip, (list, tuple)) else [skip])):
                o = o + v * (dist + 1e-4)
                done += dist + 1e-4
                continue
            return dict(d=done + dist, c=c, p=np.asarray(h["position"], float))
        return None

    # ---------------------------------------------------------------- build
    def build(self, walls=None, walls_at=None):
        """walls = (left, right) corridor wall distance from the lane centre at the sidestep line (measured before the
        build with rays; None -> spec lane half + 0.3)"""
        RA = self.RA
        self.init_lib()
        sp = self.spec
        F = self.F
        info = dict(sections=[])
        s = sp["straight"]
        info["sections"].append(dict(name="speed_straight", s=[0.0, round(s, 3)], length=round(s, 3)))

        # ---- 2. HURDLE: gym bench across the lane on its own legs
        h = sp["hurdle"]
        yb = self.yaw_across(h["kind"], "as_is")
        lat, al, H = self.lat_extent(h["kind"], "as_is", yb, h["scale"])
        c = self.W(s + al / 2, 0.0)
        self.put("hurdle_bench", h["kind"], c[0], c[1], F, "as_is", yb, h["scale"], section="hurdle")
        info["sections"].append(
            dict(
                name="hurdle",
                s=[round(s, 3), round(s + al, 3)],
                what=f"school gym bench {self.lib.catalog[h['kind']]['src']}",
                scale=h["scale"],
                top=round(H, 4),
                length_across=round(lat, 3),
                depth=round(al, 3),
            )
        )
        s += al + sp["clear"]

        # ---- 3. LOG: round piece(s) lying across the lane on the floor, end to end
        g = sp["log"]
        yl = self.yaw_across(g["kind"], "on_side")
        lat, al, H = self.lat_extent(g["kind"], "on_side", yl, g["scale"])
        need = 2 * sp["lane"]["half"]
        n = max(1, int(math.ceil(need / lat - 0.05)))
        tilts = []
        for i in range(n):
            lc = -0.5 * n * lat + (i + 0.5) * lat
            c = self.W(s + al / 2, lc)
            sg_ = 1 if i % 2 == 0 else -1  # alternate the bell ends
            yli = yl + (0.0 if sg_ > 0 else math.pi)
            part = self.put(f"log_{i}", g["kind"], c[0], c[1], F, "on_side", yli, g["scale"], section="log")
            # a pipe with a socket (bell) at one end rests on the bell AND the far barrel end: tilt it about the
            # lane axis by the slope of the lower convex hull edge under its centroid, then re-seat it on the floor
            A = RA.world_points(part["prim"])
            t_ = (A[:, :2] - c[:2]) @ self.n
            z_ = A[:, 2]
            H2 = _hull(np.c_[t_, z_])
            com_t = float(np.mean(t_))
            best = None
            for k_ in range(len(H2)):
                p1, p2 = H2[k_], H2[(k_ + 1) % len(H2)]
                if (p1[0] - com_t) * (p2[0] - com_t) <= 0 and abs(p2[0] - p1[0]) > 1e-6:
                    zc_ = p1[1] + (p2[1] - p1[1]) * (com_t - p1[0]) / (p2[0] - p1[0])
                    if best is None or zc_ < best[0]:
                        best = (zc_, math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            phi = best[1] if best else 0.0
            self.parts.pop()
            self.lib.used[g["kind"]] -= 1
            for sgp in (1, -1):
                part = self.put(
                    f"log_{i}",
                    g["kind"],
                    c[0],
                    c[1],
                    F,
                    "on_side",
                    yli,
                    g["scale"],
                    pitch=sgp * phi,
                    axis=(float(self.d[0]), float(self.d[1])),
                    section="log",
                )
                A = RA.world_points(part["prim"])
                t_ = (A[:, :2] - c[:2]) @ self.n
                z_ = A[:, 2]
                lo = z_ <= z_.min() + 0.003
                if np.ptp(t_[lo]) > 0.5 * lat or abs(phi) < 1e-4:
                    break
                self.parts.pop()
                self.lib.used[g["kind"]] -= 1
            dz = float(z_.min()) - F  # exact seat on the floor
            if abs(dz) > 1e-5:
                self.parts.pop()
                self.lib.used[g["kind"]] -= 1
                part = self.put(
                    f"log_{i}",
                    g["kind"],
                    c[0],
                    c[1],
                    F - dz,
                    "on_side",
                    yli,
                    g["scale"],
                    pitch=sgp * phi,
                    axis=(float(self.d[0]), float(self.d[1])),
                    section="log",
                )
            tilts.append(round(math.degrees(phi), 3))
        H = max(float(RA.world_points(p_["prim"])[:, 2].max()) - F for p_ in self.parts if p_["section"] == "log")
        info["sections"].append(
            dict(
                name="log",
                s=[round(s, 3), round(s + al, 3)],
                what=f"{n} x {self.lib.catalog[g['kind']]['src']} lying across",
                scale=g["scale"],
                top=round(H, 4),
                depth=round(al, 3),
                length_across=round(n * lat, 3),
                tilt_deg=tilts,
                note="cast-iron sewer pipe with a socket at one end; tilted to rest on socket + barrel",
            )
        )
        s += al + sp["clear"]

        # ---- 4. SMALL JUMP: deck A | ditch | deck B (desk-top laminates; boards ACROSS the lane, 2 side by side)
        j = sp["jump"]
        bs = j["board_scale"]
        BL, BD, BT = self.lib.flat_dims("board")
        BL, BD, BT = BL * bs, BD * bs, BT * bs
        byaw = self.yaw + math.pi / 2 + self.lib.flat_yaw("board")  # long side across the lane
        nlat = max(1, int(math.ceil(need / BL - 0.05)))
        deck_top = j["layers"] * BT
        s_a0 = s

        def deck(tag, s0, nb):
            for layer in range(j["layers"]):
                for ia in range(nb):
                    sc_ = s0 + ia * (BD + GAP) + BD / 2
                    for ic in range(nlat):
                        lc = -0.5 * (nlat * BL + (nlat - 1) * GAP) + ic * (BL + GAP) + BL / 2
                        w = self.W(sc_, lc)
                        self.put(
                            f"deck{tag}_L{layer}_{ia}_{ic}",
                            "board",
                            w[0],
                            w[1],
                            F + layer * BT,
                            "flat",
                            byaw,
                            bs,
                            section="jump",
                        )
            return s0 + nb * (BD + GAP) - GAP

        s_a1 = deck("A", s_a0, j["boards_a"])
        s_b0 = s_a1 + j["gap"]
        s_b1 = deck("B", s_b0, j["boards_b"])
        info["sections"].append(
            dict(
                name="small_jump",
                s=[round(s_a0, 3), round(s_b1, 3)],
                deck_top=round(deck_top, 4),
                deck_a=[round(s_a0, 3), round(s_a1, 3)],
                ditch=[round(s_a1, 3), round(s_b0, 3)],
                ditch_width=j["gap"],
                deck_b=[round(s_b0, 3), round(s_b1, 3)],
                width_across=round(nlat * BL + (nlat - 1) * GAP, 3),
                what=f"{j['layers']} layers of school desk tops {self.lib.catalog['board']['src']} x {bs}",
            )
        )
        s = s_b1 + sp["clear"]

        # ---- 5. CROUCH: door leaf flat across the lane on two upright cloakroom partitions
        cr = sp["crouch"]
        Lw, Dw, Tw = self.lib.flat_dims("door_leaf")
        ls = cr["leaf_scale"]
        # partitions first (their long side along the lane), then the leaf on them; the leaf is centred on them
        pk = cr["support"]
        ya = self.yaw_along(pk, "as_is")
        latp, alp, Hp = self.lat_extent(pk, "as_is", ya, 1.0)
        psc = cr["u"] / Hp
        if not (RA.SCALE_LIM[0] <= psc <= RA.SCALE_LIM[1]):
            raise ValueError(f"crouch support scale {psc:.3f} outside the band")
        Dw_all = int(cr.get("n_leaves", 1)) * Dw * ls + (int(cr.get("n_leaves", 1)) - 1) * GAP
        s_c = s + max(0.5 * alp * psc, 0.5 * Dw_all)  # centre of the crouch element
        tops = []
        for side in (1, -1):
            lc = side * (cr["l_in"] + 0.5 * latp * psc)
            w = self.W(s_c, lc)
            p = self.put(
                f"crouch_support_{'L' if side > 0 else 'R'}", pk, w[0], w[1], F, "as_is", ya, psc, section="crouch"
            )
            tops.append(F + Hp * psc)
        # the leaf (or leaves side by side along the lane): lowest vertex over the partitions at the support top
        import gym_course_v3 as GV

        nlv = int(cr.get("n_leaves", 1))
        Dl = Dw * ls
        leaves = []
        for il in range(nlv):
            off_s = (il - 0.5 * (nlv - 1)) * (Dl + GAP)
            w = self.W(s_c + off_s, 0.0)
            yawl = self.yaw + math.pi / 2 + self.lib.flat_yaw("door_leaf")
            leaf = self.put(
                f"crouch_door_leaf_{il}", "door_leaf", w[0], w[1], max(tops), "flat", yawl, ls, section="crouch"
            )
            T = GV.world_tris(leaf["prim"])
            mins = []
            for sp_ in [p_ for p_ in self.parts if p_["role"].startswith("crouch_support")]:
                z_ = [z for z in (GV.lowest_hit(T, x, y) for x, y in self.footprint(sp_, 12, 4)) if z is not None]
                mins.append(min(z_) if z_ else None)
            if None in mins:
                raise RuntimeError("crouch: the door leaf does not reach both partitions")
            dz = min(mins) - max(tops)
            if abs(dz) > 1e-5:
                self.parts.pop()
                self.lib.used["door_leaf"] -= 1
                leaf = self.put(
                    f"crouch_door_leaf_{il}",
                    "door_leaf",
                    w[0],
                    w[1],
                    max(tops) - dz,
                    "flat",
                    yawl,
                    ls,
                    section="crouch",
                )
            leaf["rest_on"] = [p_["role"] for p_ in self.parts if p_["role"].startswith("crouch_support")]
            leaves.append(leaf)
        Dw_all = nlv * Dl + (nlv - 1) * GAP
        cw = []
        if cr.get("side_walls"):
            wfun = walls_at or (lambda s_, side: sp["lane"]["half"] + 0.3)
            ends = [max(float(self.lane_of(q)[1]) * sg_ for lv in leaves for q in lv["corners"]) for sg_ in (1, -1)]
            for sgn, endl in ((+1, ends[0]), (-1, ends[1])):
                wall = wfun(s_c, sgn)
                edge = sgn * (endl + 0.01)
                start = self.P0 + self.d * (s_c + 0.5 * cr.get("wall_depth", 0.6)) + self.n * edge
                pl = None
                for rot in (False, True):  # fronts towards the robot if possible; else the outer piece turned
                    pl = self.lib.furniture_plan(
                        wall - sgn * edge, tuple(cr["side_walls"]), 0.06, (0.88, 1.10), cr.get("wall_depth", 0.6), rot
                    )
                    if pl is not None:
                        break
                r = self.lib.furniture_line(
                    start, self.n * sgn, -self.d, wall - sgn * edge, F, tuple(cr["side_walls"]), plan=pl
                )
                for k_, rc in enumerate(r["recs"]):
                    R_, P_ = self.lib.orient_matrix(rc["kind"], "as_is", rc["yaw"], rc["scale"])
                    cc = (rc["mn"] + rc["mx"]) / 2
                    self.parts.append(
                        dict(
                            role=f"crouch_wall_{'L' if sgn > 0 else 'R'}{k_}",
                            section="crouch",
                            kind=rc["kind"],
                            src=rc["src"],
                            scale=round(rc["scale"], 4),
                            orient="as_is",
                            yaw=round(rc["yaw"], 5),
                            path=str(rc["copy"]["prim"].GetPath()),
                            prim=rc["copy"]["prim"],
                            corners=P_ + cc,
                            size=self.lib.size(rc["kind"]) * rc["scale"],
                            center=cc,
                        )
                    )
                cw.append(
                    dict(
                        side="L" if sgn > 0 else "R",
                        wall=round(wall, 3),
                        from_l=round(edge, 3),
                        pieces=r["pieces"],
                        scale=r["scale"],
                        built=r["built_len"],
                    )
                )
        info["sections"].append(
            dict(
                name="crouch",
                s=[round(s_c - 0.5 * max(alp * psc, Dw_all), 3), round(s_c + 0.5 * max(alp * psc, Dw_all), 3)],
                underside=round(max(tops) - F, 4),
                leaf=dict(
                    src=self.lib.catalog["door_leaf"]["src"],
                    scale=ls,
                    across=round(Lw * ls, 3),
                    depth=round(Dw * ls, 3),
                    thick=round(Tw * ls, 3),
                    n=nlv,
                    total_depth=round(Dw_all, 3),
                ),
                supports=dict(
                    src=self.lib.catalog[pk]["src"],
                    scale=round(psc, 4),
                    height=round(Hp * psc, 4),
                    along=round(alp * psc, 3),
                    thick=round(latp * psc, 3),
                    clear_between=round(2 * cr["l_in"], 3),
                ),
                side_walls=cw,
            )
        )
        s = s_c + 0.5 * max(alp * psc, Dw_all) + sp.get("clear_sidestep", sp["clear"])

        # ---- 6. SIDESTEP: locker cross wall with one gap at the lane centre
        ss = sp["sidestep"]
        wl, wr = walls if walls else (sp["lane"]["half"] + 0.3, sp["lane"]["half"] + 0.3)
        if walls_at is not None:
            wl, wr = walls_at(s + 0.35, 1), walls_at(s + 0.35, -1)
        lines = []
        for sgn, wall in ((+1, wl), (-1, wr)):
            gl = ss.get("gap_l", 0.0)
            edge = gl + sgn * 0.5 * ss["gap"]
            start = self.P0 + self.d * s + self.n * edge
            r = self.lib.furniture_line(
                start,
                self.n * sgn,
                -self.d,
                wall - sgn * edge,
                F,
                tuple(ss["kinds"]),
                max_embed=0.06,
                srange=tuple(ss["srange"]),
                depth_target=0.75,
            )
            for k_, rc in enumerate(r["recs"]):
                R_, P_ = self.lib.orient_matrix(rc["kind"], "as_is", rc["yaw"], rc["scale"])
                cc = (rc["mn"] + rc["mx"]) / 2
                self.parts.append(
                    dict(
                        role=f"sidestep_{'L' if sgn > 0 else 'R'}{k_}",
                        section="sidestep",
                        kind=rc["kind"],
                        src=rc["src"],
                        scale=round(rc["scale"], 4),
                        orient="as_is",
                        yaw=round(rc["yaw"], 5),
                        path=str(rc["copy"]["prim"].GetPath()),
                        prim=rc["copy"]["prim"],
                        corners=P_ + cc,
                        size=self.lib.size(rc["kind"]) * rc["scale"],
                        center=cc,
                    )
                )
            lines.append(
                dict(
                    side="L" if sgn > 0 else "R",
                    wall=round(wall, 3),
                    pieces=r["pieces"],
                    scale=r["scale"],
                    built=r["built_len"],
                )
            )
            depth = max(p_[3] for p_ in r["pieces"])
        info["sections"].append(
            dict(
                name="sidestep",
                s=[round(s, 3), round(s + depth, 3)],
                gap_design=ss["gap"],
                gap_centre_l=ss.get("gap_l", 0.0),
                lines=lines,
                what=f"real school lockers {self.lib.catalog['locker']['src']}",
            )
        )
        s_goal = s + depth + sp["goal_after"]
        self.s_goal = s_goal
        # ---- task
        path = [self.W(0.0)]
        for sec in info["sections"][1:]:
            gl_ = sec.get("gap_centre_l", 0.0)
            path.append(self.W(sec["s"][0] - 0.3, gl_))
            path.append(self.W(sec["s"][1] + 0.3, gl_))
        path.append(self.W(s_goal))
        for p in path:  # walking-surface z on the jump decks
            sl, _ = self.lane_of(p)
            if s_a0 - 1e-6 <= sl <= s_b1 + 1e-6:
                p[2] = F + deck_top
        info["task"] = dict(
            start=dict(xy=np.round(self.W(0.0)[:2], 3).tolist(), z=round(F, 3), yaw=round(self.yaw, 4)),
            goal=dict(xy=np.round(self.W(s_goal)[:2], 3).tolist(), z=round(F, 3), success_radius=0.5),
            path=[np.round(p, 3).tolist() for p in path],
            length=round(s_goal, 3),
        )
        info["n_parts"] = len(self.parts)
        self.info = info
        return info

    # ---------------------------------------------------------------- verification (PhysX rays)
    def verify(self, pxq, carb):
        from pxr import Usd, UsdGeom, UsdShade

        RA = self.RA
        res = dict(parts=len(self.parts), support=[], fail=[], max_gap=0.0)
        # (1) every part rests on something: lowest-vertex band + centroid inside the supported hull
        for p in self.parts:
            A = RA.world_points(p["prim"])
            if len(A) == 0:
                res["fail"].append(dict(part=p["role"], why="no mesh points"))
                continue
            zmin = float(A[:, 2].min())
            band = A[A[:, 2] <= zmin + 0.004]
            if p.get("rest_on"):  # underside points over the named supports
                import gym_course_v3 as GV

                T = GV.world_tris(p["prim"])
                bb = []
                for rr in p["rest_on"]:
                    q_ = next(x for x in self.parts if x["role"] == rr)
                    for x, y in self.footprint(q_, 16, 5):
                        z = GV.lowest_hit(T, x, y)
                        if z is not None:
                            bb.append((x, y, z))
                band = np.array(bb) if bb else band
            if len(band) > 600:
                band = band[np.random.default_rng(0).choice(len(band), 600, replace=False)]
            sup, gaps = [], []
            for q in band:
                h = self.ray(pxq, carb, q + np.array([0, 0, 0.015]), (0, 0, -1), 0.06, skip=p["path"])
                if h is None:
                    continue
                gap = h["d"] - 0.015
                if -0.01 <= gap <= 0.005:
                    sup.append(q[:2])
                    gaps.append(gap)
            com = A.mean(0)[:2]
            Hh = _hull(np.array(sup)) if sup else np.zeros((0, 2))
            dcom = _dist_to_hull(Hh, com) if len(sup) else 1e9
            ok = len(sup) >= 3 and dcom <= 0.02
            r = dict(
                part=p["role"],
                n_band=int(len(band)),
                n_supported=len(sup),
                max_gap=round(max(gaps), 4) if gaps else None,
                com_out_of_support_hull=round(dcom, 4) if dcom < 1e8 else None,
                z_min_above_floor=round(zmin - self.F, 4),
                ok=ok,
            )
            res["support"].append(r)
            if not ok:
                res["fail"].append(dict(why="not supported", **r))
            elif gaps:
                res["max_gap"] = max(res["max_gap"], round(max(gaps), 4))
        # (2) walking surface along the lane: what a down ray hits (overhead leaf skipped) must be a VISIBLE mesh
        leaf = [p["path"] for p in self.parts if p["section"] == "crouch" and "leaf" in p["role"]]
        course = self.root
        floor = dict(n=0, course_part=0, school_floor=0, bad=[], prims={})
        prof = []
        for l in (-0.4, 0.03, 0.4):
            for s in np.arange(-0.8, self.s_goal + 0.6, 0.05):
                o = self.W(s, l, 2.4)
                h = self.ray(pxq, carb, o, (0, 0, -1), 4.0, skip=leaf)
                floor["n"] += 1
                if h is None:
                    floor["bad"].append(dict(s=round(float(s), 2), l=l, why="no hit"))
                    continue
                z = float(h["p"][2] - self.F)
                if l == 0.03:
                    prof.append((round(float(s), 3), round(z, 4), h["c"][-50:]))
                if h["c"].startswith(course):
                    floor["course_part"] += 1
                    continue
                pr = self.stage.GetPrimAtPath(h["c"])
                if pr.IsValid() and not pr.IsA(UsdGeom.Mesh):  # collider on an Xform: its visible meshes
                    ms = [
                        q
                        for q in Usd.PrimRange(pr, Usd.TraverseInstanceProxies())
                        if q.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(q).ComputeVisibility() != UsdGeom.Tokens.invisible
                    ]
                    if ms:
                        pr = ms[0]
                vis = pr.IsValid() and UsdGeom.Imageable(pr).ComputeVisibility() != UsdGeom.Tokens.invisible
                purpose = UsdGeom.Imageable(pr).ComputePurpose() if pr.IsValid() else None
                mat = UsdShade.MaterialBindingAPI(pr).ComputeBoundMaterial()[0] if pr.IsValid() else None
                ok = (
                    pr.IsValid()
                    and pr.IsA(UsdGeom.Mesh)
                    and vis
                    and purpose in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)
                    and bool(mat)
                    and "GroundPlane" not in h["c"]
                    and "CollisionPlane" not in h["c"]
                    and abs(z) < 0.03
                )
                floor["prims"][h["c"]] = floor["prims"].get(h["c"], 0) + 1
                if ok:
                    floor["school_floor"] += 1
                else:
                    floor["bad"].append(
                        dict(
                            s=round(float(s), 2),
                            l=l,
                            z=round(z, 4),
                            hit=h["c"],
                            mesh=bool(pr.IsValid() and pr.IsA(UsdGeom.Mesh)),
                            visible=bool(vis),
                            purpose=str(purpose),
                            material=str(mat.GetPath()) if mat else None,
                        )
                    )
        floor["bad"] = floor["bad"][:40]
        floor["ok"] = not floor["bad"]
        res["floor"] = floor
        res["profile_centre"] = prof
        # (3) measured section geometry
        meas = {}
        secs = {x["name"]: x for x in self.info["sections"]}

        def top_at(s, l=0.0):
            h = self.ray(pxq, carb, self.W(s, l, 2.4), (0, 0, -1), 4.0, skip=leaf)
            return None if h is None else round(float(h["p"][2] - self.F), 4)

        hs = secs["hurdle"]["s"]
        meas["hurdle_top"] = max(
            filter(None, [top_at(x, l) for x in np.linspace(hs[0] + 0.02, hs[1] - 0.02, 7) for l in (-0.5, 0, 0.5)])
        )
        ls_ = secs["log"]["s"]
        meas["log_top"] = max(
            filter(None, [top_at(x, l) for x in np.linspace(ls_[0], ls_[1], 9) for l in (-0.5, 0, 0.5)])
        )
        jm = secs["small_jump"]
        a0, a1 = jm["deck_a"]
        b0, b1 = jm["deck_b"]
        meas["deck_a_top"] = top_at(a0 + 0.3 * (a1 - a0), 0.3)
        meas["deck_b_top"] = top_at(b0 + 0.3 * (b1 - b0), 0.3)
        pz = [(s_, top_at(s_, 0.3)) for s_ in np.arange(a1 - 0.3, b0 + 0.3, 0.02)]
        low = [s_ for s_, z_ in pz if z_ is not None and z_ < 0.05]
        meas["ditch_floor_z"] = min(z_ for _, z_ in pz) if pz else None
        meas["ditch_width_profile"] = round(max(low) - min(low) + 0.05, 3) if low else None
        cs = secs["crouch"]["s"]
        scc = 0.5 * (cs[0] + cs[1])
        ups = []
        for l in (-0.3, 0.0, 0.3):
            for ds in (-0.3, -0.15, 0.15, 0.3):
                h = self.ray(pxq, carb, self.W(scc + ds, l, 0.1), (0, 0, 1), 3.0)
                if h is not None:
                    ups.append(round(float(h["p"][2] - self.F), 4))
        meas["crouch_underside"] = dict(min=min(ups), max=max(ups), n=len(ups)) if ups else None
        wl = []
        for z in (0.3, 0.7, 1.0):
            hp = self.ray(pxq, carb, self.W(scc, 0.0, z), np.r_[self.n, 0], 3.0, skip=leaf)
            hm = self.ray(pxq, carb, self.W(scc, 0.0, z), np.r_[-self.n, 0], 3.0, skip=leaf)
            if hp and hm:
                wl.append(round(hp["d"] + hm["d"], 4))
        meas["crouch_clear_width"] = wl
        ss = secs["sidestep"]["s"]
        gw, gws = [], []
        dmin = min(
            min([p_[3] for p_ in ln["pieces"] if not p_[1]] or [p_[2] for p_ in ln["pieces"]])
            for ln in secs["sidestep"]["lines"]
        )
        for ds in np.linspace(ss[0] + 0.05, ss[0] + dmin - 0.05, 4):
            for z in (0.2, 0.6, 1.0, 1.4, 1.8):
                gl_ = self.spec["sidestep"].get("gap_l", 0.0)
                hp = self.ray(pxq, carb, self.W(ds, gl_, z), np.r_[self.n, 0], 3.0)
                hm = self.ray(pxq, carb, self.W(ds, gl_, z), np.r_[-self.n, 0], 3.0)
                if hp and hm:
                    gw.append(round(hp["d"] + hm["d"], 4))
                    gws.append((round(float(ds - ss[0]), 3), z, gw[-1], hp["c"][-30:], hm["c"][-30:]))
        meas["sidestep_gap"] = dict(min=min(gw), max=max(gw), n=len(gw), samples=gws) if gw else None
        # closure: lateral rays at the wall line must hit lockers before anything else, except at the gap
        clos = []
        for sgn in (1, -1):
            for z in (0.3, 1.0, 1.7):
                o = self.W(
                    0.5 * (ss[0] + ss[1]),
                    self.spec["sidestep"].get("gap_l", 0.0) + sgn * (0.5 * self.spec["sidestep"]["gap"] + 0.15),
                    z,
                )
                h = self.ray(pxq, carb, o, np.r_[-self.d, 0], 0.6)
                clos.append(dict(side=sgn, z=z, hit=None if h is None else h["c"][-40:]))
        meas["sidestep_closure_inside_wall"] = clos
        res["measured"] = meas
        # (4) primitive audit of everything built
        res["audit"] = RA.audit_no_primitives(self.stage, [self.root])
        res["ok"] = (not res["fail"]) and floor["ok"] and res["audit"]["ok"]
        return res

    def parts_json(self):
        out = []
        for p in self.parts:
            out.append(
                dict(
                    role=p["role"],
                    section=p["section"],
                    kind=p["kind"],
                    src=p["src"],
                    scale=p["scale"],
                    orient=p["orient"],
                    yaw_rad=p["yaw"],
                    prim=p["path"],
                    center=np.round(p["center"], 4).tolist(),
                    size=np.round(p["size"], 4).tolist(),
                    aabb_min=np.round(p["corners"].min(0), 4).tolist(),
                    aabb_max=np.round(p["corners"].max(0), 4).tolist(),
                )
            )
        return out


def jsonable(o):
    if isinstance(o, np.ndarray):
        return np.round(o, 4).tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--loc", default="south_hall")
    a = ap.parse_args()
    print(json.dumps(make_spec(a.loc), indent=1, default=jsonable))
