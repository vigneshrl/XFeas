#!/usr/bin/env python
"""
SCHOOL COURSE v2a (experiments/3_school_course/v2a; additive 2026-10-20). An AMO-feasible VARIANT of v2:
school_course_v2.py, its spec and its figure are NOT changed (this module subclasses SchoolCourseV2 and overrides build()
only; verify() is v2's, unchanged). Same hall, same lane, same section positions as v2. Changes from v2:

  bench        REMOVED (AMO has no step command): clear floor; the low+narrow line is at v2's s (3.0 + bench depth + 1.4)
               + 0.25 m (clear of a wall column at s 4.5 that left a slot behind the right closure).
  low+narrow   opening 0.55 m wide x 1.10 m clear: jambs = two school cloakroom partitions SM_cloakroom_2 x1.10 (1.10 m tall,
               1.98 m long, standing across the hall), the same school plank SM_literature_props_1 x0.9 seated on them as the
               lintel; the rest of the line closed to the walls with low cabinets (as v2).
  tunnel       clear height 1.10 m: side walls = one cloakroom partition SM_cloakroom_2 x1.10 per side along the lane (1.98 m),
               carrying the same 3 school door leaves as roof; the mouth closed to the walls with low school cabinets
               (closet3 / bookcase3, like v2's low+narrow closure; v2's fallen lockers left a 0.2 m gap next to the thin partitions).
  oil, sidle (0.39 m), gym-doorway block, closures, goal: as v2.
G1 / AMO: 0.487 m hand to hand (SIDESTEP_RESULTS), AMO oracle passed a 1.10 m slab (G1_OVERHANG); AMO commands vy only while
|vx| >= 0.1 (g1_amo.py), so the 0.39 m sidle needs a sidestep with forward motion.

  B = SchoolCourseV2a(stage, make_spec_v2a()); B.prepare(); world.reset(); B.build(walls_at=f); ver = B.verify(pxq, carb)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math, os, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
import school_course_v2 as SC2
import school_course as SC

ROOT_PRIM = "/World/SchoolCourseV2a"
CATALOG = SC2.CATALOG


def make_spec_v2a(**over):
    sp = SC2.make_spec_v2()
    sp.update(version="2a", hurdle=dict(kind="bench", scale=0.85, removed=True))
    sp["low_narrow"].update(gap=float(os.environ.get("SC2A_LN_GAP", 0.55)), jamb="partition", jamb_top=1.10,
                            closure=["closet3", "bookcase3"], closure_srange=[0.85, 1.15])
    sp["low_narrow"]["ds"] = float(os.environ.get("SC2A_LN_DS", 0.25))     # past the right-wall column at v2's s 4.5
    sp["tunnel"].update(style="partition", wall="partition", wall_top=1.10, closure=["closet3", "bookcase3"], closure_srange=[0.85, 1.15])
    for k, v in over.items():
        if k in sp and isinstance(sp[k], dict) and isinstance(v, dict):
            sp[k].update(v)
        else:
            sp[k] = v
    return sp


class SchoolCourseV2a(SC2.SchoolCourseV2):
    def __init__(self, stage, spec, root=ROOT_PRIM):
        super().__init__(stage, spec, root)

    def init_lib(self):
        if self.lib is None:
            self.lib = self.RA.RealLib(self.stage, root=self.root_req, catalog=CATALOG)
            self.root = self.lib.root
            for k in ("bench", "door_leaf", "closet3", "bookcase3", "locker", "plank", "oil", "partition"):
                self.lib.size(k)
        return self.lib

    def _partition_across(self, role, section, s_front, l_inner, sg, top):
        """a partition standing ACROSS the lane: front face at s_front, inner end at lateral sg*l_inner, extending outwards"""
        sc = top / float(self.lib.size("partition")[2])
        yaw = self.yaw_across("partition", "as_is")
        lat, al, H = self.lat_extent("partition", "as_is", yaw, sc)
        c = self.W(s_front + 0.5 * al, sg * (l_inner + 0.5 * lat))
        p = self.put(role, "partition", c[0], c[1], self.F, "as_is", yaw, sc, section=section)
        return p, lat, al

    def build(self, walls=None, walls_at=None):
        RA = self.RA
        self.init_lib()
        sp = self.spec; F = self.F
        wat = walls_at or (lambda s_, side: sp["lane"]["half"] + 1.5)
        info = dict(sections=[]); self.closures = []
        # ---- gym doorway block (as v2)
        gd = sp.get("gym_door_block")
        if gd:
            sa, sb = gd["x"][0] - self.P0[0], gd["x"][1] - self.P0[0]
            wl_ = min(wat(sa - 0.4, 1), wat(sb + 0.4, 1))
            r, parts = self.line("gymdoor_block_", "doorway_block", (sa, wl_ - gd["depth"] - 0.01), (1, 0), (0, -1), sb - sa, gd["kinds"],
                                 (0.95, 1.08), depth_target=gd["depth"], rotated_ok=False, max_embed=0.06)
            front = wl_ - max(p_[3] for p_ in r["pieces"]) - 0.01
            info["doorway_block"] = dict(x=gd["x"], wall_l=round(wl_, 3), front_l=round(front, 3), pieces=r["pieces"], scale=r["scale"])
            wat0 = wat
            wat = lambda s_, side, _w=wat0, _a=sa, _b=sb, _f=front: (_f if side > 0 and _a - 0.05 <= s_ <= _b + 0.05 else _w(s_, side))
        self.wat = wat
        s = sp["straight"]
        # ---- bench REMOVED: keep v2's positions (start straight + bench depth + clear)
        h = sp["hurdle"]
        yb = self.yaw_across(h["kind"], "as_is"); _, al_b, _ = self.lat_extent(h["kind"], "as_is", yb, h["scale"])
        info["sections"].append(dict(name="start_straight", s=[0.0, round(s + al_b + sp["clear"], 3)], length=round(s + al_b + sp["clear"], 3),
                                     note="v2's bench step-over removed (clear floor)"))
        s += al_b + sp["clear"] + sp["low_narrow"].get("ds", 0.0)
        info["sections"][-1]["s"][1] = round(s, 3); info["sections"][-1]["length"] = round(s, 3)
        # ---- LOW + NARROW: partition jambs + seated plank lintel
        ln = sp["low_narrow"]
        s_l = s
        inner = []
        dpt = None
        for sg in (1, -1):
            p, lat, al = self._partition_across(f"ln_jamb_{'L' if sg > 0 else 'R'}", "low_narrow", s_l, 0.5 * ln["gap"], sg, ln["jamb_top"])
            inner.append(p); dpt = al
            wall = wat(s_l + 0.2, sg)
            l0 = 0.5 * ln["gap"] + lat - 0.02
            span = wall - l0
            if span > 0.25:
                r, parts = self.line(f"ln_closure_{'L' if sg > 0 else 'R'}", "low_narrow", (s_l, sg * l0), (0, sg), (-1, 0),
                                     span, ln["closure"], ln["closure_srange"], depth_target=None, max_embed=0.06, rotated_ok=True)
        jt = max(float(q["corners"][:, 2].max()) for q in inner)
        yawp = self.yaw + math.pi / 2 + self.lib.flat_yaw(ln["lintel"])
        w = self.W(s_l + 0.5 * dpt)
        self.seat("ln_lintel", "low_narrow", ln["lintel"], w[0], w[1], jt, "flat", yawp, ln["lintel_scale"], inner)
        self.closures.append(dict(name="low_narrow", s=s_l, depth=dpt - 0.05, open=[(-0.5 * ln["gap"], 0.5 * ln["gap"])], z=(0.3, 0.8)))
        info["sections"].append(dict(name="low_narrow", s=[round(s_l, 3), round(s_l + dpt, 3)], gap_design=ln["gap"], height_design=round(jt - F, 4),
                                     jambs=dict(src=self.lib.catalog["partition"]["src"], scale=round(ln["jamb_top"] / float(self.lib.size("partition")[2]), 4)),
                                     lintel=dict(src=self.lib.catalog[ln["lintel"]]["src"], scale=ln["lintel_scale"])))
        s = s_l + dpt + sp["clear_after_ln"]
        # ---- OIL (as v2)
        tn = sp["tunnel"]; o = sp["oil"]
        R_, Po = self.lib.orient_matrix("oil", "as_is", 0.0, 1.0)
        ext = np.ptp(Po[:, :2], axis=0)
        yo = self.yaw + (math.pi / 2 if ext[0] >= ext[1] else 0.0)
        osc = o.get("scale", RA.SCALE_LIM[0])
        o_len = float(min(ext)) * osc
        c = self.W(s + 0.5 * o_len)
        s_t0 = s + o_len + o["gap_to_tunnel"]
        op = self.put("oil_patch", "oil", c[0], c[1], F, "as_is", yo, osc, section="oil")
        from pxr import UsdShade, Usd, UsdGeom
        pm = self.stage.GetPrimAtPath(f"{self.lib.HC.GEOM}/Looks/PM_Oil")
        for q in Usd.PrimRange(op["prim"]):
            if q.IsA(UsdGeom.Gprim) and pm.IsValid():
                UsdShade.MaterialBindingAPI.Apply(q).Bind(UsdShade.Material(pm), UsdShade.Tokens.strongerThanDescendants, "physics")
        info["sections"].append(dict(name="oil", s=[round(s, 3), round(s + o_len, 3)],
                                     what=f"school oil decal {self.lib.catalog['oil']['src']} x{osc:.3f} + PM_Oil",
                                     size_along_across=[round(o_len, 3), round(float(max(ext)) * osc, 3)]))
        # ---- TUNNEL: partition side walls (along the lane) + door-leaf roof + fallen-locker mouth closure
        scw = tn["wall_top"] / float(self.lib.size("partition")[2])
        yw = self.yaw_along("partition", "as_is")
        lat, al, H = self.lat_extent("partition", "as_is", yw, scw)
        rows = []
        for sg in (1, -1):
            w = self.W(s_t0 + 0.5 * al, sg * (tn["half_w"] + 0.5 * lat))
            rows.append(self.put(f"tunnel_row_{'L' if sg > 0 else 'R'}", "partition", w[0], w[1], F, "as_is", yw, scw, section="tunnel"))
        Lrow, dep = al, lat
        s_t1 = s_t0 + Lrow
        Lw, Dw, Tw = self.lib.flat_dims("door_leaf")
        nl = tn["n_leaves"]; ls = float(np.clip((Lrow - (nl - 1) * SC.GAP) / (nl * Dw), *RA.SCALE_LIM))
        yawl = self.yaw + math.pi / 2 + self.lib.flat_yaw("door_leaf")
        top = max(float(q["corners"][:, 2].max()) for q in rows)
        for il in range(nl):
            sc_ = s_t0 + 0.5 * Lrow + (il - 0.5 * (nl - 1)) * (Dw * ls + SC.GAP)
            w = self.W(sc_)
            self.seat(f"tunnel_roof_{il}", "tunnel", "door_leaf", w[0], w[1], top, "flat", yawl, ls, rows)
        l_out = tn["half_w"] + dep
        clos = []
        for sg in (1, -1):                  # low school cabinets from the partition's outer face to the wall (as v2's low+narrow closure)
            wall = wat(s_t0 + 0.3, sg)
            l0 = l_out - 0.02
            r, parts = self.line(f"tunnel_closure_{'L' if sg > 0 else 'R'}", "tunnel", (s_t0, sg * l0), (0, sg), (-1, 0),
                                 wall - l0, tn["closure"], tn["closure_srange"], depth_target=None, max_embed=0.06, rotated_ok=True)
            clos.append(dict(side="L" if sg > 0 else "R", wall=round(wall, 3), pieces=r["pieces"], scale=r["scale"], built=r["built_len"],
                             what="low school cabinets " + ", ".join(self.lib.catalog[k]["src"] for k in tn["closure"])))
        self.closures.append(dict(name="tunnel", s=s_t0, depth=0.30, open=[(-tn["half_w"], tn["half_w"])], z=(0.3, 0.8)))
        info["sections"].append(dict(name="tunnel", s=[round(s_t0, 3), round(s_t1, 3)], length=round(Lrow, 3), clear_width=round(2 * tn["half_w"], 3),
                                     roof=dict(src=self.lib.catalog["door_leaf"]["src"], n=nl, scale=round(ls, 4)),
                                     rows=dict(src=self.lib.catalog["partition"]["src"], scale=round(scw, 4), depth=round(dep, 3), top=round(top - F, 4)),
                                     closure=clos))
        s = max(s_t1 + sp["clear"], sp["sidle"]["s0_min"])
        # ---- SIDLE STRIP (as v2)
        sd = sp["sidle"]
        s_s = s
        dep2 = 0.0
        for sg in (1, -1):
            r, parts = self.line(f"sidle_row_{'L' if sg > 0 else 'R'}", "sidle", (s_s, sg * 0.5 * sd["gap"]), (1, 0), (0, -sg),
                                 sd["length"], sd["kinds"], sd["srange"], rotated_ok=False, max_embed=0.06)
            dep2 = max(dep2, max(p_[3] for p_ in r["pieces"])); Ls = r["built_len"]
        l_o2 = 0.5 * sd["gap"] + dep2
        for sg in (1, -1):
            wall = wat(s_s + 0.35, sg)
            self.line(f"sidle_closure_{'L' if sg > 0 else 'R'}", "sidle", (s_s, sg * (l_o2 + 0.005)), (0, sg), (-1, 0),
                      wall - l_o2 - 0.005, sd["kinds"], sd["srange"], depth_target=0.72, max_embed=0.10)
        self.closures.append(dict(name="sidle", s=s_s, depth=0.6, open=[(-0.5 * sd["gap"], 0.5 * sd["gap"])], z=(0.3, 0.9, 1.6)))
        info["sections"].append(dict(name="sidle_strip", s=[round(s_s, 3), round(s_s + Ls, 3)], gap_design=sd["gap"], length=round(Ls, 3),
                                     what="school wardrobes " + ", ".join(self.lib.catalog[k]['src'] for k in sd['kinds'])))
        s_goal = s_s + Ls + sp["goal_after"]
        self.s_goal = s_goal
        gx = self.W(s_goal)[0]
        if gx > SC2.D1_X0 - 1.0:
            print(f"[sc2a] WARNING goal x {gx:.2f} within 1 m of the D1 rectangle", flush=True)
        pts = [self.W(0.0)]
        for sec in info["sections"][1:]:
            pts += [self.W(sec["s"][0] - 0.3), self.W(sec["s"][1] + 0.3)]
        pts.append(self.W(s_goal))
        pts = sorted(pts, key=lambda p: float(np.dot(p[:2] - self.P0[:2], self.d)))      # ordered along the lane
        info["task"] = dict(start=dict(xy=np.round(self.W(0)[:2], 3).tolist(), z=round(F, 3), yaw=round(self.yaw, 4)),
                            goal=dict(xy=np.round(self.W(s_goal)[:2], 3).tolist(), z=round(F, 3), success_radius=0.5),
                            path=[np.round(p, 3).tolist() for p in pts], length=round(s_goal, 3),
                            skills=dict(low_narrow="crouch (+ align) through 0.55 x 1.10 m", tunnel="crouch-walk ~2 m under 1.10 m after "
                                        "a slippery oil patch (slow down)", sidle="strafe 2.7 m in a 0.39 m slot (AMO: vy only with vx >= 0.1)"))
        info["n_parts"] = len(self.parts)
        self.info = info
        return info
