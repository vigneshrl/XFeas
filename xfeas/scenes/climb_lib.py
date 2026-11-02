"""
CLIMB study: obstacles to climb, built IN MEMORY (assets/school.usd is never written).

Obstacles are given in a LANE frame (s along the lane from P0, l lateral, z above the lane floor), like
xfeas/scenes/obstacle_lib.py. Two kinds of parts:
  * 'box'  : obstacle_lib box dict (s0, s1, l0, l1, z0, z1) -> rebuilt from REAL school assets by
             obstacle_lib.make_pool (real_assets.RealPool: plank stacks / plank terraces with the same front edges and
             top heights, furniture rows for walls); the primitive BoxPool only with XFEAS_LEGACY_PRIMITIVES=1
  * 'mat'  : a COPY of a school gym mat (hazard_copy.CopyLib, exact triangle-mesh collider), given by
             dict(kind='mat', src=<source name>, s0=<near edge along the lane>, l=<lateral centre>, layer=k)
             the long side of the mat runs ALONG the lane; layer k rests on whatever is below it: it is placed
             at the max ray-measured top under its own footprint (+1 mm), i.e. it touches the layer below at the
             highest point and never interpenetrates it.

Families (name -> (parts, s_goal, top_h_nominal)):
    mats<N>           N stacked mats (N = 1..4), a block 2.93 m long x 2.0 m wide; goal ON TOP
    matstair<N>       mat staircase: layer k starts TREAD*(k-1) further along (1, then 2, then 3 ... mats high)
    stairs<RR>        4 steps of riser RR cm, tread 0.30 m, then a landing; goal on the landing (real: plank terrace)
    ledge<HH>         one step HH cm, then a platform (real: plank stack, bottom layer set <= 2 cm into the floor)
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math
import numpy as np

MAT_FLAT = ["SM_gym_props_12", "SM_gym_props_15"]  # flat school mats, 2.93 x 2.0 x 0.082 m
MAT_ALL = [
    "SM_gym_props_12",
    "SM_gym_props_15",
    "SM_gym_props_13",
    "SM_gym_props_16",
    "SM_gym_props_14",
    "SM_gym_props_17",
]
MAT_LEN = 2.93
TREAD_MAT = 0.6


def family(name, haz, mats=None):
    """-> (parts, s_goal, nominal top height above the lane floor)"""
    mats = mats or MAT_FLAT
    if "@" in name:  # "<family>@<width m>": box obstacles narrower/wider than 2.2 m
        base, w = name.split("@")
        parts, g, top = family(base, haz, mats)
        for p in parts:
            if p["kind"] == "box":
                p["l0"], p["l1"] = -float(w) / 2, float(w) / 2
        return parts, g, top
    if name == "none":
        return [], haz + 1.6, 0.0
    if name.startswith("matstair"):
        n = int(name[8:])
        parts = [dict(kind="mat", src=mats[k % len(mats)], s0=haz + TREAD_MAT * k, l=0.0, layer=k) for k in range(n)]
        return parts, haz + TREAD_MAT * (n - 1) + 1.55, 0.082 * n
    if name.startswith("mats"):
        n = int(name[4:])
        parts = [dict(kind="mat", src=mats[k % len(mats)], s0=haz, l=0.0, layer=k) for k in range(n)]
        return parts, haz + 1.55, 0.082 * n
    if name.startswith("stairs"):
        r = int(name[6:]) / 100.0
        n = 4
        tr = 0.30
        parts = [
            dict(kind="box", s0=haz + k * tr, s1=haz + n * tr + 2.2, l0=-1.1, l1=1.1, z0=-0.05, z1=(k + 1) * r)
            for k in range(n)
        ]
        return parts, haz + n * tr + 1.3, n * r
    if name.startswith("ledge"):
        h = int(name[5:]) / 100.0
        return [dict(kind="box", s0=haz, s1=haz + 3.0, l0=-1.1, l1=1.1, z0=-0.05, z1=h)], haz + 1.6, h
    raise KeyError(name)


class ClimbBuilder:
    """Owns a BoxPool and a pool of mat copies; builds an obstacle, rays its top, reports a centre-line profile."""

    def __init__(self, stage, world, pxq, carb, n_mats=6, sources=None, root="/World/Climb"):
        import hazard_copy as HC
        import obstacle_lib as OB

        srcs = sources or MAT_ALL
        HC.SOURCES = dict(HC.SOURCES, mat=list(dict.fromkeys(list(HC.SOURCES["mat"]) + list(srcs))))  # in memory only
        self.HC, self.lib = HC, HC.CopyLib(stage)
        self.pool = OB.make_pool(stage, n=16, root=root + "/Boxes")
        self.copies = {s: [self.lib.make(f"{root}/Mats/{s}_{i}", s) for i in range(n_mats)] for s in srcs}
        self.world, self.pxq, self.carb, self.stage = world, pxq, carb, stage
        self.placed = []
        self.on_reset = None  # e.g. lambda: setattr(rob, 'ready', False) -- before any step after a reset

    def park(self):
        for lst in self.copies.values():
            for c in lst:
                self.lib.move(c, 0.0, 0.0, 0.0, -50.0 - 3 * len(lst))
        self.pool.park()

    def ray_down(self, x, y, z_hi, L=3.0, only=None):
        h = self.pxq.raycast_closest(
            self.carb.Float3(float(x), float(y), float(z_hi)), self.carb.Float3(0, 0, -1), float(L)
        )
        if h and h.get("hit"):
            who = str(h.get("collision", ""))
            if only is None or only in who:
                return z_hi - float(h["distance"]), who
        return None, None

    def _refresh(self):
        self.world.stop()
        self.world.reset()
        if self.on_reset:
            self.on_reset()
        for _ in range(2):
            self.world.step(render=False)

    def build(self, parts, P0, yaw, floor, rot=0.0, piv_s=0.0, color=(0.45, 0.36, 0.28)):
        """place parts; rot = extra yaw of the obstacle about the lane point s = piv_s (angled approach).
        Returns a log of mat placements (layer, measured support top, gap stats)."""
        self.park()
        d = np.array([math.cos(yaw), math.sin(yaw)])
        yaw_o = yaw + rot
        do = np.array([math.cos(yaw_o), math.sin(yaw_o)])
        no = np.array([-do[1], do[0]])
        P0o = np.asarray(P0, float) + d * piv_s - do * piv_s  # obstacle frame origin
        boxes = [p for p in parts if p["kind"] == "box"]
        if boxes:
            self.pool.place(boxes, P0o, yaw_o, floor, color=color)
        mats = sorted([p for p in parts if p["kind"] == "mat"], key=lambda p: p["layer"])
        used = {}
        log = []
        self._refresh()
        for p in mats:
            k = used.get(p["src"], 0)
            used[p["src"]] = k + 1
            c = self.copies[p["src"]][k]
            size = self.lib.info[p["src"]]["size"]
            src_yaw = 0.0 if size[0] >= size[1] else math.pi / 2  # long side along the lane
            L_, W_ = max(size[0], size[1]), min(size[0], size[1])
            if p.get("short"):  # short side along the lane instead
                src_yaw -= math.pi / 2
                L_, W_ = W_, L_
            cxy = P0o + do * (p["s0"] + L_ / 2) + no * p["l"]
            # support top under the footprint (inset 3 cm), from rays
            tops = []
            for u in np.linspace(-L_ / 2 + 0.03, L_ / 2 - 0.03, 25):
                for v in np.linspace(-W_ / 2 + 0.03, W_ / 2 - 0.03, 9):
                    q = cxy + do * u + no * v
                    z, _ = self.ray_down(q[0], q[1], floor + 2.0, 3.0)
                    tops.append(z if z is not None else floor)
            tops = np.array(tops)
            # the school mats are thin WAVY sheets (rigid, see survey): a real (soft) mat would sag onto the one
            # below, so the copy rests at the MEDIAN support height (static-static overlap has no physical effect;
            # the robot only touches the top surface). Signed gap: + = air under the mat, - = overlap.
            z_bot = float(np.median(tops)) + 0.001
            self.lib.move(c, cxy[0], cxy[1], yaw_o - src_yaw + p.get("yaw", 0.0), z_bot)
            gap = z_bot - tops
            log.append(
                dict(
                    src=p["src"],
                    layer=p["layer"],
                    s0=p["s0"],
                    z_bottom=round(z_bot - floor, 4),
                    x=float(cxy[0]),
                    y=float(cxy[1]),
                    yaw=float(yaw_o - src_yaw + p.get("yaw", 0.0)),
                    z_bottom_world=z_bot,
                    gap_mm_p10_50_90=[round(1000 * float(np.percentile(gap, q)), 1) for q in (10, 50, 90)],
                )
            )
            self._refresh()
        return log

    def profile(self, P0, yaw, floor, s_from, s_to, lats=(-0.25, 0.0, 0.25), ds=0.05):
        """heights above the lane floor along the lane (rays from 2 m above)"""
        d = np.array([math.cos(yaw), math.sin(yaw)])
        n = np.array([-d[1], d[0]])
        out = []
        for s in np.arange(s_from, s_to + 1e-6, ds):
            row = [round(float(s), 2)]
            for l in lats:
                q = np.asarray(P0, float) + d * s + n * l
                z, who = self.ray_down(q[0], q[1], floor + 2.0, 3.0)
                row.append(round(z - floor, 3) if z is not None else None)
            out.append(row)
        return out
