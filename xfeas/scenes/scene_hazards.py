"""
Unavoidable versions of the six school scenes ("--hazards dup" in episode_runner.py): copies of the scene's own
hazard object (hazard_copy.CopyLib: exact sub-prim, full composed pose, school deletions kept, exact colliders,
the school's own oil look + PM_Oil physics) packed in two rows ACROSS the whole free width of the path, so a robot
cannot route around them and has to traverse them.

Coverage (audit 2.4): the band spans +-10 m across the path and keeps every copy whose centre is on free floor;
each copy sits on the floor height measured at its own position; oil is never placed on top of / under other
objects (floor must be bare within 3 cm). Uses the runner-side scene geometry (maps.npz) only; no method sees it.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math, os
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
SCENE_CLS = dict(S1=["oil"], S2=["oil"], S3=["lamp"], S4=["window"], S5=["door"], S6=["window", "lamp"])
ROWS = dict(S6=(-2.8, -2.0, -1.2))  # S6: in the aisle before the doorway (hazard = the doorway)
ROW_CLS = dict(S6=["window", "window", "lamp"])
X0, Y0, CELL, NX, NY = -35.6, -13.2, 0.10, 400, 380


def _floor_at(M, L, x, y, fz, tol):
    r, c = int((y - Y0) / CELL), int((x - X0) / CELL)
    if not (0 <= r < NY and 0 <= c < NX):
        return None
    t = M[f"top{L}"][r, c]
    if not np.isfinite(t) or abs(t - fz) > tol or M[f"wall{L}"][r, c]:
        return None
    return float(t)


class LiveSurface:
    """surface heights from LIVE PhysX down-rays (the 10 cm planning map misses warped gym mats). Runs physics
    briefly before any copy exists, then stops it so the copies' colliders are created on the next play."""

    def __init__(self, floor):
        import carb
        from omni.isaac.core.world import World
        from omni.physx import get_physx_scene_query_interface

        self.carb, self.floor = carb, floor
        self.w = World.instance()
        self.w.reset()
        for _ in range(3):
            self.w.step(render=False)
        self.q = get_physx_scene_query_interface()

    def z(self, x, y):
        h = self.q.raycast_closest(
            self.carb.Float3(float(x), float(y), float(self.floor + 1.2)), self.carb.Float3(0.0, 0.0, -1.0), 2.0
        )
        return float(h["position"][2]) if (h and h.get("hit")) else None

    def close(self):
        self.w.stop()


def build(stage, sid, start, goal, floor, level, route_pts, hazard_xy, seed=0):
    from pxr import UsdGeom
    import hazard_copy as HC

    lib = HC.CopyLib(stage)
    M = dict(np.load(os.path.join(ROOT, "data", "nav_baseline", "maps.npz")))
    LS = LiveSurface(floor)

    def surf(x, y, tol):  # live surface z if it is floor-ish (within tol) and not a wall cell
        r, c = int((y - Y0) / CELL), int((x - X0) / CELL)
        if not (0 <= r < NY and 0 <= c < NX) or M[f"wall{level}"][r, c]:
            return None
        z_ = LS.z(x, y)
        return z_ if (z_ is not None and abs(z_ - floor) <= tol) else None

    rng = np.random.default_rng(1234 + int(sid[1:]) + 97 * seed)
    P = np.asarray(route_pts, float)
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    ih = int(np.argmin(np.hypot(*(P - np.asarray(hazard_xy)).T)))
    UsdGeom.Xform.Define(stage, "/World/HazardCopies")
    made = []
    for row, ds in enumerate(ROWS.get(sid, (-0.4, 1.0))):
        cls = ROW_CLS.get(sid, SCENE_CLS[sid] * 3)[row]
        names = HC.SOURCES[cls]
        size = lib.info[names[0]]["size"]
        long_ax = float(max(size[0], size[1]))
        src_yaw = 0.0 if size[0] >= size[1] else math.pi / 2  # long axis direction in the source pose
        j = int(np.clip(np.searchsorted(s, s[ih] + ds), 0, len(P) - 1))
        k = min(j + 3, len(P) - 1)
        k0 = max(j - 3, 0)
        tx, ty = (P[k] - P[k0]) / max(np.hypot(*(P[k] - P[k0])), 1e-6)
        nx, ny = -ty, tx  # across the path
        cx, cy = P[j]
        # oil only on bare floor (or on the school's own oil decal, +2 cm); gym mats stay as hazard islands inside the
        # band (mats are themselves a hazard: ~45 % success in the data). S6 aisle: frames not inside furniture.
        tol = {"oil": 0.03}.get(cls, 0.05 if (cls == "window" and sid == "S6") else 0.15)
        if cls == "oil":
            # oil = 1.2 m tiles of the school's own spill decal (+ PM_Oil), each on its LOCAL surface (bare floor or the
            # top of a mat), so the spill covers mats too and never hovers; 3 tiles deep along the path
            OIL_T = 0.8
            sxy = OIL_T / long_ax
            for dl in (-OIL_T, 0.0, OIL_T):
                for lat in np.arange(-15.0, 15.0 + 1e-6, OIL_T * 0.98):
                    px, py = cx + lat * nx + dl * tx, cy + lat * ny + dl * ty
                    fz = surf(px, py, tol)
                    if fz is None:
                        continue
                    # rest on the LOWEST surface under the footprint (5x5 samples); an uneven footprint (mat edge,
                    # warped mat) is split into 4 half-size tiles, each placed only if it lies flat (<= 3 cm) -> no hover
                    for qx, qy, half in [(0.0, 0.0, OIL_T)] + [
                        (a_ * OIL_T / 4, b_ * OIL_T / 4, OIL_T / 2) for a_ in (-1, 1) for b_ in (-1, 1)
                    ]:
                        qxp, qyp = px + qx * nx + qy * tx, py + qx * ny + qy * ty
                        zs = [
                            surf(qxp + a_ * nx + b_ * tx, qyp + a_ * ny + b_ * ty, tol)
                            for a_ in np.linspace(-0.45 * half, 0.45 * half, 5)
                            for b_ in np.linspace(-0.45 * half, 0.45 * half, 5)
                        ]
                        zs = [z_ for z_ in zs if z_ is not None]
                        uneven = (not zs) or max(zs) - min(zs) > 0.012
                        if half == OIL_T and uneven:
                            continue
                        if uneven:
                            continue
                        lib.make(
                            f"/World/HazardCopies/{sid}_oil_{len(made)}",
                            names[0],
                            qxp,
                            qyp,
                            math.atan2(ny, nx),
                            min(zs) + 0.004,
                            sxy * half / OIL_T,
                        )
                        made.append((cls, names[0], round(qxp, 2), round(qyp, 2)))
                        if half == OIL_T:
                            break
            continue
        step = long_ax * {"window": 0.6}.get(cls, 0.8)
        for lat in np.arange(-15.0, 15.0 + 1e-6, step):
            px, py = cx + lat * nx, cy + lat * ny
            fz = surf(px, py, tol)
            if fz is None:
                continue
            n = names[int(rng.integers(len(names)))]
            yaw = math.atan2(ny, nx) - src_yaw + float(rng.normal(0, math.radians(8)))
            lib.make(f"/World/HazardCopies/{sid}_{cls}_{len(made)}", n, px, py, yaw, fz)
            made.append((cls, n, round(px, 2), round(py, 2)))
    LS.close()
    print(
        f"[hazards] {sid}: {len(made)} copies of {SCENE_CLS[sid]} across the path (rows {len(ROWS.get(sid, (0, 0)))}, +-15 m)",
        flush=True,
    )
    return made
