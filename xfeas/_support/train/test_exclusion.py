#!/usr/bin/env python
"""
Test / showcase-scene proximity for TRAINING rows (added 2026-10-11 by the gen1 training agent).

Rule (same as scripts/gen/gen_lanes.py and scripts/v11/lane_filter.py): a school training row is dropped when its
lane / route segment comes within EXCL_M = 3 m of any test or showcase scene's start->goal segment or hazard point on
the same floor (|dz| < 1.5 m), crosses the D1 rectangle, or touches a stairs-scene path (every floor).
Test scenes = every episode_runner.SCENES id in gen_lanes.TEST_IDS or starting with S / N / HW / ST
(D1, D1g, T0, T1, H1, H2, HW1, S1-S6, G1, G2, N1, N2 ...) + stairs_scenes.STAIR_SCENES.

Why: the gen_v1 / Phase-1 generator lanes already obey this rule (data/gen/lanes.json), and so does v11, but the OLD
v7-v10c school sets were collected before most showcase scenes existed: 35/78 old overhang lanes are < 3 m from G1 / S1 /
S6 / D1 (oh0004 is 2.33 m from D1), and the v7 posture routes run through the S-scene hazards. Training on those rows
would contaminate the showcase evaluation of this model.

Row geometry
  old overhang / overhang2 / climb rows: lane = data/overhang/lanes.json id ('ohNNNN'): start + lane_len * heading
  old school / posture / objects rows: lane = {route, s0}: data/nav_baseline/routes.json polyline from s0 to s0 + 6 m
           (collectors: spawn at s0, drive --dist 3 m + 3 m look-ahead)
  generator rows (school): lane id in data/gen/lanes.json (gl* / goh*), else spec lane {start, yaw, len, floor}
  anything else (arena, course, other environments): not a school row -> never near a school test scene
Plain python (no Isaac).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, math, os, sys
import numpy as np

ROOT = xfeas.paths.ROOT
EXCL_M = 3.0
D1_RECT = (-7.5, -0.5, -7.5, 3.5)  # scripts/gen/gen_lanes.py
TEST_IDS = (
    "D1",
    "T0",
    "T1",
    "H1",
    "H2",
    "S1",
    "S2",
    "S3",
    "S4",
    "S5",
    "S6",
    "G1",
    "G2",
    "D1g",
    "N1",
    "N2",
    "HW1",
    "SP",
    "ST",
)
ROUTE_AHEAD_M = 6.0


def dist_poly(Q, P):
    Q = np.atleast_2d(np.asarray(Q, float))
    P = np.asarray(P, float)
    if len(P) == 1:
        return np.hypot(*(Q - P[0]).T)
    A, B = P[:-1], P[1:]
    AB = B - A
    L2 = np.maximum((AB**2).sum(1), 1e-9)
    t = np.clip(((Q[:, None] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
    C = A[None] + t[..., None] * AB[None]
    return np.hypot(*(Q[:, None] - C).transpose(2, 0, 1)).min(1)


class TestProximity:
    def __init__(self):
        import episode_runner as ER

        self.excl = []
        for sid, sc in ER.SCENES.items():
            if sid not in TEST_IDS and not sid.startswith(("S", "N", "HW", "ST")):
                continue
            fz = None
            try:
                G = ER.scene_geometry(sid)
                fz = float(G["floor"])
                self.excl.append((sid, np.array([G["start"][:2], G["goal"][:2]], float), fz))
            except Exception as e:  # noqa: BLE001
                print("[test_excl] geometry failed", sid, e)
            if sc.get("hazard") is not None:
                self.excl.append((sid, np.asarray([sc["hazard"]], float), fz))
        try:
            import stairs_scenes as SS

            for sid, sc in SS.STAIR_SCENES.items():
                self.excl.append((sid, np.asarray(sc["path"], float)[:, :2], None))
        except Exception as e:  # noqa: BLE001
            print("[test_excl] stairs scenes unavailable:", e)
        self.ids = sorted({e[0] for e in self.excl})
        O = json.load(open(os.path.join(ROOT, "data", "overhang", "lanes.json")))
        self.oh = {
            l["id"]: (
                np.array(
                    [
                        l["start"],
                        np.array(l["start"]) + O["lane_len"] * np.array([math.cos(l["yaw"]), math.sin(l["yaw"])]),
                    ]
                ),
                float(l["floor"]),
            )
            for l in O["lanes"]
        }
        R = json.load(open(os.path.join(ROOT, "data", "nav_baseline", "routes.json")))["routes"]
        self.routes = {}
        for r in R:
            P = np.array(r["pts"], float)
            s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
            zs = [z for z in (r["z"] if isinstance(r.get("z"), list) else [r.get("z")]) if z is not None]
            self.routes[r["id"]] = (P, s, float(np.median(zs)) if zs else None)
        G = json.load(open(os.path.join(ROOT, "data", "gen", "lanes.json")))  # generator school lanes (gl*, goh*)
        self.gen = {
            l["id"]: (
                np.array(
                    [
                        l["start"],
                        np.array(l["start"])
                        + float(l.get("len", G["lane_len"])) * np.array([math.cos(l["yaw"]), math.sin(l["yaw"])]),
                    ]
                ),
                float(l["floor"]),
            )
            for l in G["lanes"]
        }
        self._cache = {}

    def near_poly(self, P, fz):
        P = np.asarray(P, float)
        x0, x1, y0, y1 = D1_RECT
        Q = np.concatenate([np.linspace(P[i], P[i + 1], 20) for i in range(len(P) - 1)]) if len(P) > 1 else P
        if np.any((Q[:, 0] >= x0) & (Q[:, 0] <= x1) & (Q[:, 1] >= y0) & (Q[:, 1] <= y1)) and (fz is None or fz < 3.0):
            return "D1rect"
        for sid, poly, sfz in self.excl:
            if (sfz is None or fz is None or abs(sfz - fz) < 1.5) and dist_poly(Q, poly).min() < EXCL_M:
                return sid
        return None

    def row_near(self, scene, lane_json):
        """-> (test scene id | None, how). Non-school rows -> (None, 'not_school')."""
        if "school" not in scene and scene not in ("gen", "gen_school"):
            return None, "not_school"
        try:
            lane = json.loads(lane_json) if isinstance(lane_json, str) else (lane_json or {})
        except ValueError:
            lane = {}
        key = json.dumps(lane, sort_keys=True)[:400]
        if key in self._cache:
            return self._cache[key]
        res = (None, "no_geometry")
        lid = lane.get("lane")
        if isinstance(lid, str) and lid in self.gen:
            P, fz = self.gen[lid]
            res = (self.near_poly(P, fz), "gen_lane")
        elif isinstance(lid, str) and lid in self.oh:
            P, fz = self.oh[lid]
            res = (self.near_poly(P, fz), "oh_lane")
        elif lane.get("route") in self.routes and "s0" in lane:
            P, s, z = self.routes[lane["route"]]
            s0 = float(lane["s0"])
            m = (s >= s0 - 0.5) & (s <= s0 + ROUTE_AHEAD_M)
            seg = P[m] if m.sum() >= 2 else P[np.clip(np.searchsorted(s, [s0, s0 + ROUTE_AHEAD_M]), 0, len(P) - 1)]
            res = (self.near_poly(seg, z), "route")
        else:
            sp = lane.get("spec", {}).get("lane") if isinstance(lane.get("spec"), dict) else None
            ln = sp or lane
            if "start" in ln and "yaw" in ln:
                P0 = np.array(ln["start"][:2], float)
                P = np.array(
                    [P0, P0 + float(ln.get("len", 5.6)) * np.array([math.cos(ln["yaw"]), math.sin(ln["yaw"])])]
                )
                res = (self.near_poly(P, float(ln.get("floor", 0.681))), "spec_lane")
        self._cache[key] = res
        return res


if __name__ == "__main__":
    T = TestProximity()
    print("[test_excl] scenes:", T.ids)
    for lid in ("oh0004", "oh0000", "oh0005"):
        print(lid, T.row_near("school_overhang", json.dumps(dict(lane=lid))))
