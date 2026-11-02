"""Recording-only TOP camera for the blocked-hallway scene HW1 (rec_cam.install via REC_CAM=halltop; registered by
sidestep_scene.build). Fixed eye straight above the block wall, just under the hallway ceiling (ceiling = floor + 3.99 m
by raycast), looking straight down; image horizontal axis = the hallway direction, so the approach, the 90 deg turn,
the sidestep through the gap and the exit are all in one frame. Never seen by any method."""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math
import numpy as np


class HallTop:
    def __init__(self, sc):
        import sidestep_scene as NS

        self.sc = sc
        d0, d1 = NS.geometry(sc)["s_wall"]
        self.sm = 0.5 * (d0 + d1)
        self.fl = float(sc["floor"])
        self.H = float(sc.get("cam_h", 3.70))
        self.focal = float(sc.get("cam_focal", 8.0))
        self.ox, self.oy = NS.lane_to_world(sc, self.sm, sc["gap_e"])
        self.X0 = self.ox

    def pose(self, bs=None):
        h = self.sc["heading"]
        eye = np.array([self.ox, self.oy, self.fl + self.H])
        tgt = np.array(
            [self.ox - 0.01 * math.sin(h), self.oy + 0.01 * math.cos(h), self.fl]
        )  # tilt ACROSS -> image x = hallway
        return eye, tgt

    def lane_origin(self, eye):
        return np.array([eye[0], eye[1], self.fl + 1.0])

    def samples(self):
        e, t = self.pose()
        return [(0.0, e, t)]
