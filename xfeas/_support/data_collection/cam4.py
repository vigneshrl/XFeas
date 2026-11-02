"""
gen_v4 decision camera = THE shared camera module of the v4 rebuild (xfeas/sim/camera_mount.py, Phase 0/1), used through
the harness itself: gensim.GenSim.__init__ builds `sim.mcam` (camera_mount.MountedCamera on the harness' /World/HeadCam) and
GenSim.snapshot renders with it (G1 = 27 deg PHP/Holosoma head D435i on torso_link, Spot = body-fixed front camera; full
6-DoF link pose every frame; exact instance-segmentation self mask; headlight on the mount; clearmap heuristic self mask off).
gen_v4 adds no camera code of its own: this module only checks the mount and exposes its name / per-frame info.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os

OK_MOUNTS = dict(g1=("g1_head_php",), spot=("spot_body",))


class Cam4:
    def __init__(self, sim):
        self.sim = sim
        self.robot = "g1" if sim.kind.startswith("g1") else "spot"
        mc = getattr(sim, "mcam", None)
        self.src = "shared" if mc is not None else "legacy"
        self.mount = mc.mount if mc is not None else "legacy"
        self.mask_src = "seg" if (mc is not None and mc.seg is not None) else "box"
        self.ok = self.mount in OK_MOUNTS[self.robot]
        print(f"[cam4] robot {self.robot} camera {self.src}:{self.mount} mask {self.mask_src} ok {self.ok}", flush=True)

    def require(self):
        assert self.ok, (
            f"gen_v4 needs the shared {OK_MOUNTS[self.robot]} mount, got {self.mount} (XFEAS_CAM_MOUNT={os.environ.get('XFEAS_CAM_MOUNT')})"
        )
