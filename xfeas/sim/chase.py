#!/usr/bin/env python
"""
High-quality chase-camera recorder for the demos (Isaac python).

  - renders at full resolution (default 1280x720) every 1/fps of SIMULATED time -> smooth,
    real-time-speed playback (the sim pauses while the model decides, so no stalls on screen)
  - the camera EASES after the robot (critically-damped position + heading smoothing) instead
    of being bolted to a bouncing legged base
  - frames stream to disk as quality-95 JPEGs (a 60 s run at 30 fps is ~1800 frames; holding
    them in RAM would be ~5 GB); compose steps read them back, overlay, and encode H.264
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math, os
import numpy as np


class ChaseRecorder:
    def __init__(
        self,
        stage,
        rep,
        world,
        frame_dir,
        res=(1280, 720),
        fps=30,
        phys_hz=200,
        focal=18.0,
        back=2.6,
        side=1.4,
        up=1.9,
        look_ahead=1.0,
        ease_s=0.35,
    ):
        from pxr import UsdGeom, Gf

        self.Gf, self.world, self.rep = Gf, world, rep
        self.W, self.H = res
        c = UsdGeom.Camera.Define(stage, "/World/ChaseHQ")
        c.CreateFocalLengthAttr(focal)
        c.CreateHorizontalApertureAttr(20.955)
        c.CreateVerticalApertureAttr(20.955 * self.H / self.W)
        c.CreateClippingRangeAttr(Gf.Vec2f(0.05, 120.0))
        self.xf = UsdGeom.Xformable(c.GetPrim()).MakeMatrixXform()
        rp = rep.create.render_product("/World/ChaseHQ", (self.W, self.H))
        self.ann = rep.AnnotatorRegistry.get_annotator("rgb")
        self.ann.attach([rp])
        self.dir = frame_dir
        os.makedirs(frame_dir, exist_ok=True)
        for f in os.listdir(frame_dir):
            if f.endswith(".jpg"):
                os.remove(os.path.join(frame_dir, f))
        self.every = max(1, int(round(phys_hz / fps)))
        self.dt = self.every / phys_hz
        self.alpha = 1.0 - math.exp(-self.dt / ease_s)
        self.back, self.side, self.up, self.ahead = back, side, up, look_ahead
        self.pos = self.yaw = None
        self.n, self.tags, self.k = 0, [], 0

    def _pose(self, bs):
        p, yaw = np.asarray(bs["pos"], float), float(bs["yaw"])
        if self.pos is None:
            self.pos, self.yaw = p.copy(), yaw
        else:
            self.pos += self.alpha * (p - self.pos)
            d = (yaw - self.yaw + math.pi) % (2 * math.pi) - math.pi
            self.yaw += self.alpha * d
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        # behind, raised, offset to the robot's right so the path ahead stays visible
        eye = self.pos + np.array([-self.back * c + self.side * s, -self.back * s - self.side * c, self.up])
        tgt = self.pos + np.array([self.ahead * c, self.ahead * s, 0.15])
        return eye, tgt

    def tick(self, bs, tag=0):
        """Call EVERY physics step; renders on the fps grid."""
        self.k += 1
        if self.k % self.every or bs is None:
            return
        self.grab(bs, tag)

    def grab(self, bs, tag=0):
        import cv2

        Gf = self.Gf
        eye, tgt = self._pose(bs)
        self.xf.Set(
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, tgt)), Gf.Vec3d(0, 0, 1))
            .GetInverse()
        )
        # NOT world.render(): its annotator output lags (only every ~3rd call was a new image)
        # and after the first Replicator snapshot it stops refreshing entirely -- the robot kept
        # moving while the video froze. orchestrator.step waits for a fresh frame; delta_time=0
        # so physics does not advance.
        self.rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
        im = np.asarray(self.ann.get_data())
        if im.ndim != 3 or not im.size:
            return
        cv2.imwrite(
            os.path.join(self.dir, f"{self.n:06d}.jpg"),
            np.ascontiguousarray(im[..., 2::-1]),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        self.tags.append(int(tag))
        self.n += 1


def read_frames(frame_dir):
    """RGB frames (list of arrays) in order."""
    import cv2

    fs = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    return [cv2.imread(os.path.join(frame_dir, f))[..., ::-1].copy() for f in fs]


def display_gain(frames):
    """One brightness/contrast gain per video (no flicker): stretch the middle frame's 2-98%
    luminance to 12-240. Display only -- the model's head camera is never touched."""
    mid = frames[len(frames) // 2].astype(np.float32).mean(-1)
    lo, hi = np.percentile(mid, 1), np.percentile(mid, 99.5)
    g = min((228 - 16) / max(hi - lo, 20.0), 1.6)  # capped: no blown-out walls
    return lambda im: np.clip((im.astype(np.float32) - lo) * g + 16, 0, 255).astype(np.uint8)


def panel(im, x, y, w, h, alpha=0.55):
    """Darken a rectangle so overlay text reads on any background."""
    sub = im[y : y + h, x : x + w].astype(np.float32)
    im[y : y + h, x : x + w] = (sub * (1 - alpha)).astype(np.uint8)


def text(im, s, x, y, scale=0.8, col=(255, 255, 255), th=2):
    import cv2

    cv2.putText(im, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
    cv2.putText(im, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)
