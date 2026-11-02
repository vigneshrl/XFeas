"""
Lighting conditions for the school (data collection randomises them; evaluation picks one):

  original      the school's shipped lights (scene_clean)
  day           + strong sun and sky light through the windows
  dusk          shipped lights dimmed to 30-60 %, weak sun
  night_white   room lights ~off (0-6 %), sun off, robot HEADLIGHT 5500-6500 K
  night_yellow  same, headlight 2700-3300 K

The headlight is a cone SphereLight at the robot's head camera mount, pointing along the camera
(call Lighting.follow(bs, spec.cam) every control tick / before every snapshot).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import math
import numpy as np


class Lighting:
    def __init__(self, stage):
        from pxr import Usd, UsdGeom, UsdLux, Gf

        self.stage, self.Gf = stage, Gf
        self.base = []  # (attr, original intensity) of every active light
        for p in stage.Traverse():
            if p.HasAPI(UsdLux.LightAPI) or p.IsA(UsdLux.BoundableLightBase) or p.IsA(UsdLux.NonboundableLightBase):
                a = p.GetAttribute("inputs:intensity")
                if a and a.Get() is not None:
                    self.base.append((p, a, float(a.Get()), p.IsA(UsdLux.DistantLight)))
        self.sky = UsdLux.DomeLight.Define(stage, "/World/LightingSky")
        self.sky.CreateIntensityAttr(0.0)
        self.sky.CreateColorAttr(Gf.Vec3f(0.75, 0.85, 1.0))
        hl = UsdLux.SphereLight.Define(stage, "/World/RobotHeadlight")
        hl.CreateRadiusAttr(0.12)
        hl.CreateIntensityAttr(0.0)
        hl.CreateEnableColorTemperatureAttr(True)
        self.hl_temp = hl.CreateColorTemperatureAttr(6000.0)
        sh = UsdLux.ShapingAPI.Apply(hl.GetPrim())
        # calibrated by a render test (audit ALERT 11:05: the 4 cm / focus-2 light left night images black)
        sh.CreateShapingConeAngleAttr(55.0)
        sh.CreateShapingConeSoftnessAttr(0.3)
        sh.CreateShapingFocusAttr(0.0)
        from pxr import Sdf

        hl.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
        self.hl, self.hl_xf = hl, UsdGeom.Xformable(hl.GetPrim()).MakeMatrixXform()
        self.mode, self.info = "original", {}

    def set(self, mode, rng=None):
        rng = rng or np.random.default_rng()
        info = dict(mode=mode)
        room = sun = 1.0
        sky = 0.0
        head = 0.0
        if mode == "day":
            sun, sky = float(rng.uniform(1.5, 4.0)), float(rng.uniform(300, 1500))
        elif mode == "dusk":
            room, sun = float(rng.uniform(0.3, 0.6)), 0.3
        elif mode in ("night_white", "night_yellow"):
            room, sun = float(rng.uniform(0.0, 0.06)), 0.0
            head = float(rng.uniform(4e5, 1.5e6))
            t = rng.uniform(5500, 6500) if mode == "night_white" else rng.uniform(2700, 3300)
            self.hl_temp.Set(float(t))
            info.update(headlight_K=round(float(t)), headlight_I=round(head))
        for p, a, i0, is_sun in self.base:
            a.Set(i0 * (sun if is_sun else room))
        self.sky.GetIntensityAttr().Set(sky)
        self.hl.GetIntensityAttr().Set(head)
        info.update(room=round(room, 3), sun=round(sun, 2), sky=round(sky))
        self.mode, self.info = mode, info
        return info

    def follow(self, bs, cam):
        """put the headlight at the head camera mount, looking along the camera axis"""
        if self.hl.GetIntensityAttr().Get() <= 0:
            return
        Gf = self.Gf
        if HEAD_MOUNT_FN is not None:  # XFEAS_CAM_MOUNT=head (additive 2026-10-19): the real head-camera pose
            M = HEAD_MOUNT_FN(bs)  # camera->world (USD row vectors) incl. the 0.05 m offset along camera up
            if M is not None:
                self.hl_xf.Set(Gf.Matrix4d(*[float(v) for v in np.asarray(M, float).ravel()]))
                return
        fwd, up, pitch = cam
        c, s = math.cos(bs["yaw"]), math.sin(bs["yaw"])
        eye = np.asarray(bs["pos"], float) + np.array([fwd * c, fwd * s, up + 0.05])
        tgt = eye + np.array([c * math.cos(pitch), s * math.cos(pitch), math.sin(pitch)])
        self.hl_xf.Set(
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, tgt)), Gf.Vec3d(0, 0, 1))
            .GetInverse()
        )


HEAD_MOUNT_FN = (
    None  # set by headcam.HeadCamDecider (XFEAS_CAM_MOUNT=head): bs -> headlight matrix; None = spec.cam rule
)


DR_MODES = (("original", 0.3), ("day", 0.2), ("dusk", 0.2), ("night_white", 0.15), ("night_yellow", 0.15))


def sample_mode(rng):
    m, p = zip(*DR_MODES)
    return str(rng.choice(m, p=np.array(p) / sum(p)))
