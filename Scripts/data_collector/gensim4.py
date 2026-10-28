"""
gen_v4 harness (import AFTER SimulationApp exists). analysis/GEN4_SPEC.md.

V4Mixin (MRO: V4Mixin, gensim3.V3Mixin, gensim_v2.V2Mixin, GenSimV1 | EnvSim) adds, on top of gen_v3:
  * the shared decision camera (cam4.Cam4): every snapshot of the collector (approach memory, decision, exposure loop)
    goes through it; per frame the camera->world matrix (cam_hist, already per frame) + self-mask fraction + mount are kept
  * contact logging (PhysX contact reports on every robot rigid body): per rollout, body (non-foot) contact time and
    impulse with OUR hazard objects and with the rest of the scene -> "passed clean" vs "passed by contact" labels
  * the mode-aware fall rule (xembody/fall_judge.py semantics, generalised to posture-commanded controllers): a floor hit
    is always a fall; the tilt / min-height rule applies only while the controller is commanded an UPRIGHT posture and
    >= RECOVER_S after the last low (deep crouch / big lean) command. The legacy rule is logged next to it (fell_legacy).
  * generic foot links (G1 ankle_roll_link, Spot *_foot) for foot_dz / slip
  * the 'seq' family (course-like fenced-lane sequences built from real school assets; seq_course.py)
GenSimV4 = school; make_envsim_v4() = Isaac environments / terrain.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import math, os, re, sys, time
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
for _p in (THIS, os.path.join(ROOT, "Scripts", "data_collector"), os.path.join(ROOT, "Scripts", "data_collector"), os.path.join(ROOT, "closed_loop"),
           os.path.join(ROOT, "scenes"), os.path.join(ROOT, "closed_loop")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gensim as GS
import gensim_v1 as GV
import gensim_v2 as G2
import gensim3 as G3
import cam4 as C4

RECOVER_S = 3.0
TOP_MIN, PELVIS_MIN, UPSIDE = 0.12, 0.10, -0.7
FOOT_RE = re.compile(r"(ankle|foot|toe|sole)", re.I)
CONTACT_BIN = 0.02          # s
IMP_MIN = 1e-3              # N s per contact pair and physics step; below = no touch (zero-impulse proximity reports)

# ---------------------------------------------------------------------- per-frame camera info of the shared camera
# GenSim.snapshot (Phase 0/1) renders with the shared camera_mount and leaves its info in self.snap_info; this wrapper only
# remembers that info per camera matrix so the decision can attach mask fraction / mount per history frame. Process-local.
_ORIG_SNAPSHOT = GS.GenSim.snapshot


def _snapshot_v4(self, bs):
    out = _ORIG_SNAPSHOT(self, bs)
    if out is None or not getattr(self, "v4", False):
        return out
    info = dict(getattr(self, "snap_info", None) or {})
    log = self.__dict__.setdefault("_snap_info", {})
    if len(log) > 400:
        log.clear()
    log[np.asarray(out[2], float).tobytes()] = info
    return out


GS.GenSim.snapshot = _snapshot_v4


class ContactLog4:
    """robot <-> world contacts from PhysX contact reports. Active only between begin() and end() (one rollout)."""

    def __init__(self, sim):
        from pxr import Usd, UsdPhysics, PhysxSchema
        from omni.physx import get_physx_simulation_interface
        self.sim = sim; self.root = sim.root
        n = 0
        for q in Usd.PrimRange(sim.stage.GetPrimAtPath(sim.root), Usd.TraverseInstanceProxies()):
            if q.HasAPI(UsdPhysics.RigidBodyAPI) and not q.IsInstanceProxy():
                api = PhysxSchema.PhysxContactReportAPI.Apply(q)
                api.CreateThresholdAttr().Set(0.0)
                n += 1
        self.n_bodies = n
        self.active = False
        self._reset()
        self.sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on)
        print(f"[v4] contact reports on {n} robot bodies", flush=True)

    def _reset(self):
        self.A = dict(haz_body=set(), haz_foot=set(), scene_body=set(), haz_body_imp=0.0, scene_body_imp=0.0, haz_foot_imp=0.0,
                      links={}, t_first=None, n_ev=0)

    def begin(self):
        self._reset(); self.active = True; self.t0 = float(getattr(self.sim.rob, "_v_t", 0.0))

    def end(self):
        self.active = False
        A = self.A
        top = dict(sorted(A["links"].items(), key=lambda kv: -kv[1])[:5])
        return dict(haz_body_s=round(len(A["haz_body"]) * CONTACT_BIN, 3), haz_body_Ns=round(A["haz_body_imp"], 3),
                    haz_foot_s=round(len(A["haz_foot"]) * CONTACT_BIN, 3), haz_foot_Ns=round(A["haz_foot_imp"], 3),
                    scene_body_s=round(len(A["scene_body"]) * CONTACT_BIN, 3), scene_body_Ns=round(A["scene_body_imp"], 3),
                    haz_links={k: round(v, 3) for k, v in top.items()}, t_first_haz_body=A["t_first"], n_events=A["n_ev"])

    def _on(self, headers, data):
        if not self.active:
            return
        from pxr import PhysicsSchemaTools
        t = float(getattr(self.sim.rob, "_v_t", 0.0)) - self.t0
        b = int(t / CONTACT_BIN)
        A = self.A
        for h in headers:
            ra = str(PhysicsSchemaTools.intToSdfPath(h.actor0)); rb = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            if ra.startswith(self.root) and not rb.startswith(self.root):
                link, other = ra.rsplit("/", 1)[-1], str(PhysicsSchemaTools.intToSdfPath(h.collider1))
            elif rb.startswith(self.root) and not ra.startswith(self.root):
                link, other = rb.rsplit("/", 1)[-1], str(PhysicsSchemaTools.intToSdfPath(h.collider0))
            else:
                continue
            imp = 0.0
            for k in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                im = data[k].impulse; imp += math.sqrt(im[0] ** 2 + im[1] ** 2 + im[2] ** 2)
            A["n_ev"] += 1
            if imp < IMP_MIN:                    # proximity / speculative pairs (contact offset) carry no impulse: not a touch
                continue
            haz = other.startswith(GS.OURS)
            foot = bool(FOOT_RE.search(link))
            if haz and not foot:
                A["haz_body"].add(b); A["haz_body_imp"] += imp; A["links"][link] = A["links"].get(link, 0.0) + imp
                if A["t_first"] is None:
                    A["t_first"] = round(t, 2)
            elif haz:
                A["haz_foot"].add(b); A["haz_foot_imp"] += imp
            elif not foot:
                A["scene_body"].add(b); A["scene_body_imp"] += imp


class FallJudge4:
    """mode-aware fall rule for any controller of the collector (G1 posture / mode controllers, Spot)."""

    def __init__(self, sim):
        self.sim = sim
        inner = sim.rob._v_inner
        self.inner = inner
        self._orig = inner.fallen               # bound method of the adapter (the legacy rule)
        self.t_low_end = -1e9
        self.reset()

    def reset(self):
        """per rollout (the wrapper clock is restored to the decision time, so the low-mode memory must restart too)"""
        self.reason = None; self.legacy_fired = False; self.n_low_tilt = 0
        self.t_low_end = -1e9

    def _low_now(self):
        """commanded posture is LOW: base command height in the lowest 40 % of its declared range, or pitch > 0.5 rad.
        Humanoids only: a low / leaning G1 posture tilts the pelvis by design; a quadruped's posture range (Spot: pitch
        +-0.35, height 0.25-0.65) never does, so Spot keeps the plain rule."""
        if not self.sim.kind.startswith("g1"):
            return False
        s = self.sim.base_spec
        sent = getattr(self.sim.rob, "_v_last_sent", None) or {}
        import ctrl_spec as CS
        low = False
        h = CS.cmd(s, "pelvis_height"); p = CS.cmd(s, "torso_pitch")
        if h is not None and h["name"] in sent:
            lo, hi = h["range"]; nom = h["nominal"]
            low |= float(sent[h["name"]]) < lo + 0.4 * (nom - lo)
        if p is not None and p["name"] in sent:
            low |= float(sent[p["name"]]) > 0.5
        return low

    def fallen(self, bs=None):
        inner = self.inner
        bs = bs or inner.base_state()
        if bs is None:
            self.reason = "no_state"; return True
        if not self._orig(bs):
            if self._low_now():
                self.t_low_end = float(getattr(self.sim.rob, "_v_t", 0.0))
            return False
        self.legacy_fired = True
        floor = float(getattr(inner, "_floor_z", 0.0))
        pel = float(bs["pos"][2]) - floor
        top = None
        try:
            top = float(self.sim._top_now()) - floor
        except Exception:
            pass
        up = None
        try:
            T = self.sim._link_T(); i = self.sim._i_torso
            q = T[i, 3:7]
            x, y, _z, w_ = [float(v) for v in q]
            up = 1.0 - 2.0 * (x * x + y * y)            # (R[2, 2] of the torso quaternion xyzw): torso z-axis . world z
        except Exception:
            pass
        if (top is not None and top < TOP_MIN) or pel < PELVIS_MIN or (up is not None and up < UPSIDE):
            self.reason = f"floor_hit(top {None if top is None else round(top, 3)}, pelvis {pel:.3f}, up {None if up is None else round(up, 2)})"
            return True
        t = float(getattr(self.sim.rob, "_v_t", 0.0))
        if self._low_now():
            self.t_low_end = t; self.n_low_tilt += 1
            return False
        if 0.0 <= t - self.t_low_end < RECOVER_S:
            self.n_low_tilt += 1
            return False
        self.reason = f"tilt(tilt {float(bs['tilt']):.2f}, pelvis {pel:.3f})"
        return True


class V4Mixin:
    v4 = True

    def setup_v4(self, base_spec):
        self.setup_v3(base_spec)
        self.cam4 = C4.Cam4(self)
        self.cam4.require()
        self.contacts = ContactLog4(self)
        self._restart()                                  # contact report API takes effect after a physics restart
        self.judge = FallJudge4(self)
        self.rob._v_inner.fallen = self.judge.fallen     # instance attribute: every drive / approach / probe uses it
        self._snap_info = {}
        self.seq = None

    # ------------------------------------------------------------------ generic feet (G1 ankle_roll_link, Spot *_foot)
    def _links_init(self):
        G3.V3Mixin._links_init(self)
        self._i_feet = [i for i, n in enumerate(self._links["names"]) if re.search(r"(ankle_roll_link|_foot)$", n)]

    def _foot_dz(self):
        try:
            if self._links is None:
                self._links_init()
            if len(self._i_feet) < 2:
                return np.nan
            T = self._link_T(); z = T[self._i_feet, 2]
            return float(z.max() - z.min())
        except Exception:
            return np.nan

    # ------------------------------------------------------------------ build: + course-like sequences
    def build(self, sit):
        if self.seq is not None:
            self.seq.clear()
        if sit["family"] == "seq":
            import seq_course as SQ
            if self.seq is None:
                self.seq = SQ.SeqScene(self)
            return self.seq.build(sit)
        return super().build(sit)

    def park_scene(self):
        if self.seq is not None:
            self.seq.clear()
        return G3.V3Mixin.park_scene(self)

    # ------------------------------------------------------------------ ray checks from a camera INSIDE the robot's head
    # The shared head mount puts the eye inside the G1 head collider (and close to Spot's body): the gen_v1/v2 checks
    # (visible: first hit must be our hazard; floor_render_check: <= 4 robot skips of 1 cm) then see only the robot. Same
    # rules, but every robot collider is skipped (up to 40 hops, 2 cm past each robot hit).
    def _ray_nr(self, o, v, L):
        o = np.asarray(o, float); v = np.asarray(v, float); acc = 0.0
        for _ in range(40):
            dist, who, pos = self.ray(o + v * acc, v, L - acc)
            if dist is None:
                return None, "", None
            if who.startswith(self.root):
                acc += dist + 0.02
                if acc >= L:
                    return None, "", None
                continue
            return acc + dist, who, pos
        return None, "", None

    def visible(self, M, dep, step=8):
        eye = M[3, :3]
        n_h, n_ok, n_all = 0, 0, 0
        for v in range(step // 2, 224, step):
            for u in range(step // 2, 224, step):
                dc = np.array([(u + 0.5 - 112) / GS.FX, -(v + 0.5 - 112) / GS.FX, -1.0, 0.0])
                dw = (dc @ M)[:3]; nrm = np.linalg.norm(dw); dw = dw / nrm
                dist, who, _ = self._ray_nr(eye, dw, 12.0)
                n_all += 1
                if dist is None or not who.startswith(GS.OURS) or dep[v, u] >= 59.0:     # 59+: a masked robot pixel
                    continue
                n_h += 1
                z_ray = dist / nrm
                if abs(float(dep[v, u]) - z_ray) < 0.05 + 0.03 * z_ray:
                    n_ok += 1
        return dict(n_haz=n_h, n_agree=n_ok, n_grid=n_all)

    def floor_render_check(self, M, dep, step=12):
        eye = M[3, :3]
        n_hit, n_ok, n_all = 0, 0, 0
        for v in range(112 + step // 2, 224, step):
            for u in range(step // 2, 224, step):
                dc = np.array([(u + 0.5 - 112) / GS.FX, -(v + 0.5 - 112) / GS.FX, -1.0, 0.0])
                dw = (dc @ M)[:3]; nrm = np.linalg.norm(dw); dw = dw / nrm
                dist, who, _ = self._ray_nr(eye, dw, 10.0)
                n_all += 1
                if dist is None:
                    continue
                if dep[v, u] >= 59.0:                     # a robot pixel (masked by the shared camera): not a floor sample
                    n_all -= 1; continue
                n_hit += 1
                z_ray = dist / nrm
                n_ok += int(abs(float(dep[v, u]) - z_ray) < 0.05 + 0.03 * z_ray)
        return dict(n_all=n_all, n_hit=n_hit, n_ok=n_ok, ok=bool(n_all > 0 and n_hit >= 0.5 * n_all and n_ok >= 0.8 * n_hit))

    # ------------------------------------------------------------------ decision: + per-frame camera info
    def decision(self, sit, geo, light):
        self._snap_info = {}
        dec = super().decision(sit, geo, light)
        cams = dec["mem"]["cam_hist"]
        infos = [self._snap_info.get(np.asarray(M, float).tobytes(), {}) for M in cams]
        dec["mem"]["selfmask_hist"] = np.array([float(i.get("mask_frac", np.nan)) for i in infos], np.float32)
        # robot root (x, y, yaw) in the world per history frame (pose_hist is decision-relative), for feas_mem per-frame bases
        b = dec["mem"]["base_pose_dec"]; ph = dec["mem"]["pose_hist"]
        c0, s0 = math.cos(float(b[3])), math.sin(float(b[3]))
        dec["mem"]["base_hist_world"] = np.array([[b[0] + c0 * p[0] - s0 * p[1], b[1] + s0 * p[0] + c0 * p[1], b[3] + p[3]] for p in ph], np.float64)
        dec["mem"]["cam_h_hist"] = np.array([float(i.get("cam_h") if i.get("cam_h") is not None else np.nan) for i in infos], np.float32)
        dec["cam_info"] = dict(source=self.cam4.src, mount=self.cam4.mount, mask_src=[i.get("mask_src") for i in infos],
                               n_frames_with_info=int(sum(1 for i in infos if i)))
        return dec

    # ------------------------------------------------------------------ rollout: + contacts + fall rule details
    def rollout_v3(self, geo, dec, prog, act, vid=None):
        self.judge.reset()
        self.contacts.begin()
        try:
            out = G3.V3Mixin.rollout_v3(self, geo, dec, prog, act, vid=vid)
        finally:
            ct = self.contacts.end()
        out["contact"] = ct
        out["fall_reason"] = self.judge.reason if out.get("fell") else None
        out["fell_legacy"] = bool(self.judge.legacy_fired)
        out["n_low_tilt_exempt"] = int(self.judge.n_low_tilt)
        clean = ct["haz_body_s"] < 0.05 and ct["haz_body_Ns"] < 1.0
        out["pass_label"] = ("clean" if clean else "contact") if out["success"] else "fail"
        return out


class GenSimV4(V4Mixin, G3.V3Mixin, G2.V2Mixin, GV.GenSimV1):
    pass


def make_envsim_v4():
    import envsim as ES

    class EnvSimV4(V4Mixin, G3.V3Mixin, G2.V2Mixin, ES.EnvSim):
        pass
    return EnvSimV4
