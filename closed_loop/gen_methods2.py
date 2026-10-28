"""
Closed-loop arms for an XFeas-Gen2 checkpoint (format xfeas_gen2_v1; gen2a / gen2b). gen2 agent, 2026-10-14.
Plugins for episode_runner (module:Class) and gensim run_method, same protocol as gen_methods.py (gen1).
NOT RUN by the gen2 agent (no Isaac); the commands are in analysis/GEN2_MODEL.md section 7.

  StraightXFeasGen2    straight drive to the goal. At 4 Hz (once 2 s of proprio exist) the gen2 selector
                       (select_gen2.Selector2 via its --serve sidecar) scores, in one batch: stock, CEM 'go' setpoints over
                       the robot's own box, SIDESTEP macros (skill 4: side x vy) and -- only if the checkpoint was trained on
                       them -- HEADING offsets. Choice by the LEARNED selection (learned_selection.json: margin m over the
                       stock P, learned cost weights). Frame memory: the last K-1 decision snapshots spaced >= HIST_DT s
                       (with their root poses) are sent as history; the sidecar fuses their depth into the clearance map.
  XFeasGen2FullStack   the same layer on the map-free camera planner (local_nav.LocalNav); posture follows the gen1
                       full-stack rule (obstacle band upper limit = top of the commanded body height); macros override
                       the planner while they run.

Macro execution (no privileged scene information):
  sidestep  gap geometry from the DECISION depth frame (gap_estimate.estimate: wall front x0, wall back x1, gap centre e,
            width w in the root frame; validated on 80 gen_v1 env_gap frames: |w err| 5.7 cm, |e err| 5.7 cm median),
            then the state machine of gensim_v1.SidestepCtl (the controller that produced the training rollouts):
            approach -> turn -+90 deg -> walk sideways (vy) centring on e -> turn back past x1 -> go; the lane frame is
            the root pose at the decision. Estimation failure -> stock (logged).
  detour    GEN_V2_SPEC e_ref ramp (lat_off over L_ramp, hold to the perceived hazard end + s_back, ramp back), lane law
            wz = 2 heading error - 1.5 (e - e_ref) in the frame of the root pose at the decision.
  go2       two-phase posture/speed program: A until d_commit before the perceived hazard front, then B, nominal posture
            d_level past the front (if > 0); the arm keeps its own steering (straight drive / planner).
  detour / go2 are offered only by checkpoints trained on them (gen2b with gen_v2 data); gen2a: go + sidestep.
Env: XFEAS_GEN2_CKPT, XFEAS_GEN2_SEL (learned_selection.json; default next to the ckpt), XFEAS_GEN2_HOLD (2.0 s),
     XFEAS_GEN2_HIST_DT (0.5 s), XFEAS_GEN2_MACROS (1), XFEAS_LABEL (banner label).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import io, json, math, os, struct, subprocess, sys
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
for _p in (THIS, os.path.join(ROOT, "closed_loop"), os.path.join(ROOT, "scenes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import local_nav as LN
from door_methods import straight_act
from gen_methods import gen_box, GenDecider
import gap_estimate as GE

PY = xfeas_paths.XFEAS_PY
DEF_CKPT = os.path.join(ROOT, "data/xe_train/gen2a/xfeas_gen2_ens.pt")
# FIX mode (analysis/GEN2_G1_DEBUG.md, additive 2026-10-16): XFEAS_GEN2_FIX=1 -> the selector gates macros on perceptual
# support (select_gen2 --serve inherits the env) and a stalled macro is ended and re-decided after 5 s. Unset / 0 = as released.
FIX = os.environ.get("XFEAS_GEN2_FIX", "0") not in ("", "0")
# (additive 2026-10-17, analysis/GYM_COURSE_6DEG.md) XFEAS_GEN2_GO2_REF=onset: the go2 macro switches A -> B at d_commit
# before the RISE ONSET of the lane surface (gap_estimate.hazard_onset, support level from proprioception) when that is
# nearer than hazard_extent's x0 (B -> level stays at d_level past x0, Go2MacroRef); unset = as before (x0).
# The selector / model inputs are unchanged. The support level
# is the robot's walking camera height above its feet, calibrated online from decision frames whose near field
# (0.2-1.2 m) is one level plane (line fit |slope| < 2 deg, rms < 1 cm; height at 0.7 m; median of the last 20, minus
# the commanded dh), the prior until 3 samples exist.
GO2_REF = os.environ.get("XFEAS_GEN2_GO2_REF", "")


def CM_PRIOR(kind):
    import cmap as _CM
    k = str(kind)
    if k in _CM.PRIOR_CAM_H:
        return float(_CM.PRIOR_CAM_H[k])
    for kk, v in _CM.PRIOR_CAM_H.items():
        if k.startswith(kk):
            return float(v)
    return 0.5


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Gen2Decider(GenDecider):
    """GenDecider's head camera / proprio buffer; sidecar = select_gen2.py --serve"""

    def __init__(self):
        self.ckpt = os.path.abspath(os.environ.get("XFEAS_GEN2_CKPT") or DEF_CKPT)
        self.sel = os.environ.get("XFEAS_GEN2_SEL", "")
        self.util = self.sel
        self.srv = None

    def setup(self, stage, rep, world, rob, spec, kind):
        from xfeas_decider import XFeasDecider
        self.base = XFeasDecider.__new__(XFeasDecider)
        self.base.ckpt, self.base.log, self.base.srv = self.ckpt, [], None
        sp = subprocess.Popen
        try:
            subprocess.Popen = lambda *a, **k: None
            self.base.setup(stage, rep, world, rob, spec, kind)
        finally:
            subprocess.Popen = sp
        cmd = [PY, os.path.join(THIS, "select_gen2.py"), "--ckpt", self.ckpt, "--serve"]
        if self.sel:
            cmd += ["--selection", os.path.abspath(self.sel)]
        if os.environ.get("XFEAS_GEN2_MACROS", "1") == "0":
            cmd += ["--no-macros"]
        self.srv = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self.buf, self.kind = self.base.buf, kind
        print(f"[gen2] decider ready ({os.path.basename(self.ckpt)}, selection {self.sel or 'next to ckpt'})", flush=True)

    def ask2(self, rgb, dep, pb, pj, desc, box_json, seed, hist):
        req = io.BytesIO()
        H = dict(hist_rgb=np.stack([h[0] for h in hist]) if hist else np.zeros((0, 224, 224, 3), np.uint8),
                 hist_depth=np.stack([h[1] for h in hist]).astype(np.float32) if hist else np.zeros((0, 224, 224), np.float32),
                 hist_age=np.array([h[2] for h in hist], np.float32), hist_pose=np.array([h[3] for h in hist], np.float32).reshape(-1, 4))
        np.savez(req, rgb=rgb, depth=dep, prop_base=pb, prop_joints=pj, desc=json.dumps(desc), box=box_json,
                 robot=str(self.kind), task=json.dumps(dict(seed=int(seed), setpoint={})), **H)
        b = req.getvalue()
        self.srv.stdin.write(struct.pack("<I", len(b))); self.srv.stdin.write(b); self.srv.stdin.flush()
        r = np.load(io.BytesIO(self._rd(struct.unpack("<I", self._rd(4))[0])), allow_pickle=False)
        return json.loads(str(r["choice"]))


class SidestepMacro:
    """gensim_v1.SidestepCtl state machine on a PERCEIVED gap (lane frame = root pose at the decision)"""

    def __init__(self, kind, spec, v_nom, pose0, gap, side, vy):
        self.g1 = kind.startswith("g1")
        self.vxm = 0.1 if self.g1 else 0.0
        self.d_trig = 0.6 if self.g1 else 1.1
        self.past = 0.45 if self.g1 else 0.8
        self.cap = float(spec.max_cmd[4])
        self.v_nom, self.side, self.vy = float(v_nom), float(side), float(vy)
        self.o = np.array(pose0[:2], float); self.h = float(pose0[2])
        self.s0, self.s1, self.e_gap = float(gap["x0"]), float(gap["x1"]), float(gap["e"])
        self.phase = "approach"
        self.done = False

    def lane(self, bs):
        d = np.asarray(bs["pos"][:2], float) - self.o
        c, s = math.cos(self.h), math.sin(self.h)
        return float(d[0] * c + d[1] * s), float(-d[0] * s + d[1] * c)

    def act(self, obs):
        bs = obs["bs"]; yaw = float(bs["yaw"]); h = self.h
        s, e = self.lane(bs)
        cap = self.cap
        if self.phase == "approach":
            if s >= self.s0 - self.d_trig - 0.05:
                self.phase = "turn"
            else:
                return np.array([self.v_nom, 0.0, 0.0, float(np.clip(2.0 * wrap(h - yaw) - 1.5 * (e - self.e_gap), -1.0, 1.0))])
        target = h - self.side * math.pi / 2
        if self.phase == "turn":
            err = wrap(target - yaw)
            if abs(err) < math.radians(5):
                self.phase = "side"
            else:
                return np.array([self.vxm, 0.0, 0.0, float(np.clip(1.5 * err, -cap, cap))])
        if self.phase == "side":
            if s >= self.s1 + self.past:
                self.phase = "back"
            else:
                err = wrap(target - yaw)
                want = -self.side * 1.5 * (self.e_gap - e)
                if self.vxm > 0:
                    vx = math.copysign(self.vxm, want) if abs(self.e_gap - e) > 0.03 else (self.vxm if int(obs["t"]) % 2 == 0 else -self.vxm)
                else:
                    vx = float(np.clip(want, -0.2, 0.2))
                return np.array([vx, self.side * self.vy, 0.0, float(np.clip(1.5 * err, -cap, cap))])
        if self.phase == "back":
            err = wrap(h - yaw)
            if abs(err) < math.radians(8):
                self.phase = "go"; self.done = True
            else:
                return np.array([self.vxm, 0.0, 0.0, float(np.clip(1.5 * err, -cap, cap))])
        return None


class DetourMacro:
    """GEN_V2_SPEC 3.2 detour: lane following toward e_ref(s) (ramp 0 -> lat_off over L_ramp from the decision, hold
    until s_end + s_back, ramp back), wz = 2 * heading error - 1.5 (e - e_ref); lane frame = root pose at the decision;
    s_end = the perceived hazard end (gap_estimate.hazard_extent x1)"""

    def __init__(self, pose0, v, lat_off, L_ramp, s_back, s_end):
        self.o, self.h = np.array(pose0[:2], float), float(pose0[2])
        self.v, self.lat, self.L, self.sb, self.se = float(v), float(lat_off), max(float(L_ramp), 0.2), float(s_back), float(s_end)
        self.done, self.phase = False, "out"

    def e_ref(self, s):
        r = lambda x: min(max(x, 0.0), 1.0)
        if s < self.se + self.sb:
            return self.lat * r(s / self.L), (self.lat / self.L if s < self.L else 0.0)
        return self.lat * (1 - r((s - self.se - self.sb) / self.L)), (-self.lat / self.L if s < self.se + self.sb + self.L else 0.0)

    def act(self, obs, v, wz_cap):
        bs = obs["bs"]
        d = np.asarray(bs["pos"][:2], float) - self.o
        c, s_ = math.cos(self.h), math.sin(self.h)
        s, e = float(d[0] * c + d[1] * s_), float(-d[0] * s_ + d[1] * c)
        if s >= self.se + self.sb + self.L + 0.2:
            self.done = True
            return None
        er, de = self.e_ref(s)
        self.phase = "out" if s < self.se + self.sb else "back"
        href = self.h + math.atan(de)
        wz = float(np.clip(2.0 * wrap(href - float(bs["yaw"])) - 1.5 * (e - er), -wz_cap, wz_cap))
        return np.array([self.v, 0.0, 0.0, wz])


class Go2Macro:
    """GEN_V2_SPEC 3.1 go2: setpoint A until the base is d_commit before the perceived hazard front x0, then B; nominal
    posture (speed v_B) once d_level past x0 when d_level > 0; straight to the goal; ends 1 m past the hazard end."""

    def __init__(self, pose0, A, x0, x1):
        self.o, self.h = np.array(pose0[:2], float), float(pose0[2])
        self.A, self.x0, self.x1 = np.asarray(A, float), float(x0), float(x1)
        self.done, self.phase = False, "A"

    def setpoint(self, obs):
        bs = obs["bs"]
        d = np.asarray(bs["pos"][:2], float) - self.o
        s = float(d[0] * math.cos(self.h) + d[1] * math.sin(self.h))
        A = self.A
        if s >= self.x1 + 1.0:
            self.done = True
            return None
        if self.A[9] > 0 and s >= self.x0 + A[9]:
            self.phase = "level"; return float(A[0]), 0.0, 0.0, 0.0
        if s >= self.x0 - A[8]:
            self.phase = "B"; return float(A[0]), float(A[1]), float(A[2]), float(A[3])
        self.phase = "A"; return float(A[4]), float(A[5]), float(A[6]), float(A[7])


class Go2MacroRef(Go2Macro):
    """(additive 2026-10-17, XFEAS_GEN2_GO2_REF=onset) Go2Macro with separate references: A -> B at d_commit before
    x_commit (the rise onset), B -> level at d_level past x0 (hazard_extent's front, as Go2Macro); end as Go2Macro."""

    def __init__(self, pose0, A, x_commit, x0, x1):
        super().__init__(pose0, A, x0, x1)
        self.xc = float(x_commit)

    def setpoint(self, obs):
        bs = obs["bs"]
        d = np.asarray(bs["pos"][:2], float) - self.o
        s = float(d[0] * math.cos(self.h) + d[1] * math.sin(self.h))
        A = self.A
        if s >= self.x1 + 1.0:
            self.done = True
            return None
        if self.A[9] > 0 and s >= self.x0 + A[9]:
            self.phase = "level"; return float(A[0]), 0.0, 0.0, 0.0
        if s >= self.xc - A[8]:
            self.phase = "B"; return float(A[0]), float(A[1]), float(A[2]), float(A[3])
        self.phase = "A"; return float(A[4]), float(A[5]), float(A[6]), float(A[7])


class _Gen2Layer:
    PERIOD = 0.25
    POST_EPS = 1e-3

    def _gen_setup(self, ctx):
        self.gdec = Gen2Decider()
        self.gdec.setup(stage=ctx["stage"], rep=ctx["rep"], world=ctx["world"], rob=ctx["rob"], spec=ctx["spec"], kind=ctx["kind"])
        rob, R = ctx["rob"], ctx["spec"].extra.get("setpoint_ranges", {})
        self.nom = getattr(sys.modules.get(type(rob).__module__), "NOMINAL_H", None)
        self.posture = self.nom is not None and "body_height" in R
        self.gbox = gen_box(ctx["spec"], self.nom if self.posture else None)
        # (additive 2026-10-17, ISAACLAB_HUMANOID_XFEAS) XFEAS_GEN2_VNOM = nominal (stock) speed of the box, m/s, for a
        # controller run at a matched nominal speed other than 0.6 x vx_fwd (Isaac Lab G1: 0.3 of [0.1, 1.0]); unset = as before
        if os.environ.get("XFEAS_GEN2_VNOM"):
            self.gbox["nominal"]["v"] = float(os.environ["XFEAS_GEN2_VNOM"])
        self.box_json = json.dumps(self.gbox)
        # FIX: the robot's minimum EFFECTIVE forward speed for macro phases (gate.json 'v_eff', from the training split:
        # min speed of any successful G1 rollout on rough ground = 0.31 m/s -> 0.30); 0 = no clamp (Spot / as released)
        self.v_eff = 0.0
        if FIX:
            G = json.load(open(os.path.join(ROOT, "analysis", "gen2_g1_fix", "gate.json")))
            self.v_eff = float(G.get("v_eff", {}).get("g1c" if str(ctx["kind"]).startswith("g1") else str(ctx["kind"]), 0.0))
        self.HOLD_S = float(os.environ.get("XFEAS_GEN2_HOLD", "2.0"))
        self.HIST_DT = float(os.environ.get("XFEAS_GEN2_HIST_DT", "0.5"))
        self.K_HIST = int(os.environ.get("XFEAS_GEN2_K", "4"))
        print(f"[gen2] box {self.box_json} posture {'ON' if self.posture else 'OFF'} hold {self.HOLD_S}s "
              f"history {self.K_HIST - 1} x {self.HIST_DT}s", flush=True)

    def _gen_reset(self):
        self.gdec.reset()
        self.cap, self.t_next, self.log, self.last = None, 2.0, [], ""
        self.sp_now = dict(dh=0.0, pitch=0.0, roll=0.0)
        self.hold_until, self.v_held = -1.0, None
        self.frames = []                                   # (t, rgb, depth, x, y, yaw, z) newest last
        self._camh_samples = []                            # (additive 2026-10-17) XFEAS_GEN2_GO2_REF walking cam-height calib
        self.macro = None
        rob = self.ctx["rob"]
        if self.posture and hasattr(rob, "reset_setpoints"):
            rob.reset_setpoints()

    def _history(self, t, bs):
        """older decision frames spaced >= HIST_DT, newest first, with the old root pose in the CURRENT root frame"""
        x, y, yaw, z = float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"]), float(bs["pos"][2])
        c, s = math.cos(-yaw), math.sin(-yaw)
        out, last_t = [], t
        for (tf, rgb, dep, xf, yf, yawf, zf) in reversed(self.frames):
            if last_t - tf < self.HIST_DT - 1e-3:
                continue
            dx, dy = xf - x, yf - y
            out.append((rgb, dep, t - tf, (c * dx - s * dy, s * dx + c * dy, wrap(yawf - yaw), zf - z)))
            last_t = tf
            if len(out) >= self.K_HIST - 1:
                break
        return out

    def _set_posture(self, ch, t):
        dep_post = self.posture and any(abs(float(ch.get(k) or 0.0)) > self.POST_EPS for k in ("dh", "pitch", "roll"))
        if dep_post:
            self.hold_until = t + self.HOLD_S
            for k in ("dh", "pitch", "roll"):
                tgt = float(ch[k])
                if k == "dh" and tgt < self.sp_now[k]:
                    self.sp_now[k] = tgt
                else:
                    self.sp_now[k] += 0.5 * (tgt - self.sp_now[k])
            self.cap = self.v_held = float(ch["v"]); return "new"
        if t < self.hold_until:
            self.cap = self.v_held if self.v_held is not None else self.gbox["nominal"]["v"]; return "hold"
        for k in ("dh", "pitch", "roll"):
            self.sp_now[k] += 0.5 * (0.0 - self.sp_now[k])
        self.cap = float(ch["v"]); return "stock" if ch.get("stock") else "speed"

    def _gen_tick(self, obs):
        t, bs, rob = obs["t"], obs["bs"], obs["rob"]
        g = self.gdec
        g.buf.tick(rob, bs)
        if g.desc is None:
            g.desc = rob.descriptor(self.ctx["stage"])
        if self.macro is not None or not (t >= self.t_next and g.buf.full()):
            return
        self.t_next = t + self.PERIOD
        rgb, dep = g.snapshot(bs)
        pb, pj = g.buf.arrays()
        hist = self._history(t, bs)
        if GO2_REF == "onset":                              # (additive 2026-10-17) walking camera height above the feet,
            try:                                            # from decision frames whose near field is one flat level
                import cmap as _CM
                Pn = _CM.points(dep, self.kind, stride=2)
                Nn = Pn[(Pn[:, 0] > 0.2) & (Pn[:, 0] < 1.2) & (np.abs(Pn[:, 1]) < 0.35)]
                if len(Nn) >= 30:                           # one plane: |slope| < 2 deg, rms residual < 1 cm
                    a_, b_ = np.polyfit(Nn[:, 0], Nn[:, 2], 1)
                    rr = float(np.sqrt(np.mean((Nn[:, 2] - (a_ * Nn[:, 0] + b_)) ** 2)))
                    self._camh_dbg = (round(float(np.degrees(np.arctan(a_))), 2), round(rr, 4), round(-float(a_ * 0.7 + b_), 3))
                    if t < 8.0:
                        print(f"[gen2] camh near-field t {t:.2f} x {float(bs['pos'][0]):.2f}: slope {self._camh_dbg[0]} deg rms {self._camh_dbg[1]} h {self._camh_dbg[2]}", flush=True)
                    if abs(math.degrees(math.atan(a_))) < 2.0 and rr < 0.01:
                        self._camh_samples = (getattr(self, "_camh_samples", []) + [-float(a_ * 0.7 + b_) - float(self.sp_now["dh"])])[-20:]
            except Exception as e:
                print("[gen2] cam_h calib sample failed:", repr(e), flush=True)
        ch = g.ask2(rgb, dep, pb, pj, g.desc, self.box_json, int(t * 10), hist)
        self.frames.append((t, rgb, dep, float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"]), float(bs["pos"][2])))
        self.frames = self.frames[-4 * self.K_HIST:]
        pose0 = (float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"]))
        state = None
        if ch.get("skill") == "pass_gap":
            gap = ch.get("gap") or GE.estimate(dep, self.kind)
            if gap.get("ok"):
                self.macro = ("sidestep", SidestepMacro(self.kind, self.spec, self.gbox["nominal"]["v"], pose0, gap, ch["side"], ch["vy"]))
                for k in self.sp_now:
                    self.sp_now[k] = 0.0
                state = f"sidestep(side {ch['side']:+.0f}, vy {ch['vy']}, gap e {gap['e']:+.2f} w {gap['w']:.2f} x0 {gap['x0']:.2f})"
            else:
                ch = dict(ch, skill="go", stock=True, v=self.gbox["nominal"]["v"], dh=0.0, pitch=0.0, roll=0.0)
                state = f"sidestep->stock ({gap.get('why')})"
        elif ch.get("skill") == "detour":
            A, hz = ch["act_v2"], ch["hazard"]
            self.macro = ("detour", DetourMacro(pose0, A[0], A[10], A[13], A[14], hz["x1"]))
            state = f"detour lat {A[10]:+.2f} (hazard {hz['x0']:.2f}-{hz['x1']:.2f})"
        elif ch.get("skill") == "go2":
            A, hz = ch["act_v2"], ch["hazard"]
            x_ref, ref_note = hz["x0"], ""
            if GO2_REF == "onset":                          # (additive 2026-10-17) commit reference = rise onset, not x0
                ch_s = getattr(self, "_camh_samples", [])
                base_h = float(np.median(ch_s)) if len(ch_s) >= 3 else CM_PRIOR(self.kind)
                cam_h = base_h + float(self.sp_now["dh"])
                on = GE.hazard_onset(dep, self.kind, cam_h)
                if on.get("ok"):
                    x_ref = min(float(hz["x0"]), float(on["onset"]))
                ref_note = f" ref x0 {hz['x0']:.2f} onset {on.get('onset')} -> {x_ref:.2f}"
                print(f"[gen2] go2 commit reference: hazard x0 {hz['x0']:.3f}, onset {json.dumps(on)}, cam_h {cam_h:.3f} (walking calib n {len(ch_s)}, last near-field fit {getattr(self, '_camh_dbg', None)}, prior {CM_PRIOR(self.kind):.3f}) -> {x_ref:.3f}", flush=True)
            self.macro = ("go2", Go2MacroRef(pose0, A, x_ref, hz["x0"], hz["x1"]) if GO2_REF == "onset" else Go2Macro(pose0, A, x_ref, hz["x1"]))
            state = f"go2 A({A[4]:.2f},{A[5]:+.2f},{A[6]:+.2f}) -> B({A[0]:.2f},{A[1]:+.2f},{A[2]:+.2f}) commit {A[8]:.2f} level {A[9]:.2f}{ref_note}"
        if self.macro is None:
            s2 = self._set_posture(ch, t)
            state = f"{state} {s2}" if state else s2
        bh = (self.nom + self.sp_now["dh"]) if self.posture else None
        if self.posture:
            rob.command(dict(body_height=bh, pitch=self.sp_now["pitch"], roll=self.sp_now["roll"]))
        self.log.append(dict(t=round(t, 2), x=round(pose0[0], 2), y=round(pose0[1], 2), v=None if self.cap is None else round(self.cap, 3),
                             dh=round(self.sp_now["dh"], 3), pitch=round(self.sp_now["pitch"], 3), roll=round(self.sp_now["roll"], 3),
                             skill=ch.get("skill"), side=ch.get("side"), act_v2=ch.get("act_v2"), ch_v=ch.get("v"),
                             ch_dh=ch.get("dh"), ch_pitch=ch.get("pitch"), p=ch["p_success"], p_std=ch["p_std"], p_stock=ch["p_stock"],
                             gain=ch["gain"], p_fall=ch["p_fall"], p_slip=ch["p_slip"], u=ch["utility"], u_stock=ch["u_stock"],
                             n_hist=len(hist), state=state, body_height=None if bh is None else round(bh, 3)))
        if ch.get("verbose") is not None:                  # (additive, XFEAS_GEN2_VERBOSE=1) full candidate table
            self.log[-1]["verbose"] = ch["verbose"]
        self.last = (f"XFeas-Gen2[{state}]: v {self.cap if self.cap is not None else 0:.2f} dh {self.sp_now['dh']:+.2f} "
                     f"pitch {self.sp_now['pitch']:+.2f} P={ch['p_success']:.2f} (stock {ch['p_stock']:.2f})")

    def _stalled(self, obs, kind, m):
        """ADDITIVE 2026-10-16 (G1 terrain eval), OFF by default (XFEAS_GEN2_STALL unset / 0 = gen2 as released).
        Known flaw: a go2 / detour macro only ends past the perceived hazard end, so a robot stuck inside one never asks
        the model again (same for sidestep: it ends only after the turn back past the perceived wall). With
        XFEAS_GEN2_STALL = S > 0: if the base has made < XFEAS_GEN2_STALL_DX (0.2 m) of progress along the macro's lane
        axis for S seconds, the macro is ended and the layer re-decides at once (it may pick the same macro again).
        Sidestep: only its approach / side phases are timed (the in-place turn phases make no lane progress by design;
        the timer restarts on every phase change). The model and the selection are unchanged."""
        S = float(os.environ.get("XFEAS_GEN2_STALL", "") or (5.0 if FIX else 0.0))   # FIX mode: re-decide after 5 s stall
        if S <= 0 or kind not in ("go2", "detour", "sidestep"):
            return False
        bs, t = obs["bs"], float(obs["t"])
        o = m.o; h = m.h
        d = np.asarray(bs["pos"][:2], float) - o
        s = float(d[0] * math.cos(h) + d[1] * math.sin(h))
        ph = getattr(m, "phase", "")
        ref = getattr(m, "_stall_ref", None)
        if kind == "sidestep" and ph not in ("approach", "side"):
            m._stall_ref = None; return False
        if ref is None or ref[2] != ph or s > ref[1] + float(os.environ.get("XFEAS_GEN2_STALL_DX", "0.2")):
            m._stall_ref = (t, s, ph); return False
        if t - ref[0] < S:
            return False
        self.log.append(dict(t=round(t, 2), x=round(float(bs["pos"][0]), 2), y=round(float(bs["pos"][1]), 2), v=None if self.cap is None else round(self.cap, 3),
                             dh=round(self.sp_now["dh"], 3), pitch=round(self.sp_now["pitch"], 3), roll=round(self.sp_now["roll"], 3),
                             skill=f"{kind}_stall_end", state=f"{kind} stalled {S:.0f}s (<0.2 m) -> re-decide"))
        print(f"[gen2] {kind} macro stalled at t {t:.1f} s (s {s:.2f} m in lane) -> re-decide", flush=True)
        return True

    def _macro_twist(self, obs):
        if self.macro is None:
            return None
        kind, m = self.macro
        if self._stalled(obs, kind, m):                     # opt-in (XFEAS_GEN2_STALL > 0): end a stalled macro, re-decide
            self.macro = None; self.t_next = obs["t"]
            return None
        if kind == "go2":                                   # posture program; driving stays the arm's own (straight / planner)
            sp = m.setpoint(obs)
            if sp is None:
                self.macro = None; self.t_next = obs["t"]
                return None
            if FIX and self.v_eff > 0 and sp[0] < self.v_eff:   # FIX (e): no macro phase below the robot's effective speed
                sp = (self.v_eff,) + tuple(sp[1:])
            self.cap = sp[0]
            for k, v in zip(("dh", "pitch", "roll"), sp[1:]):
                self.sp_now[k] = v
            if self.posture:
                self.ctx["rob"].command(dict(body_height=self.nom + self.sp_now["dh"], pitch=self.sp_now["pitch"], roll=self.sp_now["roll"]))
            self.last = f"XFeas-Gen2[go2:{m.phase}] v {sp[0]:.2f} dh {sp[1]:+.2f} pitch {sp[2]:+.2f}"
            return None
        tw = m.act(obs) if kind == "sidestep" else m.act(obs, max(self.gbox["nominal"]["v"], self.v_eff if FIX else 0.0), self.wz_cap)
        if tw is None or m.done:
            self.macro = None
            self.t_next = obs["t"]                          # decide again right away
            return None
        self.last = f"XFeas-Gen2[{kind}:{getattr(m, 'phase', '')}]"
        return tw


class StraightXFeasGen2(_Gen2Layer, LN.LocalNav):
    name = "xfeasgen2_straight"
    label = "+ XFeas-Gen2 layer: speed + posture + sidestep/heading macros (straight drive)"

    def setup(self, ctx):
        LN.LocalNav.setup(self, ctx)
        self._gen_setup(ctx)
        self.label = os.environ.get("XFEAS_LABEL") or f"+ XFeas-Gen2 ({os.path.basename(os.path.dirname(self.gdec.ckpt))}): learned selection (straight drive)"

    def reset(self, ep):
        LN.LocalNav.reset(self, ep)
        self._gen_reset()

    def act(self, obs):
        self._gen_tick(obs)
        tw = self._macro_twist(obs)
        if tw is not None:
            return dict(twist=tw, info=self.last, decisions=self.log)
        v0 = self.v
        if self.cap is not None:
            self.v = self.cap
        try:
            base = straight_act(self, obs)
        finally:
            self.v = v0
        return dict(twist=base["twist"], info=self.last or base["info"], decisions=self.log)


class XFeasGen2FullStack(_Gen2Layer, LN.LocalNav):
    name = "xfeasgen2_fullstack"
    label = "+ XFeas-Gen2 layer + posture-aware camera planner (full stack)"

    def setup(self, ctx):
        import xfeas_fullstack as FS
        self.FS = FS
        LN.LocalNav.setup(self, ctx)
        self._gen_setup(ctx)
        self.band_stock = tuple(self.laser.band)
        hi = ctx["spec"].extra.get("setpoint_ranges", {}).get("body_height", [FS.H_LOW, FS.H_NOM])[1]
        self.laser.band = (self.band_stock[0], max(self.band_stock[1], FS.top_of(hi) + FS.TOP_MARGIN))
        self.label = os.environ.get("XFEAS_LABEL") or self.label

    def reset(self, ep):
        LN.LocalNav.reset(self, ep)
        self._gen_reset()
        self.upper = self.FS.top_of(self.nom if self.posture else self.FS.H_NOM) + self.FS.TOP_MARGIN

    def _points(self, t, x, y):
        P = LN.LocalNav._points(self, t, x, y)
        if len(P):
            P = P[(P[:, 2] <= self.upper) | (P[:, 2] == 0.0)]
        return P

    def act(self, obs):
        n0 = len(self.log)
        self._gen_tick(obs)
        if len(self.log) > n0:
            bh = (self.nom + self.sp_now["dh"]) if self.posture else self.FS.H_NOM
            self.upper = self.FS.top_of(bh) + self.FS.TOP_MARGIN
            self.log[-1].update(band_upper=round(self.upper, 3), mode=self.mode)
        tw = self._macro_twist(obs)
        if tw is not None:
            return dict(twist=tw, info=self.last, decisions=self.log)
        base = LN.LocalNav.act(self, obs)
        if base.get("giveup"):
            return dict(base, decisions=self.log)
        tw = np.asarray(base["twist"], float).copy()
        if self.cap is not None and tw[0] > self.cap > 0:
            r = self.cap / tw[0]; tw[0] = self.cap; tw[3] *= r
        info = self.last or base["info"]
        if self.mode != "drive":
            info = f"{info}  [planner: {self.mode}]"
        return dict(twist=tw, info=info, decisions=self.log)
