"""
Closed-loop gen3 arms for episode_runner / run_terrain (gen3 model agent, 2026-10-18). GEN3_SPEC v0.2; analysis/GEN3_MODEL.md.

  StraightXFeasGen3   straight drive to the goal (D1g, HW1, T0, terrain tiles)
  StairsXFeasGen3     the stairs path follower of the gym course (GC0 / GC2), same layer

Per episode, NO learning and NO fitting on the controller under test:
  * spec   = XFEAS_GEN3_SPEC (default scripts/gen3/specs/g1s.json, written from the release docs);
  * probe  = XFEAS_GEN3_PROBE: the controller's own <= 30 s self-test JSON (probe_sonic_gen3.py; once per session);
  * the robot adapter is wrapped in the collector's own virtual.VRob proxy (identity wrapper) so that commands use the
    spec's names and h(t) is recorded with exactly the training channels (spec 6; top / foot_dz from the link extents of
    gensim3.V3Mixin);
  * every PERIOD (0.25 s) while no program runs and 2 s of proprio exist: decision frame -> feas (clearance map from the
    decision depth -> scripts/gen3/feas.feas_vector with the probe's fp / caps) -> select_gen3 --serve sidecar (gen3
    candidates from the probe, driver programs, gates, learned selection; XFEAS_GEN3_RULE=1 = the rule baseline;
    XFEAS_GEN3_AGREE=<desc_only ckpt> = hybrid-agree) -> the chosen driver program is executed by the collector's own
    executors (gensim3.ProgramCtl3 / TurnSideCtl3) in a lane frame at the decision, hazard front / end PERCEIVED
    (gap_estimate.hazard_extent / estimate on the decision depth; feas x positions as fallback);
  * stock = the spec's nominal command (SONIC slowWalk, default speed) at the spec nominal speed 0.3 m/s;
  * a constant program is held until the perceived hazard end + 0.5 m (or 2 s without a hazard), then re-decided; a
    program with < 0.2 m progress in 5 s is ended and re-decided (gen2 F4).
Frame memory: the decision frame only (no camera->world matrices in the runner); trained with frame-memory dropout 0.3.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import io, json, math, os, sys, time
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
for _p in (THIS, os.path.join(ROOT, "Scripts", "data_collector"), os.path.join(ROOT, "Scripts", "data_collector"), os.path.join(ROOT, "closed_loop"),
           os.path.join(ROOT, "scenes"), os.path.join(ROOT, "scenes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import local_nav as LN
from door_methods import straight_act
import gap_estimate as GE
import gen3_closed_loop as GCL
import cmap as CMAP

PERIOD = 0.25


def SG_act(mac, env_act):
    import select_gen3 as SG
    return SG.act_dict(mac, env_act)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class _Lane:
    """shim for gensim3 executors: rob (VRob), driver, lane_coords"""

    def __init__(self, rob, driver):
        self.rob, self.driver = rob, driver

    @staticmethod
    def lane_coords(bs, geo):
        p = np.asarray(bs["pos"][:2], float) - geo["P0"]
        return float(p @ geo["d"]), float(p @ geo["n"])


def lane_geo(P0, yaw, s_front=None, s_end=None):
    d = np.array([math.cos(yaw), math.sin(yaw)]); n = np.array([-d[1], d[0]])
    g = dict(P0=np.asarray(P0, float)[:2], yaw=float(yaw), d=d, n=n)
    if s_front is not None:
        g.update(s_front=float(s_front), s_end=float(s_end))
    return g


class _Gen3Layer:
    def _g3_setup(self, ctx):
        import ctrl_spec as CS
        import driver as DR
        import virtual as VC
        import gensim3 as G3
        from xfeas_decider import XFeasDecider
        import subprocess
        self.CS, self.VC = CS, VC
        self.cspec = CS.load(os.environ.get("XFEAS_GEN3_SPEC") or os.path.join(ROOT, "Scripts", "model", "specs", "g1s.json"))
        pp = os.environ["XFEAS_GEN3_PROBE"]
        self.probe = json.load(open(pp)); self.probe_file = os.path.abspath(pp)
        self.driver = DR.Driver(self.cspec, self.probe)
        self.v_nom = float(self.cspec.get("nominal", {}).get("speed", 0.3))
        self.rule = os.environ.get("XFEAS_GEN3_RULE", "0") == "1"
        self.ckpt = os.environ.get("XFEAS_GEN3_CKPT") or os.path.join(ROOT, "data", "xe_train", "gen3", "xfeas_gen3_ens.pt")
        self.agree = os.environ.get("XFEAS_GEN3_AGREE", "")
        # perception fix (2026-10-18, flag-gated; defaults = the v0.3 single-frame behaviour):
        #   XFEAS_GEN3_PERC=v04       feas from the ego-motion-compensated keyframe memory with feas rules v0.4 (feas_mem.py)
        #   XFEAS_GEN3_MODEL_HIST=1   the model also gets the 4-frame obs memory of training (t = -2.25/-1.5/-0.75/0 s)
        #   XFEAS_GEN3_DUMP=<dir>     save every decision frame (rgb, depth, camera->world, pose, both feas) for diagnostics
        self.perc = os.environ.get("XFEAS_GEN3_PERC", "v03")
        self.model_hist = os.environ.get("XFEAS_GEN3_MODEL_HIST", "0") == "1"
        self.dump = os.environ.get("XFEAS_GEN3_DUMP", "")
        self.mem_prog = os.environ.get("XFEAS_GEN3_MEM_PROG", "0") == "1"      # memory keeps updating during programs
        import feas_mem as FM
        self.FM = FM
        self.fmem = FM.FrameMemory()
        # head camera + proprio buffer of the gen1/gen2 deciders (identical camera / buffer), no v8 sidecar
        # XFEAS_CAM_MOUNT=head (additive 2026-10-19, headcam.py): same camera prim / intrinsics, posed from the G1 head D435
        # v4 (2026-10-19): XFeasDecider itself uses the shared isaac_sim/camera_mount.py (G1 default = the 27 deg
        # PHP/Holosoma head D435i; XFEAS_CAM_MOUNT=head = Unitree D435; =legacy|pelvis = the old spec.cam rule)
        sys.path.insert(0, os.path.join(ROOT, "isaac_sim"))
        import camera_mount as CMT
        self.CMT = CMT
        self.cam_mount = "pelvis" if CMT.is_legacy(ctx["kind"]) else CMT.mount_name(ctx["kind"])
        _cls = XFeasDecider
        base = _cls.__new__(_cls); base.ckpt, base.log, base.srv = self.ckpt, [], None
        base._root_hint = ctx.get("robot_root")
        sp = subprocess.Popen
        try:
            subprocess.Popen = lambda *a, **k: None
            base.setup(ctx["stage"], ctx["rep"], ctx["world"], ctx["rob"], ctx["spec"], ctx["kind"])
        finally:
            subprocess.Popen = sp
        self.cam = base
        self.client = GCL.Gen3Client(self.ckpt, agree_ckpt=self.agree)
        inner = ctx["rob"]

        class _Shim(G3.V3Mixin):                       # link extents (top, foot_dz) exactly as in collection
            pass
        sh = _Shim.__new__(_Shim)
        sh.stage, sh.root, sh._links = ctx["stage"], ctx.get("robot_root"), None
        self.vrob = VC.VRob(inner, self.cspec)
        sh.rob = self.vrob
        object.__setattr__(self.vrob, "_v_top_fn", sh._top_now)
        object.__setattr__(self.vrob, "_v_foot_fn", sh._foot_dz)
        self.shim = sh
        self.jn = None
        # XFEAS_GEN3_TRIGGER=dist (additive 2026-10-19, gen3_trigger.py): posture / sidestep programs wait for a live
        # memory-map distance trigger instead of starting at the decision. Unset = unchanged.
        self.trig = os.environ.get("XFEAS_GEN3_TRIGGER", "") == "dist"
        self.safe_stand = False
        if self.trig:
            import gen3_trigger as TRG
            self.TRG = TRG
            self.settle = TRG.probe_settle_times(self.probe)
            # 2026-10-21: after a posture program, stand up stopped (vx 0, nominal posture, until upright) and only where the
            # free width >= the probe's standing width (SONIC fell standing up from a crawl inside the 0.40 m funnel throat)
            self.safe_stand = os.environ.get("XFEAS_GEN3_SAFE_STAND", "1") == "1"
            self.t_settle_max = max([v for k, v in self.settle.items() if k != "_by_top"] or [1.0])
            self.stand_top = float(self.probe["fp"]["stand_top"]); self.stand_depth = float(self.probe["fp"]["stand_depth"])
            self.top_clear = float(os.environ.get("XFEAS_GEN3_TOP_CLEAR") or 0.03)
            self.commit_n = int(os.environ.get("XFEAS_GEN3_COMMIT_N", "3"))            # v4 fix (b); 1 = old behaviour
            self.commit_pdrop = float(os.environ.get("XFEAS_GEN3_COMMIT_PDROP", "0.4"))
            print(f"[gen3] v4 execution fixes: commit/hysteresis N {self.commit_n} (P drop {self.commit_pdrop}); robust end "
                  f"{int(TRG.END_ROBUST)} (median of {TRG.END_K})", flush=True)
            print(f"[gen3] distance trigger ON: probe settle times {self.settle} (max {self.t_settle_max:.2f} s); stand top "
                  f"{self.stand_top:.3f} depth {self.stand_depth:.3f}", flush=True)
        arm = "rule" if self.rule else ("gen3 hybrid-agree" if self.agree else "gen3 full")
        self.label = os.environ.get("XFEAS_LABEL") or f"+ {arm} (zero-shot: spec + probe + h(t))"
        print(f"[gen3] perception {self.perc}; model frame history {int(self.model_hist)}; memory during programs {int(self.mem_prog)}; "
              f"driver honours probe {os.environ.get('XFEAS_GEN3_DRV_PROBE', '0')}; top clearance {os.environ.get('XFEAS_GEN3_TOP_CLEAR', '-')}; "
              f"fall rule {os.environ.get('EP_FALL_RULE', 'default')}; dump {self.dump or '-'}", flush=True)
        print(f"[gen3] arm {arm}; ckpt {self.ckpt}; agree {self.agree or '-'}; probe {pp}; spec {self.cspec['ctrl_id']}; v_nom {self.v_nom}", flush=True)

    def _g3_reset(self, ep):
        self.cam.buf.reset()
        self.floor = float(ep.get("floor_z0", 0.0))
        self.prog = None; self.prog_info = None
        self.standup, self.z_stand = None, None          # XFEAS_GEN3_SAFE_STAND (2026-10-21)
        self.t_next, self.log, self.last = 2.0, [], ""
        self.geo0 = None
        self.hnext = 0.0
        self.vrob.reset_setpoints()
        self.vrob._v_hist.clear()
        self.v_cap = self.v_nom
        self.stall = GCL.StallWatch()
        self.fmem.reset(); self.ring = []; self.floor_w = None
        self.t_mem_next, self.n_mem_prog = 0.0, 0
        self.ep_id = f"{ep.get('scene', 'sc')}_t{ep.get('trial', 0)}"
        self.pending, self._memP, self.x_head_stand = None, None, None
        self.trig_log = []

    # ------------------------------------------------------------------ perception -> feas
    def _feas(self, dep):
        import clearmap as CLM
        import feas as FE
        P = CMAP.points(dep, "g1", stride=1)
        zf, used_prior = CMAP.floor_level(P, "g1")
        Pb = np.stack([P[:, 0], P[:, 1], P[:, 2] - zf], 1)
        cm = CLM.grid(Pb, 0.0)
        fv, fd = FE.feas_vector(cm, self.probe["fp"], self.probe["caps"], P=Pb)
        self._zf = (zf, used_prior)
        return fv, fd

    @staticmethod
    def _cam_matrix(bs, cam):
        """camera->world (USD row-vector) of XFeasDecider._snapshot (SetLookAt(eye, tgt, z-up).GetInverse())"""
        fwd, up, pitch = cam
        c, s = math.cos(bs["yaw"]), math.sin(bs["yaw"])
        eye = np.asarray(bs["pos"], float) + np.array([fwd * c, fwd * s, up])
        f = np.array([c * math.cos(pitch), s * math.cos(pitch), math.sin(pitch)])
        r = np.cross(f, [0.0, 0.0, 1.0]); r /= np.linalg.norm(r); u = np.cross(r, f)
        M = np.eye(4); M[0, :3], M[1, :3], M[2, :3], M[3, :3] = r, u, -f, eye
        return M

    def _cam_M(self, bs):
        """camera->world of the frame just rendered by self.cam._snapshot (pelvis: the spec.cam rule; head: the D435 pose)"""
        if self.cam_mount != "pelvis":
            return self.cam.last["M"]
        return self._cam_matrix(bs, self.cam.spec.cam)

    def _self_x(self):
        """self-body handling of the memory: pelvis camera = feas_mem's body box (SELF_X, as before); head mounts = the exact
        link-geometry pixel mask of headcam (applied at render), so the box (which also deletes real structure within
        0.6 m, e.g. a beam the head camera only sees when close) is off"""
        return self.FM.SELF_X if self.cam_mount == "pelvis" else 0.0

    def _trig_dump(self, rec):
        p = os.environ.get("XFEAS_GEN3_TRIG_LOG", "")
        if p:
            open(p, "a").write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")

    def _base_of(self, bs):
        return (float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])) if self.cam_mount != "pelvis" else None

    def _feas_mem(self, t, bs, dep, M):
        """v0.4: add the frame to the keyframe memory, fuse, feas with rules v0.4. Floor (world z) = the newest frame's own
        near-floor estimate when it saw the floor (not clamped to the prior), else the last such estimate."""
        import clearmap as CLM
        zf, used_prior = self._zf
        if not used_prior or self.floor_w is None:
            self.floor_w = float(M[3, 2]) + float(zf)
        self.fmem.add(t, CLM.min_pool_depth(np.clip(dep, 0, 20.0), 128), M, base=self._base_of(bs))
        x, y, yaw = float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])
        P = self.FM.mem_points(self.fmem.frames(), (x, y, yaw), self.floor_w, self_x=self._self_x())
        self._memP = (P, (x, y, yaw))
        fv, fd, _ = self.FM.feas_from_points(P, self.probe["fp"], self.probe["caps"], rules="v04")
        return fv, fd

    def _mem_update(self, t, bs):
        """XFEAS_GEN3_MEM_PROG=1: while a program runs, keep adding head-camera frames to the memory every PERIOD
        (same snapshot / floor estimate as a decision; no decision is made)"""
        import clearmap as CLM
        try:
            rgb, dep = self.cam._snapshot(bs)
            P = CMAP.points(dep, "g1", stride=2)
            zf, used_prior = CMAP.floor_level(P, "g1")
            M = self._cam_M(bs)
            if not used_prior:
                self.floor_w = float(M[3, 2]) + float(zf)
            self.fmem.add(t, CLM.min_pool_depth(np.clip(dep, 0, 20.0), 128), M, base=self._base_of(bs))
            self.ring.append((t, np.asarray(rgb, np.uint8), np.asarray(dep, np.float32), M))
            self.ring = [f for f in self.ring if t - f[0] <= 3.0]
            self.n_mem_prog += 1
            if self.trig and self.prog_info is not None and self.prog_info.get("trig"):
                pose = (float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"]))
                P = self.FM.mem_points(self.fmem.frames(), pose, self.floor_w, self_x=self._self_x())
                self.prog_info["trig"]["trk"].update(P, pose, self.stand_top, self.top_clear)
                if self.safe_stand and self.prog_info["trig"]["kind"] in ("over", "lane"):
                    ge = self.TRG.gap_entrance(P, self.probe["fp"]["stand_width"], x_lo=-0.5, x_hi=1.0)
                    self.prog_info["narrow_here"] = bool(ge.get("ok") and ge["x0"] <= 0.3 and ge["x1"] >= -0.3)
                self._trig_dump(dict(t=round(t, 2), x=round(pose[0], 3), event="track update", edges=self.prog_info["trig"]["trk"].last,
                                     d=self.prog_info["trig"]["trk"].dist(bs["pos"], self.prog_info["trig"]["heading"])))
        except Exception as e:
            print("[gen3] memory update during program failed:", repr(e), flush=True)

    def _hist_obs(self, t, rgb, dep, M):
        """the 4-frame obs memory of collection (gensim_v2: t = -2.25 / -1.5 / -0.75 / 0 s, 128 px, camera->world)"""
        import cv2
        import clearmap as CLM
        self.ring.append((t, rgb, dep, M)); self.ring = [f for f in self.ring if t - f[0] <= 3.0]
        sel = []
        for th in (-2.25, -1.5, -0.75):
            best = min(self.ring[:-1], key=lambda f: abs(f[0] - (t + th)), default=None)
            if best is not None and abs(best[0] - (t + th)) <= 0.3 and all(best is not b for b in sel):
                sel.append(best)
        sel.append(self.ring[-1])
        return dict(rgb_hist=np.stack([cv2.resize(f[1], (128, 128), interpolation=cv2.INTER_AREA) for f in sel]).astype(np.uint8),
                    depth_hist=np.stack([CLM.min_pool_depth(np.clip(f[2], 0, 20.0), 128) for f in sel]).astype(np.float16),
                    t_hist=np.array([f[0] - t for f in sel], np.float32), cam_hist=np.stack([f[3] for f in sel]).astype(np.float64),
                    K128=CLM.intrinsics(128))

    # ------------------------------------------------------------------ tick
    def _g3_tick(self, obs):
        t, bs, rob = float(obs["t"]), obs["bs"], obs["rob"]
        self.cam.buf.tick(rob, bs)
        if self.geo0 is None:
            g = np.asarray(obs["goal_xy"], float); p = np.asarray(bs["pos"][:2], float)
            self.geo0 = lane_geo(p, math.atan2(g[1] - p[1], g[0] - p[0]))
        object.__setattr__(self.vrob, "_v_t", t)
        if t >= self.hnext:                              # h(t) at 10 Hz (VRob._record; the runner steps the inner adapter)
            self.hnext = t + 0.1 - 1e-9
            try:
                self.vrob._record()
            except Exception as e:
                print("[gen3] history record failed:", repr(e), flush=True)
        if self.prog is not None and self.mem_prog and self.perc == "v04" and t >= self.t_mem_next:
            self.t_mem_next = t + PERIOD
            self._mem_update(t, bs)
        if self.standup is not None:                    # safe stand-up after a posture program: stop, nominal posture, wait upright
            su = self.standup
            up = float(bs.get("tilt", 0.0)) < 0.15 and (self.z_stand is None or float(bs["pos"][2]) >= self.z_stand - 0.03)
            if (up and t - su["t0"] >= 0.5) or t - su["t0"] > 5.0:
                ev = dict(t=round(t, 2), x=round(float(bs["pos"][0]), 3), event="stand-up done", upright=bool(up), dt=round(t - su["t0"], 2))
                self.trig_log.append(ev); self.log.append(dict(ev, state="STAND-UP done")); print(f"[gen3] {ev}", flush=True)
                self.standup = None; self.t_next = t
                self.vrob.set_env_intent(self.v_nom, 9.9, 9.9, 0.0)
            else:
                tw = np.zeros(4)
                self.vrob.command(dict(self.driver.nominal_posture)); self.vrob.set_env_intent(0.0, 9.9, 9.9, 0.0)
                self.vrob.command(tw)
                self.last = "gen3[stand-up]"
                return tw
        if self.prog is not None:
            tw = self._run_prog(obs)
            if tw is not None:
                self.vrob.command(np.asarray(tw, float))
                return tw
        if t >= self.t_next and self.cam.buf.full():
            self.t_next = t + PERIOD
            self._decide(obs)
            if self.prog is not None:
                tw = self._run_prog(obs)
                if tw is not None:
                    self.vrob.command(np.asarray(tw, float))
                    return tw
        if self.trig:
            if self.x_head_stand is None and t >= 1.5 and self.prog is None and self.pending is None:
                self.x_head_stand = self._head_x(bs); self.z_stand = float(bs["pos"][2])
                print(f"[gen3] standing head (camera eye) x ahead of the pelvis: {self.x_head_stand:.3f} m", flush=True)
            if self.prog is None and self.pending is not None and self._pending_fire(obs):
                tw = self._run_prog(obs)
                if tw is not None:
                    self.vrob.command(np.asarray(tw, float))
                    return tw
        return None

    # ------------------------------------------------------------------ distance trigger (XFEAS_GEN3_TRIGGER=dist)
    def _head_x(self, bs):
        """head-camera eye (Unitree d435 frame on torso_link) ahead of the pelvis along the base yaw, from physics"""
        import headcam as HC
        from scipy.spatial.transform import Rotation
        if getattr(self.cam, "mcam", None) is not None:      # v4: the ACTIVE mount's eye (was: always the Unitree d435)
            return self.cam.mcam.eye_ahead(bs)
        sh = self.shim
        if sh._links is None:
            sh._links_init()
        T = sh._link_T()
        R = Rotation.from_quat(T[sh._i_torso, 3:7]).as_matrix()
        eye = HC.cam_matrix_from_torso(R, T[sh._i_torso, :3], "head")[3, :3]
        c, s_ = math.cos(float(bs["yaw"])), math.sin(float(bs["yaw"]))
        return float((eye[0] - float(bs["pos"][0])) * c + (eye[1] - float(bs["pos"][1])) * s_)

    def _trig_kind(self, mac, prog, A):
        """'over' / 'lane' (posture programs), 'gap' (turn_side_turn) or None (starts at the decision as before)"""
        if mac == "turn_side_turn":
            return "gap"
        if mac in ("constant", "two_phase"):
            B = (prog.get("phases", {}).get("B") or {}).get("posture") or {}
            nom = self.driver.nominal_posture
            if any(abs(float(B.get(k, v)) - float(v)) > 1e-6 for k, v in nom.items() if isinstance(v, (int, float))):
                return "over" if float(A.get("top_b", 9.9)) < self.stand_top - 0.03 else "lane"
        return None

    def _t_reach(self, prog, A=None):
        B = (prog.get("phases", {}).get("B") or {}).get("posture") or {}
        key = (round(float(B.get("body_height", 0.75)), 2), round(float(B.get("pitch", 0.0)), 2))
        if key in self.settle:
            return float(self.settle[key]), key
        # no probe segment with these exact commands (e.g. a modal controller: the posture is a mode): use the probe segment
        # whose settled top height is closest to the program's target top (probe-measured, no constant)
        bt = self.settle.get("_by_top") or []
        top_t = float((A or {}).get("top_b", 9.9))
        if bt and top_t < 9.0:
            fin, ts = min(bt, key=lambda z: abs(z[0] - top_t))
            return float(ts), ("top", round(fin, 3))
        return float(self.t_settle_max), key

    def _op_depth(self, prog):
        oid = (prog.get("phases", {}).get("B") or {}).get("op_id")
        for o in self.probe["ops"]:
            if o.get("op_id") == oid:
                return float(o.get("depth", self.stand_depth))
        return self.stand_depth

    def _pending_fire(self, obs):
        t, bs = float(obs["t"]), obs["bs"]
        Pd = self.pending
        f, e = Pd["trk"].dist(bs["pos"], Pd["heading"])
        if f is None:
            return False
        v = max(float(np.asarray(bs["v_b"], float)[0]), 0.0)
        if Pd["kind"] == "gap":
            d_plan = float(Pd["prog"].get("d_commit", 0.6)) + 0.05
        else:
            d_plan = Pd["t_reach"] * v + Pd["x_front"] + self.TRG.POS_MARGIN
        if f > d_plan:
            return False
        import gensim3 as G3
        x, y = float(bs["pos"][0]), float(bs["pos"][1])
        prog, A, mac = dict(Pd["prog"]), Pd["A"], Pd["macro"]
        if Pd["kind"] in ("over", "lane"):
            prog["d_level"] = 0.0                 # stand-up = the memory rule (_run_prog), not s_front + d_level
        geo = lane_geo((x, y), Pd["heading"], f, max(e, f + 0.1))
        sim = _Lane(self.vrob, self.driver)
        ex = (G3.TurnSideCtl3 if mac == "turn_side_turn" else G3.ProgramCtl3)(sim, geo, dict(s_dec=0.0), prog, A)
        if mac == "two_phase":
            ex.phase = "B"                       # the trigger IS the commit point
        s_done = max(e, f + 0.1) + (float(prog.get("s_back", 0.3)) + float(prog.get("L_ramp", 1.0)) if mac == "detour" else 0.0) + 0.5
        ev = dict(t=round(t, 2), x=round(x, 3), y=round(y, 3), event="trigger", macro=mac, name=Pd["name"], kind=Pd["kind"],
                  d_plan=round(d_plan, 3), d_live_front=round(f, 3), d_live_end=round(e, 3), v_meas=round(v, 3),
                  t_reach=round(Pd["t_reach"], 2), x_front=round(Pd["x_front"], 3), posture_key=Pd.get("pkey"),
                  waited_s=round(t - Pd["t0"], 2), n_track_updates=Pd["trk"].n_upd)
        self.trig_log.append(ev); self._trig_dump(ev)
        self.log.append(dict(ev, state=f"TRIGGER {mac} [{Pd['name']}] at d {f:.2f} (plan {d_plan:.2f}, {Pd['kind']})"))
        print(f"[gen3] trigger {ev}", flush=True)
        self.prog = ex
        self.prog_info = dict(macro=mac, geo=geo, t0=t, s_done=s_done, hazard=True,
                              trig=dict(kind=Pd["kind"], trk=Pd["trk"], heading=Pd["heading"], name=Pd["name"]))
        self.stall.reset()
        self.pending = None
        return True

    def _commit_hold(self, t, x, ch, mac, prog):
        """v4 execution fix (b) (2026-10-19, analysis/REBUILD_V4_PHASE01.md; XFEAS_GEN3_COMMIT_N, default 3; 1 = the old rule:
        any contrary decision cancels at once). A waiting (armed) posture / sidestep program is CANCELLED only after
        COMMIT_N CONSECUTIVE contrary decisions (stock, or a program that starts at once), or at once when the pending
        program's own P in this decision's candidate table dropped by >= XFEAS_GEN3_COMMIT_PDROP (default 0.4) below its P
        when it was (last) chosen. A decision for the same program resets the count; a decision for ANOTHER triggered
        program replaces it at once (unchanged). In gen3_headcam B' the 4 Hz model alternated side (P 0.6-0.86) / stock (P 0.01-0.17), so every
        stock decision disarmed the sidestep and it fired only when a 'side' decision happened to land in the trigger window.
        -> None (decision applies normally) or a dict (keep the pending program armed)."""
        if self.commit_n <= 1:
            return None
        Pd = self.pending
        if not ch["stock"] and self._trig_kind(mac, prog, SG_act(mac, ch["env_act"])) is not None:
            # same program: refresh; ANOTHER triggered program: replace at once (as before) -- a later, closer decision
            # refining the posture (B' t 9.0 s: h 0.60 -> h 0.30 p 1.57 for the 1.10 m beam) must not be held back
            # (validation run 1 of this fix held it and fired the h 0.60 crouch into the beam)
            return None
        Pd["n_contra"] = int(Pd.get("n_contra", 0)) + 1
        p_now = next((float(r["p"]) for r in ch.get("table", []) if r.get("name") == Pd.get("name")), None)
        drop = p_now is not None and p_now <= float(Pd.get("p_ref", Pd.get("p", 1.0))) - self.commit_pdrop
        rec = dict(event="commit hold", t=round(t, 2), x=round(x, 3), name=Pd.get("name"), n_contra=Pd["n_contra"], need=self.commit_n,
                   p_pending_now=None if p_now is None else round(p_now, 3), p_ref=round(float(Pd.get("p_ref", 0.0)), 3),
                   contrary="stock" if ch["stock"] else ch.get("name"))
        if Pd["n_contra"] >= self.commit_n or drop:
            rec.update(event="commit released (" + ("P drop" if drop else f"{Pd['n_contra']} contrary") + ")")
            self.trig_log.append(rec); self._trig_dump(rec)
            return None
        self.trig_log.append(rec); self._trig_dump(rec)
        return rec

    def _decide(self, obs):
        t, bs, rob = float(obs["t"]), obs["bs"], obs["rob"]
        rgb, dep = self.cam._snapshot(bs)
        pb, pj = self.cam.buf.arrays()
        if self.jn is None:
            names = list(rob._art.dof_names)
            self.jn = names if len(names) == pj.shape[1] else None
            print(f"[gen3] proprio joints {pj.shape[1]}, articulation dofs {len(names)} -> names {'ok' if self.jn else 'MISMATCH (proprio masked)'}", flush=True)
        H = self.vrob.hist_arrays(geo=self.geo0, floor_z=self.floor, t_ref=t)
        M = self._cam_M(bs)
        fv03, fd03 = self._feas(dep)
        fv, fd = fv03, fd03
        if self.perc == "v04":
            fv, fd = self._feas_mem(t, bs, dep, M)
        hist_obs = self._hist_obs(t, np.asarray(rgb, np.uint8), np.asarray(dep, np.float32), M)
        hz = GE.hazard_extent(dep, self.kind_cam)
        gap = GE.estimate(dep, self.kind_cam)
        lat_room = (float(hz.get("room_left", 1.0)), float(hz.get("room_right", 1.0))) if hz.get("ok") else (1.0, 1.0)
        req = dict(rgb=np.asarray(rgb, np.uint8), depth=np.asarray(dep, np.float32), prop_base=pb.astype(np.float32),
                   prop_joints=(pj if self.jn else np.zeros_like(pj)).astype(np.float32), fmt="gen_v3",
                   joint_names=json.dumps(self.jn or [f"j{i}" for i in range(pj.shape[1])]),
                   hist_t=H["t"], hist_valid=H["valid"], hist_cmd=H["cmd"], hist_env=H["env"], hist_ach=H["ach"], hist_q=H["q"], hist_dq=H["dq"],
                   ach_keys=json.dumps(list(self.VC.ACH_KEYS)), feas=np.asarray(fv, np.float32))
        if self.model_hist and len(hist_obs["t_hist"]) > 1:
            req.update(hist_obs)
        if self.cam_mount != "pelvis":                   # the sidecar's clearance map back-projects with the exact matrix
            req.update(cam_mount_M=np.asarray(M, np.float64), cam_mount_base=np.asarray(self._base_of(bs), np.float64))
            if self.cam.last.get("cam_h") is not None:
                req.update(cam_mount_camh=np.float64(self.cam.last["cam_h"]))
        if self.dump and os.environ.get("XFEAS_GEN3_DUMPREQ", "0") == "1":     # additive 2026-10-19: full request for offline replay
            os.makedirs(self.dump, exist_ok=True)
            np.savez_compressed(os.path.join(self.dump, f"req_{self.ep_id}_{t:07.2f}.npz"), lat_room=np.array(lat_room, np.float32),
                                gap_e=np.float32(float(gap.get("e", 0.0)) if gap.get("ok") else 0.0), base=np.array([float(bs["pos"][0]), float(bs["pos"][1]), float(bs["pos"][2]), float(bs["yaw"])]),
                                **{k: (np.asarray(v) if not isinstance(v, str) else np.array(v)) for k, v in req.items()})
        t0 = time.time()
        ch = self.client.ask(req, self.cspec, self.probe, lat_room, float(gap.get("e", 0.0)) if gap.get("ok") else 0.0, rule=self.rule)
        dt_ms = (time.time() - t0) * 1e3
        mac, prog = ch["macro"], ch["program"]
        x, y, yaw = float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])
        posture = None
        trig_rec = None
        hold = self._commit_hold(t, x, ch, mac, prog) if (self.trig and self.pending is not None) else None
        if hold is not None:
            # v4 fix (b): the waiting program stays armed (walking nominal, as while waiting); tracker keeps updating
            if self._memP is not None:
                self.pending["trk"].update(self._memP[0], self._memP[1], self.stand_top, self.top_clear)
            self.prog = None
            self.vrob.command(dict(self.driver.nominal_posture)); self.vrob.set_env_intent(self.v_nom, 9.9, 9.9, 0.0)
            self.v_cap = self.v_nom
            f_, e_ = self.pending["trk"].dist(bs["pos"], self.pending["heading"])
            trig_rec = dict(hold, kind=self.pending["kind"], d_front=None if f_ is None else round(f_, 3), d_end=None if e_ is None else round(e_, 3))
            state = (f"WAIT(commit {hold['n_contra']}/{self.commit_n}) {self.pending['macro']} [{self.pending['name']}] vs {('stock' if ch['stock'] else ch.get('name'))}")
        elif ch["stock"]:
            self.prog = None
            self.vrob.command(dict(self.driver.nominal_posture)); self.vrob.set_env_intent(self.v_nom, 9.9, 9.9, 0.0)
            self.v_cap = self.v_nom
            state = "stock"
            if self.trig and self.pending is not None:
                trig_rec = dict(event="pending cancelled (stock decision)", name=self.pending["name"])
                self.trig_log.append(dict(trig_rec, t=round(t, 2), x=round(x, 3)))
                self.pending = None
        elif self.trig and self._trig_kind(mac, prog, SG_act(mac, ch["env_act"])) is not None:
            # XFEAS_GEN3_TRIGGER=dist: wait (nominal walking, re-deciding every PERIOD) for the live distance trigger
            A = SG_act(mac, ch["env_act"])
            kind = self._trig_kind(mac, prog, A)
            Bp = (prog.get("phases", {}).get("B") or {}).get("posture") or {}
            key = (mac, kind, json.dumps(Bp, sort_keys=True), prog.get("side"))
            heading = self.geo0["yaw"] if abs(wrap(yaw - self.geo0["yaw"])) < 0.6 else yaw
            if self.pending is None or self.pending["key"] != key:
                if self.pending is not None:
                    self.trig_log.append(dict(event="pending replaced", t=round(t, 2), x=round(x, 3), old=self.pending["name"], new=ch.get("name")))
                tr, pkey = self._t_reach(prog, A)
                od = self._op_depth(prog)
                self.pending = dict(key=key, t0=t, trk=self.TRG.Track(kind, self.probe["fp"]["stand_width"]), kind=kind, heading=heading, t_reach=tr, pkey=pkey,
                                    x_front=max(od - 0.5 * self.stand_depth, 0.5 * self.stand_depth))
                self.trig_log.append(dict(event="pending", t=round(t, 2), x=round(x, 3), name=ch.get("name"), kind=kind,
                                          t_reach=round(tr, 2), op_depth=round(od, 3), x_front=round(self.pending["x_front"], 3)))
            self.pending.update(prog=prog, A=A, macro=mac, name=ch.get("name"), p=ch["p_success"], n_contra=0, p_ref=float(ch["p_success"]))
            if self._memP is not None:
                self.pending["trk"].update(self._memP[0], self._memP[1], self.stand_top, self.top_clear)
            self.prog = None
            self.vrob.command(dict(self.driver.nominal_posture)); self.vrob.set_env_intent(self.v_nom, 9.9, 9.9, 0.0)
            self.v_cap = self.v_nom
            f_, e_ = self.pending["trk"].dist(bs["pos"], heading)
            trig_rec = dict(kind=kind, d_front=None if f_ is None else round(f_, 3), d_end=None if e_ is None else round(e_, 3),
                            edges=self.pending["trk"].last)
            ph = prog.get("phases", {})
            posture = {k: v.get("posture") for k, v in ph.items()}
            state = (f"WAIT {mac} [{ch.get('name')}] {kind} front " + ("unseen" if f_ is None else f"{f_:.2f}") +
                     (f" end {e_:.2f}" if e_ is not None else ""))
        else:
            import gensim3 as G3
            import select_gen3 as SG
            A = SG.act_dict(mac, ch["env_act"])
            # perceived hazard extent along the lane (decision frame x == lane s from the decision pose, lane yaw = heading)
            xs = [v for v in (fd.get("x_at_h_free"), fd.get("x_at_gap")) if v is not None and np.isfinite(v)]
            if mac == "turn_side_turn" and gap.get("ok"):
                s0, s1 = float(gap["x0"]), float(gap.get("x1", gap["x0"] + 0.3))
            elif hz.get("ok"):
                s0, s1 = float(hz["x0"]), float(hz["x1"])
            elif xs:
                s0 = min(xs); s1 = s0 + 0.6
            else:
                s0, s1 = 0.6, 1.2
            geo = lane_geo((x, y), self.geo0["yaw"] if abs(wrap(yaw - self.geo0["yaw"])) < 0.6 else yaw, s0, s1)
            sim = _Lane(self.vrob, self.driver)
            ex = (G3.TurnSideCtl3 if mac == "turn_side_turn" else G3.ProgramCtl3)(sim, geo, dict(s_dec=0.0), prog, A)
            s_done = s1 + (float(prog.get("s_back", 0.3)) + float(prog.get("L_ramp", 1.0)) if mac == "detour" else 0.0) + 0.5
            self.prog = ex; self.prog_info = dict(macro=mac, geo=geo, t0=t, s_done=s_done, hazard=bool(hz.get("ok") or xs or gap.get("ok")))
            self.stall.reset()
            ph = prog.get("phases", {})
            posture = {k: v.get("posture") for k, v in ph.items()}
            state = f"{mac} [{ch.get('name')}] hazard s {s0:.2f}-{s1:.2f}"
            if self.trig and self.pending is not None:
                self.trig_log.append(dict(event="pending cancelled (non-triggered program)", t=round(t, 2), x=round(x, 3), name=self.pending["name"]))
                self.pending = None
        self.log.append(dict(t=round(t, 2), x=round(x, 3), y=round(y, 3), macro=mac, name=ch.get("name"), stock=ch["stock"],
                             p=round(ch["p_success"], 3), p_stock=round(ch["p_stock"], 3), env_act=ch["env_act"], posture=posture,
                             feas=np.round(fv, 3).tolist(), feas_v03_single=np.round(fv03[:5], 3).tolist(), perc=self.perc,
                             n_mem=self.fmem.n(), n_mem_prog=self.n_mem_prog, n_model_frames=int(len(hist_obs["t_hist"])) if self.model_hist else 1, n_hist=int(H["valid"].sum()), lat_room=lat_room,
                             blocked={w: sum(1 for r in ch["table"] if r.get("why") == w) for w in {r.get("why") for r in ch["table"]} if w},
                             top=sorted([dict(name=r["name"], p=r["p"], std=r["std"], allowed=r["allowed"]) for r in ch["table"]],
                                        key=lambda r: -r["p"])[:6], ms=round(dt_ms, 1), state=state, rule=self.rule))
        if trig_rec is not None:
            self.log[-1]["trig"] = trig_rec
        if self.cam_mount != "pelvis":
            self.log[-1]["cam"] = dict(mount=self.cam_mount, eye_z=round(float(M[3, 2]), 3),
                                       axis_elev_deg=round(math.degrees(math.asin(float(np.clip(-M[2, 2], -1, 1)))), 1),
                                       self_frac=round(self.cam.last.get("mask_frac", 0.0), 4))
        self.last = f"gen3[{state}] P={ch['p_success']:.2f} (stock {ch['p_stock']:.2f}) h_free {fd['h_free_min']:.2f}"
        if self.dump:
            os.makedirs(self.dump, exist_ok=True)
            np.savez_compressed(os.path.join(self.dump, f"{self.ep_id}_{t:07.2f}.npz"), rgb=np.asarray(rgb, np.uint8), depth=np.asarray(dep, np.float32),
                                M=M, base=np.array([x, y, float(bs["pos"][2]), yaw]), t=np.float64(t), feas=np.asarray(fv, np.float32),
                                feas_v03=np.asarray(fv03, np.float32), floor_w=np.float64(self.floor_w if self.floor_w is not None else np.nan),
                                choice=json.dumps(dict(macro=mac, name=ch.get("name"), stock=ch["stock"], p=ch["p_success"], p_stock=ch["p_stock"])))

    def _run_prog(self, obs):
        t, bs = float(obs["t"]), obs["bs"]
        I = self.prog_info
        s, e = _Lane.lane_coords(bs, I["geo"])
        done = False
        if I["macro"] == "constant":
            done = (s >= I["s_done"]) if I["hazard"] else (t - I["t0"] >= 2.0)
        elif I["macro"] == "turn_side_turn":
            done = getattr(self.prog, "phase", "") == "go" and s >= I["geo"]["s_end"] + 0.5
        else:
            done = s >= I["s_done"]
        TI = I.get("trig")
        if TI is not None:                       # XFEAS_GEN3_TRIGGER=dist: live memory-map edges
            f_l, e_l = TI["trk"].dist(bs["pos"], TI["heading"])
            if TI["kind"] in ("over", "lane"):
                xh = self._head_x(bs)
                xs_ = self.x_head_stand if self.x_head_stand is not None else xh
                thr = min(xh, xs_) - (0.5 * self.stand_depth + self.TRG.POS_MARGIN)
                done = e_l is not None and e_l <= thr
                if done and self.safe_stand and I.get("narrow_here"):
                    if not I.get("held_logged"):
                        ev = dict(t=round(t, 2), x=round(float(bs["pos"][0]), 3), event="release held (free width < standing width here)", name=TI["name"])
                        self.trig_log.append(ev); self.log.append(dict(ev, state="RELEASE HELD narrow")); print(f"[gen3] {ev}", flush=True)
                        I["held_logged"] = True
                    done = False
                if done:
                    ev = dict(t=round(t, 2), x=round(float(bs["pos"][0]), 3), event="release (hazard end behind the head)", name=TI["name"],
                              kind=TI["kind"], d_live_end=round(e_l, 3), x_head_live=round(xh, 3), x_head_stand=round(xs_, 3),
                              threshold=round(thr, 3), n_track_updates=TI["trk"].n_upd, t_in_prog=round(t - I["t0"], 2))
                    self.trig_log.append(ev); self._trig_dump(ev); self.log.append(dict(ev, state=f"RELEASE {TI['name']} end {e_l:.2f} <= {thr:.2f}"))
                    print(f"[gen3] release {ev}", flush=True)
            elif e_l is not None and getattr(self.prog, "phase", "") in ("approach", "turn", "side"):
                I["geo"]["s_end"] = s + e_l; self.prog.s1 = s + e_l
        if t - I["t0"] > 40.0:
            done = True
        ph = getattr(self.prog, "phase", "")
        if (I["macro"] == "turn_side_turn" and ph in ("turn", "back") and os.environ.get("XFEAS_GEN3_STALL_PHASE", "0") == "1"):
            self.stall.reset()     # additive 2026-10-19: the 5 s stall window must not span the (exempt) in-place turn
        if not done and not (I["macro"] == "turn_side_turn" and ph in ("turn", "back")) and self.stall.update(t, s):
            self.log.append(dict(t=round(t, 2), x=round(float(bs["pos"][0]), 3), state=f"{I['macro']} stalled 5 s (<0.2 m) -> re-decide"))
            done = True
        if done:
            was_posture = TI is not None and TI["kind"] in ("over", "lane")
            self.prog = None; self.t_next = t
            self.vrob.command(dict(self.driver.nominal_posture)); self.vrob.set_env_intent(self.v_nom, 9.9, 9.9, 0.0)
            if self.safe_stand and was_posture:
                self.standup = dict(t0=t)
                self.vrob.set_env_intent(0.0, 9.9, 9.9, 0.0)
                print(f"[gen3] stand-up start t={t:.2f}", flush=True)
                return np.zeros(4)
            return None
        r = self.prog.act(obs)
        self.last = f"gen3[{I['macro']}:{ph}]"
        return np.asarray(r["twist"], float)

    def _mode_tag(self):
        ap = getattr(self.vrob._v_inner, "applied", None) or {}
        return f" | SONIC mode {ap.get('mode')} speed {ap.get('speed')}"


class StraightXFeasGen3(_Gen3Layer, LN.LocalNav):
    name = "gen3_straight"
    label = "+ XFeas gen3 (zero-shot)"

    def setup(self, ctx):
        LN.LocalNav.setup(self, ctx)
        self.kind_cam = str(ctx["kind"])
        self._g3_setup(ctx)

    def reset(self, ep):
        LN.LocalNav.reset(self, ep)
        self._g3_reset(ep)

    def act(self, obs):
        tw = self._g3_tick(obs)
        if tw is not None:
            return dict(twist=tw, info=self.last + self._mode_tag(), decisions=self.log)
        v0 = self.v
        self.v = self.v_cap
        try:
            base = straight_act(self, obs)
        finally:
            self.v = v0
        self.vrob.command(np.asarray(base["twist"], float))
        return dict(twist=base["twist"], info=(self.last or base["info"]) + self._mode_tag(), decisions=self.log)


try:
    from stairs_methods import StairsStock

    class StairsXFeasGen3(_Gen3Layer, StairsStock):
        name = "gen3_stairs"
        label = "+ XFeas gen3 (zero-shot) on the stairs path follower"

        def setup(self, ctx):
            StairsStock.setup(self, ctx)
            self.kind_cam = str(ctx["kind"])
            self._g3_setup(ctx)

        def reset(self, ep):
            StairsStock.reset(self, ep)
            self._g3_reset(ep)

        def setpoints(self, obs, s, k):
            return None, self.v_cap, 0.0

        def act(self, obs):
            tw = self._g3_tick(obs)
            if tw is not None:
                return dict(twist=np.asarray(tw, float), info=self.last + self._mode_tag(), decisions=self.log)
            r = StairsStock.act(self, obs)
            self.vrob.command(np.asarray(r["twist"], float))
            return dict(twist=r["twist"], info=(self.last or r["info"]) + self._mode_tag(), decisions=self.log)
except Exception as _e:                              # outside Isaac
    print("[gen3_methods] stairs arm not loaded:", repr(_e))
