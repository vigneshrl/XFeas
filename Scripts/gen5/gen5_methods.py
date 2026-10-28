"""
Closed-loop gen5 arm for episode_runner (2026-10-27): the learned gen5 model (train5g4, data/gen_train/gen5g4/runs/g1c/
model.pt) picks the controller's OWN command every 0.25 s and that command is sent DIRECTLY to the controller.

  StairsXFeasGen5   the course path follower (stairs_methods.StairsStock: pure pursuit on the scene polyline -> wz, exactly
                    as the stock arm), env XFEAS_GEN5_ARM = gen5 | stock

REUSED UNCHANGED from scripts/gen4/gen4_methods._Gen4Layer (perception / model input only): head camera of the shared mount
(XFeasDecider._snapshot + the support-floor hook), 2 s proprio buffer, h(t) at 10 Hz through virtual.VRob (identity
wrapper, link-extent top / foot channels) above the support floor, the collector's 4-frame obs memory picked by DISTANCE
(_pick_hist / _hist_obs), feas v0.4 exactly as the gen_v4 collector (_feas_mem / _feas4), the request dict of
gen3_methods._Gen3Layer._decide (perception lines copied verbatim into _decide5), the diagnostics recorder hook (head camera
frames + joint state per video frame).

REMOVED (hand-written execution of gen3 / gen4; none of it is reachable from this arm):
  1. the DRIVER (scripts/gen3/driver.Driver: abstract option -> programs; driver.plan / feas_act / nominal_posture):
     driver.Driver is replaced by a raising guard WHILE _g3_setup runs, so the object is never even constructed;
     self.driver is the guard (any attribute access raises).
  2. the PROGRAM EXECUTORS gensim3.ProgramCtl3 / TurnSideCtl3 (and _run_prog): module attributes replaced by raising
     guards; _run_prog raises.
  3. the DISTANCE TRIGGER (gen3_trigger: pending programs, _pending_fire, Track, settle times, POS_MARGIN): env
     XFEAS_GEN3_TRIGGER=off (asserted), so _g3_setup never imports gen3_trigger and self.trig is False (asserted);
     _pending_fire / _trig_kind / _t_reach raise.
  4. the STAND-UP RULE (SAFE_STAND: stop, nominal posture, wait upright; the 'release when the hazard end is behind the
     head' rule; the narrow-width release hold): XFEAS_GEN3_SAFE_STAND=0 (asserted); _g3_tick (the only place it runs)
     raises -- this arm has its own tick (_g5_tick) that has no such branch.
  5. COMMIT HYSTERESIS (COMMIT_N / COMMIT_PDROP): XFEAS_GEN3_COMMIT_N=1 (asserted), COMMIT_PDROP unset; _commit_hold raises.
  6. GATES / rule / candidate vocabulary of gen3-gen4 (select_gen3.decide / hard_gates / rule_pick / candidates_from_probe,
     select_gen4 learned_selection + gate.json, XFEAS_GEN3_TOP_CLEAR): the gen4 sidecar is never started; select_gen3 /
     select_gen4 must not be in sys.modules (checked every episode); XFEAS_GEN3_TOP_CLEAR must be unset (asserted).
  7. the stall re-decide of programs (StallWatch) -- there are no programs.
  8. gen3_methods._Gen3Layer._decide / _g3_tick / _mem_update are overridden to raise.
  There is NO emergency stop in this arm; a fall ends the episode through the runner's mode-aware fall rule (outcome
  'fall', never success).
Verification: every guard RAISES (an episode that completes never touched them); each episode prints
'[gen5] VERIFY ...' at its first decision and at the next reset (guards installed, env flags, modules absent, count of
decisions executed directly); every decision row carries exec='direct'. A grep of this file for driver.plan / ProgramCtl3 /
_pending_fire / _commit_hold finds them only in these guards.

EXECUTION (both arms, the same setpoint path the gen4 stock arm uses for its nominal command):
  every 0.25 s (from t = 2 s, once 2 s of proprio exist): perception -> request -> sidecar scripts/gen5/select_gen5.py
  (gsam2 env, this job's GPU) -> chosen command {lin_vel_x, lin_vel_y, pelvis_height, torso_pitch, torso_roll[, mode, speed]}
    posture roles -> vrob.command({spec name: value})        (VRob identity wrapper -> the controller adapter)
    lin_vel_x     -> v_cap: the stock path follower's speed   (StairsStock.act; G1's own vmin 0.1 of that follower kept)
    lin_vel_y     -> twist[1]
    wz            -> the stock path follower (pure pursuit), exactly as stock
  arm 'stock': the sidecar is asked (its table is logged) but the NOMINAL command (candidate 0) is sent every tick.
  env intent channel of h(t) (VRob.set_env_intent; gen3 driver concept) is held at the stock value (v_nom, 9.9, 9.9, 0)
  in both arms, as during the gen4 snapshot history the model was trained on.
Logs (env XFEAS_GEN5_DIAG=<dir>): decisions5_t<k>.jsonl (one row per decision: t, x, s along the path, distance to the
first overhang front / past its end, body top above the floor, chosen + executed command, top-3 table, full table),
head_t<k>/ (head camera frames), state_t<k>.jsonl (gen4 diag hook).
"""
import atexit, json, math, os, struct, io, subprocess, sys, time
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
for _p in (os.path.join(ROOT, "scripts", "gen4"), os.path.join(ROOT, "scripts", "gen_train"), os.path.join(ROOT, "scripts", "gen3"),
           os.path.join(ROOT, "scripts", "gen"), os.path.join(ROOT, "scripts", "xembody"), os.path.join(ROOT, "scripts", "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gen4_methods as G4M
import gen3_methods as GM
import gen3_closed_loop as GCL

PERIOD = float(os.environ.get("XFEAS_GEN5_PERIOD", "") or GM.PERIOD)   # (2026-10-28) decision interval sweep; unset = GM.PERIOD (0.25 s)
SIDECAR = os.path.join(THIS, "select_gen5.py")
CKPT5 = os.path.join(ROOT, "data", "gen_train", "gen5g4", "runs", "g1c", "model.pt")
FORBIDDEN_MODULES = ("select_gen3", "select_gen4", "gen3_trigger")


class GuardHit(RuntimeError):
    pass


class _Forbidden:
    def __init__(self, what):
        object.__setattr__(self, "_what", what)

    def __getattr__(self, k):
        raise GuardHit(f"[gen5] FORBIDDEN: {object.__getattribute__(self, '_what')}.{k} used in the gen5 arm")

    def __call__(self, *a, **k):
        raise GuardHit(f"[gen5] FORBIDDEN: {object.__getattribute__(self, '_what')}() called in the gen5 arm")


# ---------------------------------------------------------------------- sidecar client (framing of select_gen5 / select_gen4)
def _read(f, n):
    b = b""
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            raise EOFError
        b += c
    return b


class Gen5Client:
    PY = "/p/cral/vignesh/envs/gsam2/bin/python"

    def __init__(self, ckpt, log=None):
        e = dict(os.environ)
        assert e.get("CUDA_VISIBLE_DEVICES", "0") == "0", "GPU 0 only"
        e["CUDA_VISIBLE_DEVICES"] = "0"
        e["OMP_NUM_THREADS"] = e.get("OMP_NUM_THREADS", "2")
        e["OPENBLAS_NUM_THREADS"] = "1"; e["MKL_NUM_THREADS"] = "1"      # RLIMIT_NPROC 2000 per user with ~8 Isaac jobs
        e.pop("PYTHONPATH", None)                      # the gsam2 env resolves its own modules from scripts/gen5
        self.p = subprocess.Popen([self.PY, SIDECAR, "--ckpt", ckpt, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=(open(log, "a") if log else None), env=e)
        print(f"[gen5] sidecar pid {self.p.pid} ({SIDECAR}, ckpt {ckpt})", flush=True)
        atexit.register(self.close)

    def ask(self, obs, ctrl_id, spec_path, probe_path, robot):
        b = io.BytesIO()
        np.savez(b, ctrl_id=ctrl_id, spec_path=spec_path, probe_path=probe_path, robot=robot, **obs)
        raw = b.getvalue()
        self.p.stdin.write(struct.pack("<I", len(raw))); self.p.stdin.write(raw); self.p.stdin.flush()
        hdr = self.p.stdout.read(4)
        if len(hdr) < 4:
            raise RuntimeError("[gen5] sidecar died")
        n = struct.unpack("<I", hdr)[0]
        z = np.load(io.BytesIO(_read(self.p.stdout, n)), allow_pickle=False)
        ch = json.loads(str(z["choice"]))
        if "error" in ch:
            raise RuntimeError(f"[gen5] sidecar error: {ch['error']}")
        return ch

    def close(self):
        try:
            if self.p.poll() is None:
                self.p.stdin.close(); self.p.wait(10)
        except Exception:
            try:
                self.p.kill()
            except Exception:
                pass


class _Gen5Layer(G4M._Gen4Layer):
    # -------------------------------------------------------------- removed execution parts: raise if ever reached
    def _g3_tick(self, obs):
        raise GuardHit("[gen5] FORBIDDEN: gen3 tick (programs / trigger / stand-up rule)")

    def _decide(self, obs):
        raise GuardHit("[gen5] FORBIDDEN: gen3/gen4 _decide (driver programs, trigger, commit hold)")

    def _run_prog(self, obs):
        raise GuardHit("[gen5] FORBIDDEN: _run_prog (ProgramCtl3 execution / release rule)")

    def _pending_fire(self, obs):
        raise GuardHit("[gen5] FORBIDDEN: _pending_fire (distance trigger)")

    def _commit_hold(self, *a, **k):
        raise GuardHit("[gen5] FORBIDDEN: _commit_hold (commit hysteresis)")

    def _trig_kind(self, *a, **k):
        raise GuardHit("[gen5] FORBIDDEN: _trig_kind (distance trigger)")

    def _t_reach(self, *a, **k):
        raise GuardHit("[gen5] FORBIDDEN: _t_reach (distance trigger)")

    def _mem_update(self, t, bs):
        raise GuardHit("[gen5] FORBIDDEN: _mem_update (memory during programs)")

    # -------------------------------------------------------------- setup
    def _g5_env(self):
        self.arm = os.environ.get("XFEAS_GEN5_ARM", "gen5")
        assert self.arm in ("gen5", "stock"), self.arm
        assert os.environ.get("XFEAS_GEN3_TRIGGER") == "off", "XFEAS_GEN3_TRIGGER must be 'off'"
        assert os.environ.get("XFEAS_GEN3_SAFE_STAND") == "0", "XFEAS_GEN3_SAFE_STAND must be 0"
        assert os.environ.get("XFEAS_GEN3_COMMIT_N") == "1", "XFEAS_GEN3_COMMIT_N must be 1"
        assert not os.environ.get("XFEAS_GEN3_COMMIT_PDROP"), "XFEAS_GEN3_COMMIT_PDROP must be unset"
        assert not os.environ.get("XFEAS_GEN3_TOP_CLEAR"), "XFEAS_GEN3_TOP_CLEAR must be unset"
        self.ctrl = os.environ.get("XFEAS_GEN5_CTRL", "g1c")
        self.robot5 = "spot" if self.ctrl.startswith("spot") else "g1"
        self.spec_path = os.path.join(ROOT, "data", "gen_v4", "specs", f"{self.ctrl}.json")
        self.probe_path = os.path.join(ROOT, "data", "probes_v4", self.ctrl, "office_v4.json")
        os.environ["XFEAS_GEN3_SPEC"] = self.spec_path
        os.environ["XFEAS_GEN3_PROBE"] = self.probe_path
        os.environ["XFEAS_GEN3_RULE"] = "0"
        os.environ["XFEAS_GEN3_PERC"] = "v04"; os.environ["XFEAS_GEN3_MODEL_HIST"] = "1"; os.environ["XFEAS_GEN3_MEM_PROG"] = "0"
        self.ckpt5 = os.environ.get("XFEAS_GEN5_CKPT") or CKPT5
        lbl = {"gen5": "+ XFeas gen5 (learned, direct commands)", "stock": "stock (nominal command)"}[self.arm]
        os.environ.setdefault("XFEAS_LABEL", f"{self.ctrl} {lbl}")

    def _g5_setup(self, ctx):
        self._g5_env()
        import driver as DR
        Gc, Dc = GCL.Gen3Client, DR.Driver
        GCL.Gen3Client = lambda *a, **k: None                       # the gen3 sidecar is never started
        DR.Driver = lambda *a, **k: _Forbidden("driver.Driver")     # the driver is never constructed
        try:
            self._g3_setup(ctx)
        finally:
            GCL.Gen3Client, DR.Driver = Gc, Dc
        assert isinstance(self.driver, _Forbidden) and self.trig is False and self.safe_stand is False
        import gensim3 as G3
        G3.ProgramCtl3 = _Forbidden("gensim3.ProgramCtl3"); G3.TurnSideCtl3 = _Forbidden("gensim3.TurnSideCtl3")
        self.G3 = G3
        self.client = _Forbidden("gen3/gen4 sidecar client")
        self.diag = os.environ.get("XFEAS_GEN5_DIAG", "")
        if self.diag:
            os.makedirs(self.diag, exist_ok=True)
        self.client5 = Gen5Client(self.ckpt5, log=os.path.join(self.diag, "sidecar.log") if self.diag else None)
        self._feas4_now, self._fd4_now = None, None
        self._bases = {}
        self._ctx = ctx
        snap0 = self.cam._snapshot

        def snap(bs, _s0=snap0):                                     # == _Gen4Layer._g4_setup support-floor hook
            r = _s0(bs)
            ch_ = self.cam.last.get("cam_h") if isinstance(getattr(self.cam, "last", None), dict) else None
            if ch_ is not None and np.isfinite(ch_):
                self.floor_sup = float(self.cam.last["M"][3, 2]) - float(ch_)
                self.floor = self.floor_sup
            self._last_bs = bs
            return r
        self.cam._snapshot = snap
        self.floor_sup = None
        import ctrl_spec as CS
        self.CS5 = CS
        sc = None
        try:
            import episode_runner as ER
            sc = ER.SCENES.get(os.environ.get("XFEAS_GEN5_SCENE", "SC2A_O1"))
        except Exception:
            pass
        self.oh = (sc or {}).get("overhang")
        print(f"[gen5] arm {self.arm}; ctrl {self.ctrl}; spec {self.spec_path}; probe {self.probe_path}; ckpt {self.ckpt5}; "
              f"TRIGGER {os.environ.get('XFEAS_GEN3_TRIGGER')} SAFE_STAND {os.environ.get('XFEAS_GEN3_SAFE_STAND')} "
              f"COMMIT_N {os.environ.get('XFEAS_GEN3_COMMIT_N')}; overhang {self.oh}; diag {self.diag or '-'}", flush=True)
        self._n_exec_total = 0
        if self.diag:
            os.makedirs(self.diag, exist_ok=True)
            self._diag_setup()

    def _verify_line(self, where):
        mods = [m for m in FORBIDDEN_MODULES if m in sys.modules]
        ok = (isinstance(self.driver, _Forbidden) and isinstance(self.G3.ProgramCtl3, _Forbidden) and isinstance(self.G3.TurnSideCtl3, _Forbidden)
              and self.trig is False and self.safe_stand is False and not mods and self.prog is None and self.pending is None and self.standup is None)
        s = (f"[gen5] VERIFY ({where}) ep {getattr(self, 'ep_id', '-')} arm {self.arm}: driver guard {isinstance(self.driver, _Forbidden)}, "
             f"ProgramCtl3/TurnSideCtl3 guards {isinstance(self.G3.ProgramCtl3, _Forbidden)}/{isinstance(self.G3.TurnSideCtl3, _Forbidden)}, "
             f"trigger {self.trig}, safe_stand {self.safe_stand}, COMMIT_N {os.environ.get('XFEAS_GEN3_COMMIT_N')}, forbidden modules loaded {mods}, "
             f"prog/pending/standup {self.prog}/{self.pending}/{self.standup}; decisions executed directly this episode "
             f"{getattr(self, '_n_exec', 0)} -> {'OK' if ok else 'VIOLATION'}")
        print(s, flush=True)
        return ok

    def _g5_reset(self, ep):
        if getattr(self, "_n_exec", 0):
            self._verify_line("end of previous episode")
        self._g4_reset(ep)                      # _g3_reset (buffers, h(t), memory) + gen4 diag reset
        self.v_cap, self.vy_cmd = self.v_nom, 0.0
        self.cmd_now, self.ch_now = None, None
        self._t_sw, self._n_switch = None, 0
        self._n_exec, self._t_dec0 = 0, None
        self._top_pass = []
        import stairs_scenes as SS
        self._pp5 = SS.PathProgress(SS.STAIR_SCENES[ep["scene"]]["path"])
        if self.diag:
            p = os.path.join(self.diag, f"decisions5_t{self.ep_trial}.jsonl")
            if os.path.exists(p):
                os.remove(p)

    # -------------------------------------------------------------- perception + decision (no execution logic)
    def _decide5(self, obs):
        t, bs, rob = float(obs["t"]), obs["bs"], obs["rob"]
        self._last_bs = bs
        hb = os.environ.get("XFEAS_GEN5_HB", "")
        if hb and time.time() - getattr(self, "_hb_t", 0.0) > 20.0:
            self._hb_t = time.time()
            try:
                open(hb, "w").write(f"{time.time():.0f} t {t:.2f} x {float(bs['pos'][0]):.3f}\n")
            except Exception:
                pass
        # ---- verbatim perception of gen3_methods._Gen3Layer._decide (with the gen4 overrides _feas_mem / _hist_obs)
        rgb, dep = self.cam._snapshot(bs)
        self._last_rgb = np.asarray(rgb, np.uint8)
        pb, pj = self.cam.buf.arrays()
        if self.jn is None:
            names = list(rob._art.dof_names)
            self.jn = names if len(names) == pj.shape[1] else None
            print(f"[gen5] proprio joints {pj.shape[1]}, articulation dofs {len(names)} -> names {'ok' if self.jn else 'MISMATCH (proprio masked)'}", flush=True)
        H = self.vrob.hist_arrays(geo=self.geo0, floor_z=self.floor, t_ref=t)
        M = self._cam_M(bs)
        fv03, fd03 = self._feas(dep)
        fv, fd = self._feas_mem(t, bs, dep, M)
        hist_obs = self._hist_obs(t, np.asarray(rgb, np.uint8), np.asarray(dep, np.float32), M)
        req = dict(rgb=np.asarray(rgb, np.uint8), depth=np.asarray(dep, np.float32), prop_base=pb.astype(np.float32),
                   prop_joints=(pj if self.jn else np.zeros_like(pj)).astype(np.float32), fmt="gen_v3",
                   joint_names=json.dumps(self.jn or [f"j{i}" for i in range(pj.shape[1])]),
                   hist_t=H["t"], hist_valid=H["valid"], hist_cmd=H["cmd"], hist_env=H["env"], hist_ach=H["ach"], hist_q=H["q"], hist_dq=H["dq"],
                   ach_keys=json.dumps(list(self.VC.ACH_KEYS)), feas=np.asarray(fv, np.float32))
        if self._feas4_now is not None:                                  # == _Gen4Client.ask
            req["feas"] = np.asarray(self._feas4_now, np.float32)
        if self.model_hist and len(hist_obs["t_hist"]) > 1:
            req.update(hist_obs)
        if self.cam_mount != "pelvis":
            req.update(cam_mount_M=np.asarray(M, np.float64), cam_mount_base=np.asarray(self._base_of(bs), np.float64))
            if self.cam.last.get("cam_h") is not None:
                req.update(cam_mount_camh=np.float64(self.cam.last["cam_h"]))
        # decision state for gen5b checkpoints (train5b / dataset5b.decision_state): the command in force, time since the
        # last switch, any switch yet, measured forward speed. Ignored by the gen5g4 sidecar mode.
        req.update(g5_cur_cmd=json.dumps(self.cmd_now) if self.cmd_now is not None else "",
                   g5_t_since=np.float64(t - self._t_sw if self._t_sw is not None else t), g5_any_switch=np.bool_(self._t_sw is not None),
                   g5_vx=np.float64(np.asarray(bs["v_b"], float)[0]))
        t0 = time.time()
        ch = self.client5.ask(req, self.ctrl, self.spec_path, self.probe_path, self.robot5)
        ms = (time.time() - t0) * 1e3
        # ---- execution = send the command (gen5: the model's pick; stock: candidate 0 = nominal)
        cmd = dict(ch["cmd"]) if self.arm == "gen5" else dict(ch["table"][0]["cmd"])
        assert self.arm == "gen5" or ch["table"][0]["kind"] == "NOMINAL"
        assert ch["table"][0]["kind"] == "NOMINAL"
        prev = self.cmd_now if self.cmd_now is not None else dict(ch["table"][0]["cmd"])     # before the first decision: nominal
        if json.dumps(prev, sort_keys=True) != json.dumps(cmd, sort_keys=True):
            self._t_sw = t; self._n_switch += 1
        self._send(cmd)
        self.cmd_now, self.ch_now = cmd, ch
        self._n_exec += 1; self._n_exec_total += 1
        if self._n_exec == 1:
            self._verify_line("first decision")
        elif self._n_exec % 40 == 0:
            self._verify_line(f"decision {self._n_exec}")
        self._log5(obs, ch, cmd, ms, fd)

    def _send(self, cmd):
        """the chosen command -> the controller (same calls the gen4 stock arm makes for its nominal command)"""
        post = {}
        for role in ("pelvis_height", "torso_pitch", "torso_roll", "mode", "speed"):
            if role in cmd:
                c = self.CS5.cmd(self.cspec, role)
                if c is not None:
                    post[c["name"]] = cmd[role]
        if post:
            self.vrob.command(post)
        self.vrob.set_env_intent(self.v_nom, 9.9, 9.9, 0.0)
        self.v_cap = float(cmd.get("lin_vel_x", self.v_nom))
        self.vy_cmd = float(cmd.get("lin_vel_y", 0.0) or 0.0)

    def _top_above_floor(self):
        try:
            v = float(self.shim._top_now())
            fz = self.oh["floor"] if self.oh else (self.floor_sup or 0.0)
            return round(v - fz, 3) if np.isfinite(v) else None
        except Exception:
            return None

    def _log5(self, obs, ch, cmd, ms, fd):
        t, bs = float(obs["t"]), obs["bs"]
        x, y = float(bs["pos"][0]), float(bs["pos"][1])
        s, _ = self._pp5.update(x, y)
        top = self._top_above_floor()
        tab = sorted(ch["table"], key=lambda r: -r["U"])
        cmp = lambda r: dict(cls=r["cls"], cmd=r["cmd"], U=r["U"], p=r["p"], p_std=r["p_std"], p_fall=r["p_fall"])
        oh = self.oh or {}
        row = dict(t=round(t, 2), arm=self.arm, x=round(x, 3), y=round(y, 3), z=round(float(bs["pos"][2]), 3), s=round(s, 3),
                   d_to_overhang=None if not oh else round(oh["x0"] - x, 3), past_overhang_end=None if not oh else round(x - oh["x1"], 3),
                   top=top, overhang_under=oh.get("under_h"), tilt=round(float(bs["tilt"]), 3), vx_meas=round(float(np.asarray(bs["v_b"])[0]), 3),
                   chosen=dict(cls=ch["cls"], cmd=ch["cmd"], U=round(ch["U"], 4), p=round(ch["p"], 4), stock=ch["stock"]),
                   executed=cmd, exec="direct", n_switch=self._n_switch, t_sw=None if self._t_sw is None else round(self._t_sw, 2),
                   state=ch.get("state"), model=ch.get("model"), U_stock=round(ch["U_stock"], 4), p_stock=round(ch["p_stock"], 4),
                   top3=[cmp(r) for r in tab[:3]], h_free_min=None if not fd else fd.get("h_free_min"), x_at_h_free=None if not fd else fd.get("x_at_h_free"),
                   feas4=None if self._feas4_now is None else np.round(self._feas4_now, 3).tolist(), floor_sup=None if self.floor_sup is None else round(self.floor_sup, 3),
                   n_frames=int(len(self._peek["t_hist"])) if getattr(self, "_peek", None) else None, ms=round(ms, 1),
                   frame=int(self._dx["rec"].n) if self.diag and self._dx.get("rec") is not None else None, table=ch["table"])
        self.log.append({k: v for k, v in row.items() if k not in ("table", "feas4")})
        self.last = f"gen5[{self.arm}] {ch['cls']} vx {cmd.get('lin_vel_x')} h {cmd.get('pelvis_height')} p {cmd.get('torso_pitch')} U {ch['U']:.2f}"
        if self.diag:
            open(os.path.join(self.diag, f"decisions5_t{self.ep_trial}.jsonl"), "a").write(json.dumps(row, default=G4M._js) + "\n")
        print(f"[gen5] t {t:6.2f} x {x:8.3f} d_oh {row['d_to_overhang']} top {top} -> {self.arm} {'SENT ' + ch['cls'] if self.arm == 'gen5' else 'nominal (model: ' + ch['cls'] + ')'} "
              f"vx {cmd.get('lin_vel_x')} h {cmd.get('pelvis_height')} p {cmd.get('torso_pitch')} | U {ch['U']:.3f} stock {ch['U_stock']:.3f} | {ms:.0f} ms", flush=True)

    # -------------------------------------------------------------- tick
    def _g5_tick(self, obs):
        t, bs, rob = float(obs["t"]), obs["bs"], obs["rob"]
        self.cam.buf.tick(rob, bs)
        if self.geo0 is None:
            g = np.asarray(obs["goal_xy"], float); p = np.asarray(bs["pos"][:2], float)
            self.geo0 = GM.lane_geo(p, math.atan2(g[1] - p[1], g[0] - p[0]))
        object.__setattr__(self.vrob, "_v_t", t)
        if t >= self.hnext:                                   # h(t) at 10 Hz, as _g3_tick
            self.hnext = t + 0.1 - 1e-9
            try:
                self.vrob._record()
            except Exception as e:
                print("[gen5] history record failed:", repr(e), flush=True)
        if t >= self.t_next and self.cam.buf.full():
            self.t_next = t + PERIOD
            self._decide5(obs)
        if self.oh and self.oh["x0"] - 0.6 <= float(bs["pos"][0]) <= self.oh["x1"] + 0.3 and int(round(t * 50)) % 5 == 0:
            tp = self._top_above_floor()
            if tp is not None:
                self._top_pass.append(tp)


try:
    from stairs_methods import StairsStock

    class StairsXFeasGen5(_Gen5Layer, StairsStock):
        name = "gen5_stairs"
        label = "+ XFeas gen5 on the course path follower"

        def setup(self, ctx):
            StairsStock.setup(self, ctx)
            self.kind_cam = str(ctx["kind"])
            self._g5_setup(ctx)
            self.label = os.environ.get("XFEAS_LABEL", self.label)

        def reset(self, ep):
            StairsStock.reset(self, ep)
            self._g5_reset(ep)

        def setpoints(self, obs, s, k):
            return None, self.v_cap, 0.0                     # speed = the command's lin_vel_x; no setpoint dict here

        def act(self, obs):
            self._g5_tick(obs)
            r0 = StairsStock.act(self, obs)                  # the stock path follower: wz (and its turn-first / vmin)
            tw = np.asarray(r0["twist"], float).copy()
            tw[1] = self.vy_cmd
            self.vrob.command(tw)
            info = (self.last or r0["info"])
            if self._top_pass:
                info += f" | top under lintel max {max(self._top_pass):.2f}"
            r = dict(twist=tw, info=info, decisions=self.log)
            if self.diag:
                self._dx["t"] = float(obs["t"]); self._dx["tw"] = tw
            return r
except Exception as _e:                              # outside Isaac
    print("[gen5_methods] stairs arm not loaded:", repr(_e))
