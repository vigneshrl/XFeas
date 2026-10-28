"""
Closed-loop gen4 arms (Phase 4 of the v4 rebuild, 2026-10-21) for episode_runner (course scenes) and run_terrain (tiles).

  StraightXFeasGen4   straight drive to the goal (terrain tiles)
  StairsXFeasGen4     the course path follower (CC1 = L1, CC3 = L2, CFB110 = v3flat-b)

The layer is gen3_methods._Gen3Layer UNCHANGED (decision period 0.25 s, distance trigger, v4 execution fixes END_ROBUST /
COMMIT_N 3, memory during programs, stall re-decide, executors gensim3.ProgramCtl3 / TurnSideCtl3) with three gen4 parts:
  1. sidecar = closed_loop/select_gen4.py (gen4 release ensemble + learned_selection + gate, gen4 featurisation; spec /
     probe sent by PATH, robot 'g1');
  2. the model's feas input is computed EXACTLY as the gen_v4 collector (run_gen_v4.feas_v4): feas_mem.mem_points over
     the 4-frame obs memory with every frame's own camera matrix and robot root pose, self_x 0 (exact self mask at
     render), floor = the robot's support height (eye z - cam_h of the shared mount = its lowest body point), v0.4 rules,
     MINPOOL fix. The 4 frames are picked like the collector picks them: by DISTANCE travelled, 0.85 v_nom x (1.5, 1.0,
     0.5) s behind the decision (time fallback while the robot stands still). The long keyframe memory of gen3 is kept for
     the trigger / tracker (execution) only;
  3. h(t) z_base / top are expressed above the same support floor (the collector's lane floor), not the episode's spawn
     floor.
Arms (env XFEAS_GEN4_ARM): gen4 = learned decision; rule = select_gen3.rule_pick on the same perception / probe /
driver; stock = the controller's nominal command at every decision (the sidecar is still asked, so P(stock) is logged,
but its answer is never executed). Nothing is tuned; no gate is added (XFEAS_GEN3_TOP_CLEAR must be unset).
Controller (env XFEAS_GEN4_CTRL): spec data/gen_v4/specs/<ctrl>.json, probe data/probes_v4/<ctrl>/office_v4.json.
Diagnostics (env XFEAS_GEN4_DIAG=<dir>): every decision -> decisions_t<k>.jsonl (full candidate table), dec_t<k>/<i>.npz
(decision rgb / depth / camera matrix / 4-frame memory), floor check (PhysX rays vs rendered depth, robot colliders
skipped -- the gen4 head-camera fix), and recorder hooks (head cam + joint state per video frame) like gen3_diag_methods.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import gc, json, math, os, sys, time
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas_paths.ROOT
for _p in (THIS, os.path.join(ROOT, "closed_loop"), os.path.join(ROOT, "Scripts", "data_collector"), os.path.join(ROOT, "Scripts", "data_collector"),
           os.path.join(ROOT, "closed_loop"), os.path.join(ROOT, "isaac_sim")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gen3_methods as GM
import gen3_closed_loop as GCL
import local_nav as LN
from door_methods import straight_act

T_HIST_V2 = (-1.5, -1.0, -0.5)          # gensim_v2.T_HIST (distance targets at 0.85 x v_nom), the decision frame last
V_EST = 0.85
T_FALLBACK = (-2.0, -1.33, -0.67)       # typical collected t_hist (QA: g1c -1.98 / -1.32 / -0.66) when standing still


class _StockAnswer:
    pass


class _Gen4Client:
    """adapts the gen3 layer's client.ask(req, cspec, probe, lat_room, gap_e, rule) to select_gen4 (paths, robot)"""

    def __init__(self, layer, ckpt):
        import select_gen4 as S4
        self.layer = layer
        self.c = S4.Gen4Client(ckpt)
        self.last = None

    def ask(self, req, cspec, probe, lat_room=(1.0, 1.0), gap_e=0.0, rule=False):
        L = self.layer
        req = dict(req)
        fv4 = L._feas4_now
        if fv4 is not None:
            req["feas"] = np.asarray(fv4, np.float32)
        ch = self.c.ask(req, L.spec_path, L.probe_path, robot=("spot" if "spot" in L.ctrl else "g1"), lat_room=lat_room, gap_e=gap_e, rule=(L.arm == "rule"))
        ch["arm"] = L.arm
        if L.arm == "stock" and not ch["stock"]:
            t0 = ch["table"][0]
            ch.update(model_pick=dict(name=ch["name"], macro=ch["macro"], p=ch["p_success"]), macro="constant", env_act=t0["env_act"],
                      name="stock", stock=True, p_success=ch["p_stock"], program=None)
            ch["program"] = L.driver.plan(dict(macro="constant", v_b=L.v_nom, top_b=9.9, wid_b=9.9, pitch_b=0.0, stock=True))
        self.last = ch
        return ch

    def close(self):
        self.c.close()


class _Gen4Layer(GM._Gen3Layer):
    def _g4_env(self):
        self.ctrl = os.environ.get("XFEAS_GEN4_CTRL", "g1c")
        self.arm = os.environ.get("XFEAS_GEN4_ARM", "gen4")
        assert self.arm in ("gen4", "rule", "stock"), self.arm
        assert not os.environ.get("XFEAS_GEN3_TOP_CLEAR"), "XFEAS_GEN3_TOP_CLEAR set: not the fitted gen4 gate"
        self.spec_path = os.path.join(ROOT, "data", "gen_v4", "specs", f"{self.ctrl}.json")
        self.probe_path = os.path.join(ROOT, "data", "probes_v4", self.ctrl, "office_v4.json")
        os.environ["XFEAS_GEN3_SPEC"] = self.spec_path
        os.environ["XFEAS_GEN3_PROBE"] = self.probe_path
        os.environ["XFEAS_GEN3_RULE"] = "1" if self.arm == "rule" else "0"
        os.environ["XFEAS_GEN3_PERC"] = "v04"; os.environ["XFEAS_GEN3_MODEL_HIST"] = "1"; os.environ["XFEAS_GEN3_MEM_PROG"] = "1"
        os.environ.setdefault("XFEAS_GEN3_TRIGGER", "dist")
        self.ckpt4 = os.environ.get("XFEAS_GEN4_CKPT") or os.path.join(ROOT, "data", "xe_train", "gen4", "xfeas_gen4_ens.pt")
        os.environ["XFEAS_GEN3_CKPT"] = self.ckpt4
        lbl = {"gen4": "+ XFeas gen4 (learned)", "rule": "+ simple rule (same perception / probe / driver)", "stock": "stock (nominal command)"}[self.arm]
        os.environ.setdefault("XFEAS_LABEL", f"{self.ctrl} {lbl}")

    def _g4_setup(self, ctx):
        self._g4_env()
        Gc = GCL.Gen3Client
        GCL.Gen3Client = lambda *a, **k: None                      # the gen3 sidecar is never started
        try:
            self._g3_setup(ctx)
        finally:
            GCL.Gen3Client = Gc
        self.client = _Gen4Client(self, self.ckpt4)
        self._feas4_now, self._fd4_now = None, None
        self._bases = {}
        self.diag = os.environ.get("XFEAS_GEN4_DIAG", "")
        self._ctx = ctx
        snap0 = self.cam._snapshot

        def snap(bs, _s0=snap0):
            r = _s0(bs)
            ch_ = self.cam.last.get("cam_h") if isinstance(getattr(self.cam, "last", None), dict) else None
            if ch_ is not None and np.isfinite(ch_):
                self.floor_sup = float(self.cam.last["M"][3, 2]) - float(ch_)
                self.floor = self.floor_sup                       # h(t) z_base / top above the support floor (collector lane floor)
            self._last_bs = bs
            return r
        self.cam._snapshot = snap
        self.floor_sup = None
        print(f"[gen4] arm {self.arm}; ctrl {self.ctrl}; spec {self.spec_path}; probe {self.probe_path}; ckpt {self.ckpt4}; "
              f"exec flags DRV_LAT {os.environ.get('XFEAS_GEN3_DRV_LAT')} DRV_PROBE {os.environ.get('XFEAS_GEN3_DRV_PROBE')} "
              f"DRV_LAT_VX {os.environ.get('XFEAS_GEN3_DRV_LAT_VX')} SPEED_FOLLOW_V {os.environ.get('XFEAS_GEN3_SPEED_FOLLOW_V')} "
              f"MODE_DEFAULT_SPEED {os.environ.get('XFEAS_GEN3_MODE_DEFAULT_SPEED')} TRIGGER {os.environ.get('XFEAS_GEN3_TRIGGER')} "
              f"COMMIT_N {os.environ.get('XFEAS_GEN3_COMMIT_N', '3')} END_ROBUST {os.environ.get('XFEAS_GEN3_END_ROBUST', '1')}; diag {self.diag or '-'}",
              flush=True)
        if self.diag:
            self._diag_setup()

    def _g4_reset(self, ep):
        self._g3_reset(ep)
        self._bases = {}
        self._feas4_now, self._fd4_now = None, None
        self.floor_sup = None
        self.ep_trial = int(ep.get("trial", 0))
        if self.diag:
            self._diag_reset(ep)

    # ------------------------------------------------------------------ the collector's 4-frame memory + feas
    def _pick_hist(self, t, x, y):
        """frames (ring entries) for the 3 history slots, the collector's distance rule; returns list oldest first"""
        ring = self.ring
        if not ring:
            return []
        dist = []
        for f in ring:
            b = self._bases.get(round(f[0], 4))
            dist.append(None if b is None else math.hypot(x - b[0], y - b[1]))
        sel = []
        moved = max([d for d in dist if d is not None] or [0.0])
        for th, tf in zip(T_HIST_V2, T_FALLBACK):
            d_tgt = V_EST * self.v_nom * abs(th)
            best = None
            if moved >= d_tgt - 0.05:
                c = [(abs(d - d_tgt), i) for i, d in enumerate(dist) if d is not None and all(ring[i] is not s for s in sel)]
                if c:
                    e, i = min(c)
                    if e <= 0.08:
                        best = ring[i]
            if best is None:
                c = [(abs(f[0] - (t + tf)), i) for i, f in enumerate(ring) if all(f is not s for s in sel)]
                if c:
                    e, i = min(c)
                    if e <= 0.3:
                        best = ring[i]
            if best is not None:
                sel.append(best)
        sel.sort(key=lambda f: f[0])
        return sel

    def _hist_obs(self, t, rgb, dep, M):
        """gen3 layer's obs memory, frames picked by the collector's distance rule (gensim_v2.decision)"""
        import cv2
        import clearmap as CLM
        bs = self._last_bs
        x, y = float(bs["pos"][0]), float(bs["pos"][1])
        old = [f for f in self.ring if t - f[0] <= 4.0]
        self.ring = old
        sel = self._pick_hist(t, x, y)
        cur = (t, rgb, dep, M)
        self.ring.append(cur); self._bases[round(t, 4)] = (x, y, float(bs["yaw"]))
        sel = [f for f in sel if f is not cur] + [cur]
        self._hist_sel = sel
        return dict(rgb_hist=np.stack([cv2.resize(f[1], (128, 128), interpolation=cv2.INTER_AREA) for f in sel]).astype(np.uint8),
                    depth_hist=np.stack([CLM.min_pool_depth(np.clip(f[2], 0, 20.0), 128) for f in sel]).astype(np.float16),
                    t_hist=np.array([f[0] - t for f in sel], np.float32), cam_hist=np.stack([f[3] for f in sel]).astype(np.float64),
                    K128=CLM.intrinsics(128))

    def _feas4(self, t, bs, hist_obs):
        """run_gen_v4.feas_v4 on the 4-frame memory: per-frame matrices and root poses, self_x 0, support floor"""
        FM = self.FM
        fz = self.floor_sup if self.floor_sup is not None else self.floor_w
        if fz is None:
            return None, None
        sel = self._hist_sel
        dh = hist_obs["depth_hist"].astype(np.float32)
        frames = [(dh[k], sel[k][3], self._bases.get(round(sel[k][0], 4))) for k in range(len(sel))]
        T_dec = (float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"]))
        Pm = FM.mem_points(frames, T_dec, float(fz), self_x=0.0)
        fv, fd, _ = FM.feas_from_points(Pm, self.probe["fp"], self.probe["caps"], rules="v04")
        return fv, fd

    def _feas_mem(self, t, bs, dep, M):
        """keeps the gen3 keyframe memory (tracker / trigger) up to date; the model / gate feas is the 4-frame one"""
        fv_long, fd_long = GM._Gen3Layer._feas_mem(self, t, bs, dep, M)
        self._feas_long = (fv_long, fd_long)
        rgb = self._last_rgb
        ho = self._hist_obs_peek(t, rgb, dep, M)
        fv, fd = self._feas4(t, bs, ho)
        if fv is None:
            fv, fd = fv_long, fd_long
        self._feas4_now, self._fd4_now = np.asarray(fv, np.float32), fd
        return fv, fd

    def _hist_obs_peek(self, t, rgb, dep, M):
        """the 4-frame obs WITHOUT adding the frame to the ring (the parent's _hist_obs call right after adds it)"""
        ring0, bases0 = list(self.ring), dict(self._bases)
        try:
            ho = self._hist_obs(t, rgb, dep, M)
        finally:
            self.ring, self._bases = ring0, bases0
        self._peek = ho
        return ho

    def _mem_update(self, t, bs):
        n0 = len(self.ring)
        GM._Gen3Layer._mem_update(self, t, bs)
        if len(self.ring) and (len(self.ring) != n0 or True):
            f = self.ring[-1]
            self._bases.setdefault(round(f[0], 4), (float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])))

    def _decide(self, obs):
        self._last_bs = obs["bs"]
        hb = os.environ.get("XFEAS_GEN4_HB", "")
        if hb and time.time() - getattr(self, "_hb_t", 0.0) > 20.0:            # queue hang-watchdog heartbeat
            self._hb_t = time.time()
            try:
                open(hb, "w").write(f"{time.time():.0f} t {float(obs['t']):.2f} x {float(obs['bs']['pos'][0]):.3f}\n")
            except Exception:
                pass
        snap = self.cam._snapshot
        box = {}

        def snap2(bs, _s=snap):
            r = _s(bs); box["rd"] = r
            self._last_rgb = np.asarray(r[0], np.uint8)
            return r
        self.cam._snapshot = snap2
        self.client.last = None
        n0 = len(self.log)
        try:
            GM._Gen3Layer._decide(self, obs)
        finally:
            self.cam._snapshot = snap
        ch = self.client.last
        if ch is None or len(self.log) <= n0:
            return
        d = self.log[-1]
        d["arm"] = self.arm
        d["feas4"] = None if self._feas4_now is None else np.round(self._feas4_now, 3).tolist()
        d["feas_long"] = np.round(self._feas_long[0], 3).tolist() if getattr(self, "_feas_long", None) else None
        d["floor_sup"] = None if self.floor_sup is None else round(self.floor_sup, 3)
        d["t_hist"] = np.round(self._peek["t_hist"], 2).tolist() if getattr(self, "_peek", None) else None
        d["model_pick"] = ch.get("model_pick")
        row = next((r for r in ch.get("table", []) if r.get("name") == ch.get("name")), None)
        d["chosen_std"] = None if row is None else row.get("std")
        d["p_clean"] = None if row is None else row.get("p_clean")
        prog = ch.get("program") or {}
        d["driver"] = dict(phases={k: {kk: v.get(kk) for kk in ("op_id", "op_label", "posture", "v_cmd", "vy_cmd", "pred") if kk in v}
                                   for k, v in (prog.get("phases") or {}).items()}, macro=prog.get("macro"))
        if self.diag:
            self._diag_decision(obs, ch, box)

    # ------------------------------------------------------------------ diagnostics (recording only)
    def _diag_setup(self):
        os.makedirs(self.diag, exist_ok=True)
        self._dx = dict(rec=None, names=None, t=0.0, tw=np.zeros(4), n_log=0)
        import chase as CH
        recs = [o for o in gc.get_objects() if isinstance(o, CH.ChaseRecorder)]
        if not recs:
            print("[gen4 diag] no recorder: decisions only", flush=True); return
        rec = recs[0]; self._dx["rec"] = rec
        g0 = rec.grab
        every = int(os.environ.get("DIAG_HEAD_EVERY", "2"))

        def grab(bs, tag=0, _g0=g0, _rec=rec):
            n = _rec.n
            hd = os.path.join(self.diag, f"head_t{self.ep_trial}")
            if bs is not None and n % every == 0:
                try:
                    import cv2
                    rgb, dep = self._snap_raw(bs)
                    os.makedirs(hd, exist_ok=True)
                    cv2.imwrite(os.path.join(hd, f"{n:06d}.jpg"), np.ascontiguousarray(np.asarray(rgb)[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 92])
                    np.save(os.path.join(hd, f"{n:06d}.npy"), np.asarray(dep, np.float16))
                except Exception as e:
                    print("[gen4 diag] head snapshot failed:", repr(e), flush=True)
            _g0(bs, tag)
            try:
                self._state_rec(n, bs)
            except Exception as e:
                print("[gen4 diag] state record failed:", repr(e), flush=True)
        rec.grab = grab
        print(f"[gen4 diag] recorder hooked (head every {every}); out {self.diag}", flush=True)

    def _snap_raw(self, bs):
        """render the decision camera WITHOUT touching the layer state (cmap mount is reset after)"""
        import cmap as CM_
        last = dict(self.cam.last) if isinstance(getattr(self.cam, "last", None), dict) else None
        fl = (self.floor, self.floor_sup)
        try:
            out = type(self.cam)._snapshot(self.cam, bs)
        finally:
            if last is not None:
                self.cam.last = last
            self.floor, self.floor_sup = fl
            CM_.set_mount(None)
        return out

    def _state_rec(self, n, bs):
        rob = self._ctx["rob"]
        art = getattr(rob, "_art", None)
        q = np.asarray(art.get_joint_positions(), float).ravel().tolist() if art is not None else []
        if self._dx["names"] is None and art is not None:
            self._dx["names"] = list(art.dof_names)
            json.dump(self._dx["names"], open(os.path.join(self.diag, "joint_names.json"), "w"))
        ap = getattr(getattr(self.vrob, "_v_inner", None), "applied", None) or {}
        sent = getattr(self.vrob, "_v_last_sent", None) or {}
        rec = dict(n=int(n), t=round(self._dx["t"], 3), x=round(float(bs["pos"][0]), 3), y=round(float(bs["pos"][1]), 3),
                   z=round(float(bs["pos"][2]), 3), roll=round(float(bs["roll"]), 3), pitch=round(float(bs["pitch"]), 3),
                   yaw=round(float(bs["yaw"]), 3), tilt=round(float(bs["tilt"]), 3),
                   vb=[round(float(v), 3) for v in np.asarray(bs["v_b"], float)], cmd=[round(float(v), 3) for v in self._dx["tw"]],
                   sent={str(k): (round(float(v), 3) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in sent.items()},
                   applied={str(k): (round(float(v), 3) if isinstance(v, (int, float, np.floating)) else str(v)) for k, v in ap.items()},
                   q=[round(v, 4) for v in q], info=str(self.last)[:120], prog=None if self.prog is None else self.prog_info.get("macro"),
                   pending=None if self.pending is None else self.pending.get("name"))
        open(os.path.join(self.diag, f"state_t{self.ep_trial}.jsonl"), "a").write(json.dumps(rec) + "\n")

    def _diag_reset(self, ep):
        for f in (f"state_t{self.ep_trial}.jsonl", f"decisions_t{self.ep_trial}.jsonl"):
            p = os.path.join(self.diag, f)
            if os.path.exists(p):
                os.remove(p)
        self._ddir = os.path.join(self.diag, f"dec_t{self.ep_trial}"); os.makedirs(self._ddir, exist_ok=True)
        self._dx["n_log"] = 0

    def _floor_check(self, M, dep, step=16):
        """gensim4.V4Mixin.floor_render_check rules from the decision camera (eye inside the head: robot colliders skipped)"""
        ctx = self._ctx
        pxq, carb, root = ctx["pxq"], ctx["carb"], ctx.get("robot_root", "/World")
        FX = 224 / 2 / math.tan(math.radians(82.25) / 2)
        eye = np.asarray(M, float)[3, :3]
        n_hit = n_ok = n_all = 0
        for v in range(112 + step // 2, 224, step):
            for u in range(step // 2, 224, step):
                dc = np.array([(u + 0.5 - 112) / FX, -(v + 0.5 - 112) / FX, -1.0, 0.0])
                dw = (dc @ np.asarray(M, float))[:3]; nrm = np.linalg.norm(dw); dw = dw / nrm
                acc, dist = 0.0, None
                for _ in range(40):
                    o = eye + dw * acc
                    h = pxq.raycast_closest(carb.Float3(*map(float, o)), carb.Float3(*map(float, dw)), float(10.0 - acc))
                    if not (h and h.get("hit")):
                        break
                    if str(h.get("collision", "")).startswith(root):
                        acc += float(h["distance"]) + 0.02
                        if acc >= 10.0:
                            break
                        continue
                    dist = acc + float(h["distance"]); break
                n_all += 1
                if dist is None:
                    continue
                if dep[v, u] >= 59.0:
                    n_all -= 1; continue
                n_hit += 1
                n_ok += int(abs(float(dep[v, u]) - dist / nrm) < 0.05 + 0.03 * dist / nrm)
        return dict(n_all=n_all, n_hit=n_hit, n_ok=n_ok, ok=bool(n_all > 0 and n_hit >= 0.5 * n_all and n_ok >= 0.8 * n_hit))

    def _diag_decision(self, obs, ch, box):
        d = self.log[-1]
        d["frame"] = int(self._dx["rec"].n) if self._dx.get("rec") is not None else None
        d["table"] = ch.get("table")
        d["sel"] = ch.get("sel"); d["stand_top"] = ch.get("stand_top"); d["lateral"] = ch.get("lateral")
        d["fd4"] = {str(k): (round(float(v), 4) if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else None)
                    for k, v in (self._fd4_now or {}).items() if isinstance(v, (int, float, np.floating))}
        d["z_base"] = round(float(obs["bs"]["pos"][2]), 3); d["yaw"] = round(float(obs["bs"]["yaw"]), 3)
        if "rd" in box:
            rgb, dep = box["rd"]
            M = np.asarray(self.cam.last["M"], np.float64)
            try:
                if len(self.log) % 8 == 1:
                    d["floor_check"] = self._floor_check(M, np.asarray(dep, np.float32))
            except Exception as e:
                d["floor_check"] = dict(error=repr(e)[:80])
            ho = getattr(self, "_peek", None) or {}
            np.savez_compressed(os.path.join(self._ddir, f"{len(self.log) - 1:04d}.npz"), rgb=np.asarray(rgb, np.uint8),
                                depth=np.asarray(dep, np.float16), M=M, base=np.array([float(obs["bs"]["pos"][0]), float(obs["bs"]["pos"][1]),
                                                                                      float(obs["bs"]["pos"][2]), float(obs["bs"]["yaw"])]),
                                feas4=np.asarray(self._feas4_now if self._feas4_now is not None else np.full(16, np.nan), np.float32),
                                t_hist=np.asarray(ho.get("t_hist", []), np.float32), cam_hist=np.asarray(ho.get("cam_hist", np.zeros((0, 4, 4)))),
                                floor_sup=np.float64(self.floor_sup if self.floor_sup is not None else np.nan))

    def _diag_flush(self):
        while self._dx["n_log"] < len(self.log):
            r = dict(self.log[self._dx["n_log"]]); r["idx"] = self._dx["n_log"]
            open(os.path.join(self.diag, f"decisions_t{self.ep_trial}.jsonl"), "a").write(json.dumps(r, default=_js) + "\n")
            self._dx["n_log"] += 1

    def _diag_act(self, obs, r):
        if self.diag:
            self._dx["t"] = float(obs["t"]); self._dx["tw"] = np.asarray(r["twist"], float)
            self._diag_flush()


def _js(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


class StraightXFeasGen4(_Gen4Layer, LN.LocalNav):
    name = "gen4_straight"
    label = "+ XFeas gen4"

    def setup(self, ctx):
        LN.LocalNav.setup(self, ctx)
        self.kind_cam = str(ctx["kind"])
        self._g4_setup(ctx)
        self.label = os.environ.get("XFEAS_LABEL", self.label)

    def reset(self, ep):
        LN.LocalNav.reset(self, ep)
        self._g4_reset(ep)

    def act(self, obs):
        tw = self._g3_tick(obs)
        if tw is not None:
            r = dict(twist=tw, info=self.last + self._mode_tag(), decisions=self.log)
        else:
            v0 = self.v
            self.v = self.v_cap
            try:
                base = straight_act(self, obs)
            finally:
                self.v = v0
            self.vrob.command(np.asarray(base["twist"], float))
            r = dict(twist=base["twist"], info=(self.last or base["info"]) + self._mode_tag(), decisions=self.log)
        self._diag_act(obs, r)
        return r


try:
    from stairs_methods import StairsStock

    class StairsXFeasGen4(_Gen4Layer, StairsStock):
        name = "gen4_stairs"
        label = "+ XFeas gen4 on the course path follower"

        def setup(self, ctx):
            StairsStock.setup(self, ctx)
            self.kind_cam = str(ctx["kind"])
            self._g4_setup(ctx)
            self.label = os.environ.get("XFEAS_LABEL", self.label)

        def reset(self, ep):
            StairsStock.reset(self, ep)
            self._g4_reset(ep)

        def setpoints(self, obs, s, k):
            return None, self.v_cap, 0.0

        def act(self, obs):
            tw = self._g3_tick(obs)
            if tw is not None:
                r = dict(twist=np.asarray(tw, float), info=self.last + self._mode_tag(), decisions=self.log)
            else:
                r0 = StairsStock.act(self, obs)
                self.vrob.command(np.asarray(r0["twist"], float))
                r = dict(twist=r0["twist"], info=(self.last or r0["info"]) + self._mode_tag(), decisions=self.log)
            self._diag_act(obs, r)
            return r
except Exception as _e:                              # outside Isaac
    print("[gen4_methods] stairs arm not loaded:", repr(_e))
