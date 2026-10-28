"""
Closed-loop arms for an XFeas-Gen checkpoint (format xfeas_gen_v1, e.g. gen1). Plugins for episode_runner
(module:Class) and for gensim.GenSim.make_method / run_method (scripts/gen/run_gen.py ARM_SPECS, run_heldout.py).
Needs scripts/gen_train, scripts/xembody, scripts/overhang on PYTHONPATH (episode_runner adds scripts/xembody).

  StraightXFeasGen       straight drive to the goal (no planner; door_methods / oh_methods style). At 4 Hz, once 2 s of
                         proprio exist, the gen selector (select_gen.Selector via gen_serve.py, CEM over the robot's OWN
                         full box, stock always a candidate, cost-aware utility) picks {v, body height, pitch, roll}.
  XFeasGenFullStack      the same layer on the map-free camera planner (local_nav.LocalNav), whose obstacle band's
                         upper limit follows the commanded body height (xfeas_fullstack.top_of, Spot geometry).

Env:
  XFEAS_GEN_CKPT     checkpoint (default data/xe_train/gen1/xfeas_gen_ens.pt)
  XFEAS_GEN_UTILITY  utility json: utility.json (plan objective, 'chosen') or utility_posture.json (posture-first, flat)
  XFEAS_GEN_HOLD     posture hold seconds (default 2.0, = oh_methods.StraightXFeasHold.HOLD_S)

Setpoint box (the box gen1 was TRAINED on, gen/gensim.py full_box, passed to the selector as the gen_v1 'box' json):
  v [max(v_min, 0.1), max_cmd vx], body_height / pitch / roll = the controller's declared setpoint_ranges
  (Spot ReLIC: 0.25-0.65 m, +-0.35, +-0.35; G1 AMO: 0.30-0.80 m, pitch -0.52..1.57, roll +-0.7), nominal = (v_nom, NOMINAL_H, 0, 0)
  with v_nom = max(0.6 max_cmd vx, v_min) = the runner's speed 0.6.

Decision rule (commit/hold of oh_methods.StraightXFeasHold, LAYER_DIAGNOSIS fix #4; no extra MARGIN gate, because the
gen utility already contains the departure trade-off and the refuse rule):
  * choice departs from the nominal posture and is not a refusal -> adopt (lowering the body at once, other posture
    dims through the 0.5 low-pass) and HOLD for XFEAS_GEN_HOLD s; speed = the chosen v;
  * while held, a stock-posture choice is ignored (held posture + held speed);
  * otherwise posture low-passes back to nominal and speed = the chosen v (speed-only changes need no hold).
  * refuse (no candidate with U > 0) -> the stock setpoint (select_gen rule), logged; the arm never stops on its own.
  Straight drive: forward speed = chosen v (may exceed the nominal speed, up to the box top, as in the training rollouts).
  Full stack: the chosen v only CAPS the planner's forward speed (the planner's DWA keeps its own speed limit).
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

PY = xfeas_paths.XFEAS_PY
DEF_CKPT = os.path.join(ROOT, "data/xe_train/gen1/xfeas_gen_ens.pt")
DEF_UTIL = os.path.join(ROOT, "data/xe_train/gen1/utility_posture.json")


def gen_box(spec, nom_h):
    """gensim.full_box + nominal, as the gen_v1 'box' json"""
    R = spec.extra.get("setpoint_ranges", {})
    box = dict(v=[max(spec.v_min, 0.1), float(spec.max_cmd[0])])
    for k in ("body_height", "pitch", "roll"):
        box[k] = list(map(float, R[k])) if k in R else [0.0, 0.0]
    if nom_h is None or "body_height" not in R:
        box["body_height"] = [0.0, 0.0]
    nominal = dict(v=max(0.6 * spec.max_cmd[0], spec.v_min), body_height=nom_h if nom_h is not None else 0.0, pitch=0.0, roll=0.0)
    return dict(box=box, nominal=nominal)


class GenDecider:
    """head camera + proprio buffer of xfeas_decider.XFeasDecider (identical camera / buffer), gen sidecar"""

    def __init__(self):
        self.ckpt = os.path.abspath(os.environ.get("XFEAS_GEN_CKPT") or DEF_CKPT)
        self.util = os.path.abspath(os.environ.get("XFEAS_GEN_UTILITY") or DEF_UTIL)
        self.srv = None

    def setup(self, stage, rep, world, rob, spec, kind):
        from xfeas_decider import XFeasDecider
        self.base = XFeasDecider.__new__(XFeasDecider)            # reuse its camera / buffer / snapshot (no v8 sidecar)
        self.base.ckpt, self.base.log, self.base.srv = self.ckpt, [], None
        sp = subprocess.Popen                                       # do not start select_xe: patch Popen for this call
        try:
            subprocess.Popen = lambda *a, **k: None
            self.base.setup(stage, rep, world, rob, spec, kind)
        finally:
            subprocess.Popen = sp
        self.srv = subprocess.Popen([PY, os.path.join(THIS, "gen_serve.py"), "--ckpt", self.ckpt, "--utility", self.util],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self.buf, self.kind = self.base.buf, kind
        print(f"[gen] decider ready ({os.path.basename(self.ckpt)}, utility {os.path.basename(self.util)})", flush=True)

    def reset(self):
        self.base.buf.reset()
        self.desc = None

    def snapshot(self, bs):
        return self.base._snapshot(bs)

    def _rd(self, n):
        b = b""
        while len(b) < n:
            c = self.srv.stdout.read(n - len(b))
            if not c:
                raise RuntimeError("gen sidecar died")
            b += c
        return b

    def ask(self, rgb, dep, pb, pj, desc, box_json, seed):
        req = io.BytesIO()
        np.savez(req, rgb=rgb, depth=dep, prop_base=pb, prop_joints=pj, desc=json.dumps(desc), box=box_json,
                 robot=str(self.kind), task=json.dumps(dict(seed=int(seed), setpoint={})))
        b = req.getvalue()
        self.srv.stdin.write(struct.pack("<I", len(b))); self.srv.stdin.write(b); self.srv.stdin.flush()
        r = np.load(io.BytesIO(self._rd(struct.unpack("<I", self._rd(4))[0])), allow_pickle=False)
        return json.loads(str(r["choice"]))

    def __del__(self):
        try:
            if self.srv is not None:
                self.srv.stdin.close(); self.srv.wait(timeout=10)
        except Exception:
            pass


class _GenLayer:
    PERIOD = 0.25
    POST_EPS = 1e-3

    def _gen_setup(self, ctx):
        self.gdec = GenDecider()
        self.gdec.setup(stage=ctx["stage"], rep=ctx["rep"], world=ctx["world"], rob=ctx["rob"], spec=ctx["spec"], kind=ctx["kind"])
        rob, R = ctx["rob"], ctx["spec"].extra.get("setpoint_ranges", {})
        self.nom = getattr(sys.modules.get(type(rob).__module__), "NOMINAL_H", None)
        self.posture = self.nom is not None and "body_height" in R
        self.gbox = gen_box(ctx["spec"], self.nom if self.posture else None)
        self.box_json = json.dumps(self.gbox)
        self.HOLD_S = float(os.environ.get("XFEAS_GEN_HOLD", "2.0"))
        print(f"[gen] box {self.box_json} posture {'ON' if self.posture else 'OFF'} hold {self.HOLD_S}s", flush=True)

    def _gen_reset(self):
        self.gdec.reset()
        self.cap, self.t_next, self.log, self.last = None, 2.0, [], ""
        self.sp_now = dict(dh=0.0, pitch=0.0, roll=0.0)
        self.hold_until, self.v_held = -1.0, None
        rob = self.ctx["rob"]
        if self.posture and hasattr(rob, "reset_setpoints"):
            rob.reset_setpoints()

    def _gen_tick(self, obs):
        """4 Hz decision; sets self.cap (forward speed) and the posture setpoints"""
        t, bs, rob = obs["t"], obs["bs"], obs["rob"]
        g = self.gdec
        g.buf.tick(rob, bs)
        if g.desc is None:
            g.desc = rob.descriptor(self.ctx["stage"])
        if not (t >= self.t_next and g.buf.full()):
            return
        self.t_next = t + self.PERIOD
        rgb, dep = g.snapshot(bs)
        pb, pj = g.buf.arrays()
        ch = g.ask(rgb, dep, pb, pj, g.desc, self.box_json, seed=int(t * 10))
        v_nom = self.gbox["nominal"]["v"]
        dep_post = (not ch["refuse"]) and self.posture and any(abs(float(ch[k])) > self.POST_EPS for k in ("dh", "pitch", "roll"))
        if dep_post:
            self.hold_until = t + self.HOLD_S
            for k in ("dh", "pitch", "roll"):
                tgt = float(ch[k])
                if k == "dh" and tgt < self.sp_now[k]:
                    self.sp_now[k] = tgt                                  # lowering: at once
                else:
                    self.sp_now[k] += 0.5 * (tgt - self.sp_now[k])
            self.cap = self.v_held = float(ch["v"]); state = "new"
        elif t < self.hold_until:
            self.cap = self.v_held if self.v_held is not None else v_nom; state = "hold"
        else:
            for k in ("dh", "pitch", "roll"):
                self.sp_now[k] += 0.5 * (0.0 - self.sp_now[k])
            self.cap = v_nom if ch["refuse"] else float(ch["v"]); state = "refuse" if ch["refuse"] else "stockpost"
        bh = (self.nom + self.sp_now["dh"]) if self.posture else None
        if self.posture:
            rob.command(dict(body_height=bh, pitch=self.sp_now["pitch"], roll=self.sp_now["roll"]))
        self.log.append(dict(t=round(t, 2), x=round(float(bs["pos"][0]), 2), y=round(float(bs["pos"][1]), 2), v=round(self.cap, 3),
                             dh=round(self.sp_now["dh"], 3), pitch=round(self.sp_now["pitch"], 3), roll=round(self.sp_now["roll"], 3),
                             ch_v=ch["v"], ch_dh=ch["dh"], ch_pitch=ch["pitch"], ch_roll=ch["roll"], p=ch["p_success"],
                             p_std=ch["p_std"], p_stock=ch["p_stock"], u=ch["utility"], u_stock=ch["u_stock"], cost=ch["cost"],
                             refuse=ch["refuse"], state=state, body_height=None if bh is None else round(bh, 3)))
        self.last = (f"XFeas-Gen[{state}]: v {self.cap:.2f}  dh {self.sp_now['dh']:+.2f}  pitch {self.sp_now['pitch']:+.2f}  "
                     f"P={ch['p_success']:.2f} (stock {ch['p_stock']:.2f})")


class StraightXFeasGen(_GenLayer, LN.LocalNav):
    name = "xfeasgen_straight"
    label = "+ XFeas-Gen layer (gen1): speed + posture setpoints (straight drive)"

    def setup(self, ctx):
        LN.LocalNav.setup(self, ctx)
        self._gen_setup(ctx)
        u = os.path.basename(self.gdec.util)
        self.label = f"+ XFeas-Gen gen1 ({'posture-first' if 'posture' in u else 'plan objective'} utility): speed + posture (straight drive)"

    def reset(self, ep):
        LN.LocalNav.reset(self, ep)
        self._gen_reset()

    def act(self, obs):
        self._gen_tick(obs)
        v0 = self.v
        if self.cap is not None:
            self.v = self.cap                                   # straight drive at the chosen speed
        try:
            base = straight_act(self, obs)
        finally:
            self.v = v0
        return dict(twist=base["twist"], info=self.last or base["info"], decisions=self.log)


class XFeasGenFullStack(_GenLayer, LN.LocalNav):
    name = "xfeasgen_fullstack"
    label = "+ XFeas-Gen layer (gen1): posture setpoints + posture-aware camera planner (full stack)"

    def setup(self, ctx):
        import xfeas_fullstack as FS
        self.FS = FS
        LN.LocalNav.setup(self, ctx)
        self._gen_setup(ctx)
        self.band_stock = tuple(self.laser.band)
        hi = ctx["spec"].extra.get("setpoint_ranges", {}).get("body_height", [FS.H_LOW, FS.H_NOM])[1]
        self.laser.band = (self.band_stock[0], max(self.band_stock[1], FS.top_of(hi) + FS.TOP_MARGIN))
        u = os.path.basename(self.gdec.util)
        self.label = f"+ XFeas-Gen gen1 ({'posture-first' if 'posture' in u else 'plan objective'} utility) + posture-aware camera planner (full stack)"
        print(f"[gen-fullstack] stock band {self.band_stock}; scan band {self.laser.band}", flush=True)

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
