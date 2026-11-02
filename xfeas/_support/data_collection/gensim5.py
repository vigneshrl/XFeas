"""
gen_v5 harness: COMMAND-SWITCH ROLLOUTS (import AFTER SimulationApp exists). analysis/GEN5_DATA_SPEC.md 5.

V5Mixin (MRO: V5Mixin, gensim4.V4Mixin, gensim3.V3Mixin, gensim_v2.V2Mixin, GenSimV1 | EnvSim) adds on top of gen_v4
(shared head camera, exact self mask, contacts, mode-aware fall judge, robot-skipping ray checks, real-asset builders, seq):
  * ContactLog5: the gen_v4 contact log + per-0.02 s-bin impulse arrays (hazard body / hazard foot / scene body), so the
    label 'hazard contact within 1.5 s' can be computed for every tick;
  * the 'jog' family (jog5.JogScene) and a path polyline for every situation (straight lanes: the lane axis; jogs and
    terrain_jog: the bent path) followed by the stock path follower (path_follower);
  * episode5(): one episode = spawn s_spawn on the path, 1.5 s NOMINAL warm-up (frame history), then every 0.25 s a
    DECISION TICK: render the shared head camera, 4-frame feas (v0.4 + memory + minpool, as gen_v4), the candidate set
    (candidates) + descriptors (descriptors) given the command in force and the measured speed, the behaviour policy
    (policy5) picks a candidate, the command is applied (steering always from the follower); runs to the path goal / a
    fall (mode-aware) / a stall (no 0.1 m progress for STALL_S) / timeout. Everything per tick is logged (episode npz).
GenSimV5 = school; make_envsim_v5() = Isaac environments / terrain.
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import json, math, os, re, sys, time
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = xfeas.paths.ROOT
import gensim as GS
import gensim_v1 as GV
import gensim_v2 as G2
import gensim3 as G3
import gensim4 as G4
import ctrl_spec as CS
import virtual as VC
import virtual5 as V5
import candidates as CA
import descriptors as DS
import policy5 as PO
import path_follower as ST

TICK_S = 0.25
WARM_S = 1.5
HIST_TICKS = (-6, -4, -2, 0)  # = gen_v2 T_HIST (-1.5, -1.0, -0.5, 0) s at 0.25 s ticks
NEAR_M = 2.5
COMMIT_REACH = dict(
    g1=(0.25, 0.35), spot=(0.90, 0.60)
)  # (front, rear) reach of the body from the base along the path (m);
# Spot: the stowed arm reaches ~0.9 m ahead (probe depth 1.59 m)
STALL_END_G1, STALL_END_SPOT = 10.0, 8.0  # episode ends after this long without 0.1 m path progress (gen_v4: 8 / 5 s);
# longer than gen_v4 so the random policy gets more attempts at an obstacle


_CL4 = getattr(G4, "_CL4_ORIG", G4.ContactLog4)
G4._CL4_ORIG = _CL4


class ContactLog5(_CL4):
    def _reset(self):
        _CL4._reset(self)
        self.B = dict(hb={}, hf={}, sb={})

    def _on(self, headers, data):
        if not self.active:
            return
        from pxr import PhysicsSchemaTools

        t = float(getattr(self.sim.rob, "_v_t", 0.0)) - self.t0
        b = int(t / G4.CONTACT_BIN)
        A = self.A
        Bn = self.B
        for h in headers:
            ra = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
            rb = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
            if ra.startswith(self.root) and not rb.startswith(self.root):
                link, other = ra.rsplit("/", 1)[-1], str(PhysicsSchemaTools.intToSdfPath(h.collider1))
            elif rb.startswith(self.root) and not ra.startswith(self.root):
                link, other = rb.rsplit("/", 1)[-1], str(PhysicsSchemaTools.intToSdfPath(h.collider0))
            else:
                continue
            imp = 0.0
            for k in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                im = data[k].impulse
                imp += math.sqrt(im[0] ** 2 + im[1] ** 2 + im[2] ** 2)
            A["n_ev"] += 1
            if imp < G4.IMP_MIN:
                continue
            haz = other.startswith(GS.OURS)
            foot = bool(G4.FOOT_RE.search(link))
            if haz and not foot:
                A["haz_body"].add(b)
                A["haz_body_imp"] += imp
                A["links"][link] = A["links"].get(link, 0.0) + imp
                Bn["hb"][b] = Bn["hb"].get(b, 0.0) + imp
                if A["t_first"] is None:
                    A["t_first"] = round(t, 2)
            elif haz:
                A["haz_foot"].add(b)
                A["haz_foot_imp"] += imp
                Bn["hf"][b] = Bn["hf"].get(b, 0.0) + imp
            elif not foot:
                A["scene_body"].add(b)
                A["scene_body_imp"] += imp
                Bn["sb"][b] = Bn["sb"].get(b, 0.0) + imp

    def bins(self, n):
        out = {}
        for k, D in self.B.items():
            a = np.zeros(n, np.float32)
            for b, v in D.items():
                if 0 <= b < n:
                    a[b] += v
            out[k] = a
        return out


def feas_v4(frames, base_dec, fz, P):
    """gen_v4 feas recipe (run_gen_v4.feas_v4) on the tick's 4-frame memory. frames: [(depth128 f32, M 4x4, (x, y, yaw))]"""
    import feas_mem as FM

    Pm = FM.mem_points(frames, base_dec, fz, self_x=0.0)
    fv4, fd4, cm4 = FM.feas_from_points(Pm, P["fp"], P["caps"], rules="v04")
    return fv4, fd4


class V5Mixin:
    v5 = True

    def setup_v5(self, base_spec):
        G4.ContactLog4 = ContactLog5  # the gen_v4 setup instantiates the module-level class
        if os.environ.get("GEN5_DEBUG_NOCONTACT", "0") == "1":  # DEBUG ONLY (never in data)

            class _NoCL:
                def __init__(self, sim):
                    self.A = {}

                def begin(self):
                    pass

                def end(self):
                    return dict(
                        haz_body_s=0.0,
                        haz_body_Ns=0.0,
                        haz_foot_s=0.0,
                        haz_foot_Ns=0.0,
                        scene_body_s=0.0,
                        scene_body_Ns=0.0,
                        haz_links={},
                        t_first_haz_body=None,
                        n_events=0,
                    )

                def bins(self, n):
                    return dict(hb=np.zeros(n, np.float32), hf=np.zeros(n, np.float32), sb=np.zeros(n, np.float32))

            G4.ContactLog4 = _NoCL
        if os.environ.get("GEN5_DEBUG_V3SETUP", "0") == "1":  # DEBUG ONLY (never in data): gen3 setup, no v4 extras
            self.setup_v3(base_spec)
            self.cam4 = type("C", (), dict(src="shared", mount="g1_head_php"))()
            self.contacts = (
                G4.ContactLog4(self)
                if False
                else type(
                    "NoCL",
                    (),
                    dict(
                        begin=lambda s: None,
                        end=lambda s: dict(
                            haz_body_s=0.0,
                            haz_body_Ns=0.0,
                            haz_foot_s=0.0,
                            haz_foot_Ns=0.0,
                            scene_body_s=0.0,
                            scene_body_Ns=0.0,
                            haz_links={},
                            t_first_haz_body=None,
                            n_events=0,
                        ),
                        bins=lambda s, n: dict(
                            hb=np.zeros(n, np.float32), hf=np.zeros(n, np.float32), sb=np.zeros(n, np.float32)
                        ),
                    ),
                )()
            )
            self.judge = type("J", (), dict(reset=lambda s: None, reason=None, legacy_fired=False, n_low_tilt=0))()
            self._snap_info = {}
            self.seq = None
            self.jog = None
            return
        self.setup_v4(base_spec)
        self.jog = None
        if self.kind == "g1p" and getattr(self, "mcam", None) is not None:
            # PHP drives its joints in EFFORT mode with drive gains zeroed at runtime; the shared mount hides the torso_link
            # RIGID-BODY prim during our renders (its visuals are instanceable), and that visibility toggle on a physics
            # body made PHP fall 1.2 s into every episode (smoke 10-23: 5/5 falls with renders, 0 without). Hide only the
            # visuals child for PHP (the same prim PHP's own depth render hides).
            from pxr import UsdGeom

            vp = self.stage.GetPrimAtPath(self.root + "/pelvis/torso_link/visuals")
            if vp.IsValid():
                self.mcam.hide = [UsdGeom.Imageable(vp)]
                print(f"[v5] g1p: head-camera hide list -> {vp.GetPath()} (not the torso_link body)", flush=True)

    # ------------------------------------------------------------------ build: + jog, + path / hazard list
    def build(self, sit):
        if self.jog is not None:
            self.jog.clear()
        fam = sit["family"]
        if fam == "jog":
            import jog5 as JG

            if self.seq is not None:
                self.seq.clear()
            if self.jog is None:
                self.jog = JG.JogScene(self)
            geo = self.jog.build(sit)
        else:
            geo = G4.V4Mixin.build(self, sit)
        return self._path_geo(sit, geo)

    def park_scene(self):
        if self.jog is not None:
            self.jog.clear()
        return G4.V4Mixin.park_scene(self)

    def _path_geo(self, sit, geo):
        L = float(sit["lane"]["len"])
        P0, d, n = geo["P0"], geo["d"], geo["n"]
        if "path_world" not in geo:
            if sit.get("jog") and sit["family5"] == "terrain_jog":
                import jog5 as JG

                J = sit["jog"]
                poly = JG.jog_polyline(L, float(J["s_bend"]), float(J["theta_deg"]), float(J["Lj"]), float(J["side"]))
                geo["path_lane"] = [[round(s, 4), round(l, 4)] for s, l in poly]
                geo["path_world"] = [[float(x) for x in (P0 + d * s + n * l)] for s, l in poly]
            else:
                geo["path_lane"] = [[0.0, 0.0], [L, 0.0]]
                geo["path_world"] = [[float(x) for x in P0], [float(x) for x in (P0 + d * L)]]
        if "hazards" not in geo:
            Hz = []
            if sit["family"] == "seq":
                for e in geo["checks"].get("elements", []):
                    Hz.append(dict(kind=e["kind"], arc0=float(e["s_front"]), arc1=float(e["s_end"])))
            elif sit["family5"] not in ("null",):
                Hz.append(
                    dict(kind=sit["family5"], arc0=float(geo["s_front"]), arc1=float(geo.get("s_end", geo["s_front"])))
                )
            if sit.get("jog") and sit["family5"] == "terrain_jog":
                J = sit["jog"]
                Hz.append(
                    dict(
                        kind=f"bend{int(J['theta_deg'])}",
                        arc0=float(J["s_bend"]),
                        arc1=float(J["s_bend"]) + float(J["Lj"]),
                    )
                )
            geo["hazards"] = Hz
        if "arc_goal" not in geo:
            pf = ST.PathFollower(geo["path_world"])
            if sit["family"] == "seq":
                ends = [h["arc1"] for h in geo["hazards"]]
                g = min(max(ends) + 0.6, pf.total - 0.25)
            elif geo.get("on_top"):
                g = float(geo["s_goal"])  # on the platform / landing (gen_v4 goal, checked against z_top)
            elif sit["family5"] == "terrain_jog":
                g = pf.total - 0.4
            else:
                g = min(float(geo["s_goal"]) + 1.0, pf.total - 0.3)
                g = max(g, float(geo["s_goal"]))
            geo["arc_goal"] = round(float(g), 3)
        return geo

    # ------------------------------------------------------------------ one switching episode
    def episode5(self, sit, geo, ctl, P, light, rng, vid=None, log=print):
        spec, C, D = ctl["spec"], ctl["cands"], ctl["desc"]
        rob = self.rob
        pf = ST.PathFollower(geo["path_world"])
        fz = float(geo["fz"])
        HZ, EVERY = self.HZ, self.EVERY
        arc_sp = float(sit["s_spawn"])
        q = pf.point(arc_sp)
        _, _, seg = pf.project(q)
        hd = pf.heading(seg)
        nvec = np.array([-math.sin(hd), math.cos(hd)])
        q = q + nvec * float(sit.get("lat0", 0.0))
        # audit view of the hazard with the robot parked (gen_v2 decision recipe)
        vis_check = None
        if geo.get("aim") is not None:
            self.rob.teleport(q[0], q[1], hd, fz - 30.0)
            self.world.step(render=False)
            low = geo.get("family") in ("overhang", "compound") or (geo.get("family") == "null" and geo["aim"][2] > 0.6)
            vis_check = self.check_view(geo, 1.6, h=0.25) if low else self.check_view(geo, 2.5)
        g1 = self.kind.startswith("g1")
        self.settle(
            q[0],
            q[1],
            hd + float(sit.get("yaw0", 0.0)),
            fz,
            float(os.environ.get("GEN5_DEBUG_SETTLE", 2.0 if g1 else 1.0)),
        )
        bs = rob.base_state()
        if bs is None or rob.fallen(bs):
            raise GS.Reject("bad_spawn")
        self.stand_z = float(bs["pos"][2]) - fz
        self.judge.reset()
        self.contacts.begin()
        rob._v_trace = []
        hist_prefix = list(rob._v_hist)
        t_clock0 = float(rob._v_t)
        PS_b, PS_j = (
            [np.asarray(x, np.float32) for x in self.buf.base],
            [np.asarray(x, np.float32) for x in self.buf.joints],
        )
        last_ref = self.buf.base[-1] if self.buf.base else None
        nom = CA.nominal_cmd(spec)
        cur = dict(nom)
        rob.set_env_intent(float(nom.get("lin_vel_x", 0.3) or 0.3), 9.9, 9.9, 0.0)
        rob.command(CA.to_names(spec, cur))
        X0 = D.describe(C, cur, 0.0)
        style = "persist" if rng.random() < 0.5 else "spec"
        p_commit = float(sit.get("pol_commit", 0.0) or 0.0)  # crawl re-collection only (policy5 COMMIT); 0 = unchanged
        if p_commit > 0:
            pol = PO.BehaviourPolicy5(
                C,
                X0,
                rng,
                style=style,
                aggr=bool(sit.get("aggr")),
                commit=p_commit,
                front_reach=COMMIT_REACH["g1" if self.kind.startswith("g1") else "spot"][0],
                rear_reach=COMMIT_REACH["g1" if self.kind.startswith("g1") else "spot"][1],
            )
        else:
            pol = PO.BehaviourPolicy5(C, X0, rng, style=style, aggr=bool(sit.get("aggr")))
        moving = lambda c: bool((CS.mode_info(spec, c["mode"]) or {}).get("moving", True)) if "mode" in c else True
        resend = bool(spec.get("binding", {}).get("resend_every_step")) or bool(
            self.base_spec.get("binding", {}).get("resend_every_step")
        )
        t_max = float(min(self.T_CAP, max(12.0, (geo["arc_goal"] - arc_sp) / self.V_FLOOR + 6.0)))
        n_steps = int(t_max * HZ)
        T = []  # per tick records
        frames = []  # per tick frames
        best, t_best = -1e9, 0.0
        end = None
        t_next = 0.0
        k_tick = 0
        first_dec = True
        floor_render = None
        vis_dec = None
        haz = geo.get("hazards", [])
        expo = None
        t = 0.0
        prof = dict(phys=0.0, render=0.0, feas=0.0, desc=0.0, ext=0.0)
        _tp = time.time()
        for step in range(n_steps):
            t = (step + 1) / HZ
            self.world.step(render=False)
            self.buf.tick(rob)
            if self.buf.base and self.buf.base[-1] is not last_ref:
                last_ref = self.buf.base[-1]
                PS_b.append(np.asarray(self.buf.base[-1], np.float32))
                PS_j.append(np.asarray(self.buf.joints[-1], np.float32))
            is_tick = t + 1e-9 >= t_next
            if (step + 1) % EVERY and not is_tick:
                continue
            bs = rob.base_state()
            if bs is None or rob.fallen(bs):
                end = dict(
                    reason="fell", t=t, fall_reason=self.judge.reason or ("no_state" if bs is None else "legacy")
                )
                break
            er = pf.errors(bs["pos"][:2], float(bs["yaw"]))
            sc = ST.follower_cmd(spec, None, er, moving=moving(cur))
            if sc:
                rob.command(sc)
            if resend and os.environ.get("GEN5_DEBUG_NORESEND", "0") != "1":
                rob._v_inner.command(dict(rob._v_last_sent or {}))
            if er["s"] > best + 0.10:
                best, t_best = er["s"], t
            if er["s"] >= geo["arc_goal"] and abs(er["e"]) < 0.5:
                if geo.get("z_top") is not None and float(bs["pos"][2]) - geo["z_top"] < 0.25:
                    end = dict(reason="crossed_low", t=t)
                else:
                    end = dict(reason="success", t=t)
                break
            if t - t_best > (STALL_END_G1 if g1 else STALL_END_SPOT):
                end = dict(reason="stall", t=t)
                break
            if not is_tick:
                continue
            # ------------------------------------------------ TICK
            t_next += TICK_S
            _t1 = time.time()
            prof["phys"] += _t1 - _tp
            _dbg = os.environ.get("GEN5_DEBUG_NOSNAP", "0")
            if _dbg == "2":  # DEBUG ONLY: never render the head camera
                snap = (np.zeros((224, 224, 3), np.uint8) + 128, np.full((224, 224), 5.0, np.float32), np.eye(4))
            elif k_tick == 0:
                snap = self.snapshot(bs)
                expo = getattr(self, "expo", None)
                snap, light, relit = self._relight_if_black(snap, bs, light)
            elif (
                os.environ.get("GEN5_DEBUG_NOSNAP", "0") == "1" and frames
            ):  # DEBUG ONLY (never in data): no head render
                snap = (frames[-1]["rgb"], frames[-1]["dep"].astype(np.float32), frames[-1]["M"])
            else:
                snap = self._snap_raw(bs)
            if snap is None:
                raise GS.Reject(f"render_failed tick {k_tick}")
            rgb, dep, M = snap
            if os.environ.get("GEN5_DEBUG_PHPREFRESH", "0") == "1" and hasattr(rob._v_inner, "_maybe_depth"):
                rob._v_inner._maybe_depth(force=True)  # DEBUG: PHP re-renders its own depth now
            _t2 = time.time()
            prof["render"] += _t2 - _t1
            info = dict(getattr(self, "snap_info", None) or {})
            dep = np.nan_to_num(np.asarray(dep, np.float32), posinf=60.0)
            import clearmap as CMg

            d128 = CMg.min_pool_depth(np.clip(dep, 0, 20.0), 128).astype(np.float32)
            frames.append(
                dict(
                    rgb=np.asarray(rgb, np.uint8),
                    dep=dep.astype(np.float16),
                    d128=d128.astype(np.float16),
                    M=np.asarray(M, np.float64),
                    mask=float(info.get("mask_frac", np.nan)),
                    cam_h=float(info.get("cam_h") if info.get("cam_h") is not None else np.nan),
                    bw=(float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])),
                )
            )
            dec = t >= WARM_S - 1e-6
            rec = dict(
                t=round(t, 4),
                k=k_tick,
                dec=dec,
                arc=er["s"],
                e=er["e"],
                herr=er["herr"],
                pos=[float(x) for x in bs["pos"]],
                yaw=float(bs["yaw"]),
                pitch=float(bs["pitch"]),
                roll=float(bs["roll"]),
                v_b=[float(x) for x in bs["v_b"]],
                w_b=[float(x) for x in bs["w_b"]],
                tilt=float(bs["tilt"]),
                hist_n=len(hist_prefix) + len(rob._v_trace or []),
                prop_n=len(PS_b),
                glitch=V5.glitch_count(rob),
                mean_rgb=round(float(rgb.mean()), 1),
            )
            _t3 = time.time()
            try:
                rec["top"] = float(self.link_extents(bs)["top_abs"]) - fz
            except Exception:
                rec["top"] = float("nan")
            prof["ext"] += time.time() - _t3
            dist = min([h["arc0"] - er["s"] for h in haz if er["s"] <= h["arc1"] + 0.5] or [99.0])
            near = any(h["arc0"] - NEAR_M <= er["s"] <= h["arc1"] + 0.3 for h in haz)
            rec.update(d_haz=round(float(dist), 3), near=bool(near))
            if dec:
                if first_dec and _dbg == "2":
                    floor_render = dict(ok=True, debug=True)
                    first_dec = False
                if first_dec:
                    floor_render = self.floor_render_check(M, dep)
                    if not floor_render["ok"]:
                        if os.environ.get("GEN5_FLOOR_DUMP"):  # diagnosis only: frame + camera of the failed check
                            np.savez_compressed(
                                os.environ["GEN5_FLOOR_DUMP"] + f"_{k_tick}.npz",
                                M=M,
                                dep=dep,
                                rgb=np.asarray(rgb, np.uint8),
                                pos=np.asarray(bs["pos"], float),
                                fz=float(fz),
                            )
                        raise GS.Reject(f"floor_not_rendered {floor_render}")
                    try:
                        vis_dec = self.visible(M, dep)
                    except Exception:
                        vis_dec = None
                    first_dec = False
                idx = [max(0, len(frames) - 1 + o) for o in HIST_TICKS]
                fr = [(frames[i]["d128"].astype(np.float32), frames[i]["M"], frames[i]["bw"]) for i in idx]
                _t4 = time.time()
                try:
                    fv, fd = feas_v4(fr, (float(bs["pos"][0]), float(bs["pos"][1]), float(bs["yaw"])), fz, P)
                except Exception as ex:
                    fv, fd = np.full(16, np.nan, np.float32), dict(error=repr(ex)[:200])
                _t5 = time.time()
                prof["feas"] += _t5 - _t4
                X = D.describe(C, cur, float(bs["v_b"][0]))
                prof["desc"] += time.time() - _t5
                if p_commit > 0:
                    hz_c = [h for h in haz if er["s"] <= h["arc1"] + 0.5]
                    d_end = (min(hz_c, key=lambda h: h["arc0"])["arc1"] - er["s"]) if hz_c else None
                    ci, pinfo = pol.choose(near, d_front=float(dist), d_end=d_end, t=t)
                else:
                    ci, pinfo = pol.choose(near)
                c = C[ci]
                prev = dict(cur)
                if c["kind"] != "KEEP":
                    cur = dict(c["cmd"])
                    rob.command(CA.to_names(spec, cur))
                    sc = ST.follower_cmd(spec, None, er, moving=moving(cur))
                    if sc:
                        rob.command(sc)
                rec.update(
                    chosen=int(ci),
                    pol_mode=pinfo["mode"],
                    prop=round(float(pinfo["prop"]), 6),
                    cls_prev=CA.transition_class(spec, prev),
                    cls_new=CA.transition_class(spec, cur),
                    switched=not CA.same_cmd(prev, cur),
                    feas=np.asarray(fv, np.float32),
                    feas_json=fd,
                    X=X.astype(np.float16),
                )
            rec["cmd"] = dict(cur)
            rec["state"] = {k_: (v_ if isinstance(v_, str) else round(float(v_), 4)) for k_, v_ in rob._v_state.items()}
            rec["sent"] = {
                k_: (v_ if isinstance(v_, str) else round(float(v_), 4)) for k_, v_ in (rob._v_last_sent or {}).items()
            }
            T.append(rec)
            if vid is not None and k_tick % 2 == 0:
                self.side_frame(vid, bs, geo)
            k_tick += 1
            _tp = time.time()
        if end is None:
            end = dict(reason="timeout", t=t)
        rob.stop()
        ct = self.contacts.end()
        nb = int(t / G4.CONTACT_BIN) + 2
        bins = self.contacts.bins(nb)
        trace = list(rob._v_trace or [])
        rob._v_trace = None
        R = hist_prefix + trace
        out = dict(
            ticks=T,
            frames=frames,
            end=end,
            contact=ct,
            bins=bins,
            records=R,
            n_prefix=len(hist_prefix),
            t_clock0=t_clock0,
            prop_base=np.stack(PS_b) if PS_b else np.zeros((0, 13), np.float32),
            prop_joints=np.stack(PS_j) if PS_j else np.zeros((0, 0, 7), np.float32),
            style=style,
            vis_check=vis_check,
            vis_dec=vis_dec,
            floor_render=floor_render,
            light=light,
            expo=expo,
            fall_legacy=bool(self.judge.legacy_fired),
            n_low_tilt_exempt=int(self.judge.n_low_tilt),
            t_max=t_max,
            prof={k_: round(v_, 1) for k_, v_ in prof.items()},
            aggr_set=[int(i) for i in np.where(pol.aggr_set)[0]],
            posture_set=[int(i) for i in np.where(pol.posture)[0]],
            pol_commit=dict(
                p=p_commit,
                commit_d=pol.commit_d if p_commit > 0 else None,
                log=list(pol.commit_log) if p_commit > 0 else [],
                lowmove=[int(i) for i in np.where(pol.lowmove)[0]] if p_commit > 0 else [],
            ),
        )
        return out


class GenSimV5(V5Mixin, G4.V4Mixin, G3.V3Mixin, G2.V2Mixin, GV.GenSimV1):
    pass


def make_envsim_v5():
    import envsim as ES

    class EnvSimV5(V5Mixin, G4.V4Mixin, G3.V3Mixin, G2.V2Mixin, ES.EnvSim):
        pass

    return EnvSimV5
