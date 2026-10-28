#!/usr/bin/env python
"""
Camera check sheet for the gen5 wheeled / aerial mounts (camera_mount jackal_front / iris_front), in the gen5 harness itself
(GenSimV5, school, a training lane, night_white headlight + one day frame). Per pose:
  * the layer RGB / masked depth / robot mask (exactly what episode5 records) and the same render with the mask off;
  * a chase view with the eye projected (red dot) and the optical axis (red line) -- drawn on the image only;
  * numbers: eye in the mount link frame, the body's front face at the eye height (visual mesh points within |y| < 3 cm,
    |z - z_eye| < 3 cm) and the eye - face gap (must be 0 .. +2 cm: not inside the body, not floating), eye above the floor,
    optical-axis elevation, robot pixel fraction, floor render check (PhysX rays vs rendered depth).
  scripts/gen5/run_isaac5.sh 0 scripts/gen5/cam_check5x.py --base jackal --out analysis/camera_check_v5_xbody/jackal
"""
import argparse, json, math, os, sys, time
import numpy as np

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS, "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.path.insert(0, THIS)
import ctrl5 as K5
os.makedirs(a.out, exist_ok=True)
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 224, "height": 224, "renderer": "RaytracedLighting"})
import gensim5 as G5
import ctrl_spec as CS
import candidates5 as CA
import camera_mount as CMT
import cv2

G = G5.GenSimV5(app, K5.FACT[a.base], a.base, seed=3, side_cam=True, env_id="school")
BASE = CS.load(K5.spec_path(a.base))
G.setup_v5(BASE)
G.park_scene()
lanes = json.load(open(os.path.join(ROOT, "data", "gen_v5", "suites", "school_spotp_p1.json")))
lane = next(s["lane"] for s in lanes if s["family5"] == "null" and s["lane"].get("split") == "train")
P0, yaw, d, n, fz = G.lane_frame(lane)
fz = G.floor_at(P0 + d * 1.0, fz)
mc = G.mcam
geo = mc.geometry()
i_link = mc._names.index(mc.link)
eye_l, Rm = CMT.mount_in_link(mc.mount)
Pl = geo[i_link]
band = Pl[(np.abs(Pl[:, 1] - eye_l[1]) < 0.03) & (np.abs(Pl[:, 2] - eye_l[2]) < 0.03)]
face_x = float(band[:, 0].max()) if len(band) else float("nan")
all_x = float(np.concatenate([g for g in geo if g is not None])[:, 0].max())
print(f"[chk] {a.base} mount {mc.mount} link {mc.link} eye {eye_l.tolist()} face_x(eye band) {face_x:.4f} gap {eye_l[0] - face_x:+.4f} "
      f"(n {len(band)}), body max x {all_x:.4f}", flush=True)
nom = CA.nominal_cmd(BASE)
poses = [("stand_night", {}, "night_white", 0.0), ("moving_night", {"lin_vel_x": nom.get("lin_vel_x", 1.0)}, "night_white", 1.5),
         ("stand_day", {}, "original", 0.0)]
if CS.has(BASE, "pelvis_height"):
    poses += [("alt0.73_night", {"pelvis_height": 0.733}, "night_white", 0.0), ("alt0.35_night", {"pelvis_height": 0.35}, "night_white", 0.0)]
rows = []
for name, cmd, lmode, run_s in poses:
    q = P0 + d * 0.8
    G.settle(q[0], q[1], yaw, fz, 2.5)
    c = dict(nom, lin_vel_x=0.0); c.update(cmd)
    G.rob.command(CA.to_names(BASE, c))
    for _ in range(int((3.0 + run_s) * G.HZ)):
        G.world.step(render=False)
    G.set_lighting(dict(lighting=dict(mode=lmode, dark=False), seed=7))
    bs = G.rob.base_state()
    rgb, dep, M = G.snapshot(bs)
    info = dict(G.snap_info or {})
    mask_img = np.asarray(mc.last.get("mask")) if mc.last is not None and mc.last.get("mask") is not None else np.zeros((224, 224), bool)
    T = mc.link_T()
    out2 = mc.render(G.cam_xf, G.rep, G.ann_rgb, G.ann_dep, bs, hide=False, mask=False)
    rgb_nm = out2[0]
    fr = G.floor_render_check(M, dep)
    R, p = mc.link_pose(T=T)
    eye_w = M[3, :3]
    eye_in_link = R.T @ (eye_w - p)
    el = CMT.axis_elev_deg(M)
    # chase view + eye dot
    from pxr import Gf
    yaw_r = float(bs["yaw"]); cy, sy = math.cos(yaw_r), math.sin(yaw_r)
    side = np.array([-sy, cy])
    pos = np.asarray(bs["pos"], float)
    ce = np.r_[pos[:2] - 1.2 * np.array([cy, sy]) + 0.9 * side, pos[2] + 0.5]
    ct = np.r_[pos[:2] + 0.2 * np.array([cy, sy]), pos[2] + 0.05]
    Mc = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, ce)), Gf.Vec3d(*map(float, ct)), Gf.Vec3d(0, 0, 1)).GetInverse()
    G.cam_xf.Set(Mc)
    G.rep.orchestrator.step(rt_subframes=8, delta_time=0.0, pause_timeline=False)
    chase = np.asarray(G.ann_rgb.get_data())[..., :3].astype(np.uint8).copy()
    Mc = np.array(Mc, float)
    def proj(X):
        v = (np.r_[X, 1.0] @ np.linalg.inv(Mc))[:3]
        if v[2] >= 0:
            return None
        return int(112 + G5.GS.FX * v[0] / -v[2]), int(112 - G5.GS.FX * v[1] / -v[2])
    pe, pa = proj(eye_w), proj(eye_w - 0.35 * M[2, :3])
    if pe is not None:
        if pa is not None:
            cv2.line(chase, pe, pa, (255, 0, 0), 1)
        cv2.circle(chase, pe, 3, (255, 0, 0), -1)
    r = dict(pose=name, light=lmode, cmd=c, eye_link=np.round(eye_in_link, 4).tolist(), eye_above_floor=round(float(eye_w[2] - fz), 3),
             body_z=round(float(pos[2] - fz), 3), axis_elev_deg=round(el, 1), mask_frac=info.get("mask_frac"), floor_render=fr,
             tilt=round(float(bs["tilt"]), 3), vx=round(float(bs["v_b"][0]), 3))
    rows.append(r)
    print("[chk]", json.dumps(r, default=float), flush=True)
    np.savez_compressed(os.path.join(a.out, f"cap_{name}.npz"), rgb=rgb, dep=dep, M=M, rgb_nomask=rgb_nm, chase=chase)
    g = lambda x: np.clip(255 * (x / 255.0) ** 0.6, 0, 255).astype(np.uint8)
    dv = np.clip(np.asarray(dep, float), 0, 8) / 8 * 255
    dv = cv2.applyColorMap(dv.astype(np.uint8), cv2.COLORMAP_VIRIDIS)[..., ::-1]
    mk = np.zeros((224, 224, 3), np.uint8); mk[np.asarray(dep) >= 59.0] = (60, 60, 60); mk[mask_img] = (255, 0, 255)
    tile = np.concatenate([g(rgb), dv, mk, g(rgb_nm), cv2.resize(chase, (224, 224))], 1)
    txt = np.full((40, tile.shape[1], 3), 255, np.uint8)
    cv2.putText(txt, f"{name}: eye_link {r['eye_link']} gap_to_face {eye_l[0] - face_x:+.3f} m  eye_h {r['eye_above_floor']} m  axis {el:.0f} deg  "
                f"mask {100 * float(info.get('mask_frac') or 0):.1f}%  floor {fr['n_ok']}/{fr['n_hit']} ok={fr['ok']}",
                (4, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    r["_tile"] = np.concatenate([txt, tile], 0)
sheet = np.concatenate([r.pop("_tile") for r in rows], 0)
hdr = np.full((30, sheet.shape[1], 3), 255, np.uint8)
cv2.putText(hdr, f"{a.base} {mc.mount}: columns = layer RGB (gamma 0.6) | masked depth 0-8 m | robot mask (magenta; grey = no return) | RGB mask off | chase (red = eye + axis)",
            (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
cv2.imwrite(os.path.join(a.out, "check_sheet.png"), np.concatenate([hdr, sheet], 0)[..., ::-1])
json.dump(dict(base=a.base, mount=mc.mount, link=mc.link, eye_link_const=eye_l.tolist(), face_x_eye_band=face_x, gap_m=float(eye_l[0] - face_x),
               body_max_x=all_x, n_band=int(len(band)), rows=rows, src=CMT.SRC[mc.mount]), open(os.path.join(a.out, "check.json"), "w"), indent=1, default=float)
print("[chk] DONE", flush=True)
app.close()
