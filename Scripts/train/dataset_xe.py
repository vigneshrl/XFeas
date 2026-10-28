#!/usr/bin/env python
"""
XFeas sample contract + torch Dataset. Keep this file and robots.py in sync -- they are the
contract between the Isaac collector (writes) and the trainer (reads).

One EPISODE = one decision:  pre-decision proprio history -> RGB-D snapshot -> one candidate
action executed -> outcome. Stored as <run>/ep/NNNNNN.npz with keys:

  rgb          (H,W,3) uint8      snapshot at decision time
  depth        (H,W)   float32    metres (distance to image plane)
  prop_base    (T,13)  float32    robots.N_BASE, oldest first, 50 Hz
  prop_joints  (T,J,7) float32    robots.N_JOINT
  desc         str (json)         RobotAdapter.descriptor()
  action       (K,4)   float32    body-frame [vx,vy,vz,wz] per chunk step (m/s, rad/s)
  action_dt    float              seconds per chunk step
  skill        int                index into SKILLS
  success, fell, collision, stall : bool
  progress     float              fraction of the intended advance achieved (can be <0)
  max_tilt     float              rad
  failure_tag  str
  robot, hazard, scene, run_id    str
  value        float              hazard parameter (mu, step height, gap width, ...)
  region_id    str                same physical place across actions -> grouped splits + ranking
  variant      str (json)         set_variant() args used (ground truth, NOT a model input)

The descriptor is turned into model inputs here (featurize_descriptor), so the collector never
has to know the model's feature layout.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import glob, json, math, os
import numpy as np

SKILLS = ["go", "stop", "back_off", "climb_step", "pass_gap", "crouch_under", "fly_over", "spare"]
FAMILIES = ["legged", "wheeled", "aerial"]
N_JOINT_STATIC = 5
N_BODY = 12
T_HIST = 100
K_MAX = 16


def featurize_descriptor(desc):
    """descriptor dict -> jstatic (J,5), body (12,). Unknown limits (-1) map to 0 with the
    log features, which the model sees as 'no limit reported' (wheels, casters)."""
    lo = np.asarray(desc["joint_lower"], float); hi = np.asarray(desc["joint_upper"], float)
    span = hi - lo
    bounded = np.isfinite(span) & (span > 1e-6) & (span < 2 * math.pi + 1e-3)
    lo = np.where(bounded, lo, 0.0); hi = np.where(bounded, hi, 0.0)
    vmax = np.asarray(desc["joint_vmax"], float); tmax = np.asarray(desc["joint_tau_max"], float)
    wheel = np.array([("wheel" in n.lower()) for n in desc["joint_names"]], float)
    js = np.stack([lo, hi, np.log1p(np.maximum(vmax, 0)), np.log1p(np.maximum(tmax, 0)), wheel], 1)
    mc = desc["max_cmd"]
    bbox = np.nan_to_num(np.asarray(desc["bbox_xyz_m"], float), nan=0.0)
    fam = np.zeros(3); fam[FAMILIES.index(desc["family"])] = 1.0
    body = np.concatenate([[math.log(max(desc["mass_kg"], 1e-3))], bbox,
                           [mc["vx_fwd"], mc["vx_rev"], mc["vy"], mc["vz"], mc["wz"]], fam])
    assert body.shape == (N_BODY,)
    return js.astype(np.float32), body.astype(np.float32)


def load_episode(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    for k in ("desc", "variant", "failure_tag", "robot", "hazard", "scene", "run_id", "region_id"):
        if k in d:
            d[k] = str(d[k])
    return d


def index_runs(roots):
    """All episode files under the given run dirs, plus light metadata for splitting."""
    rows = []
    for r in roots:
        for p in sorted(glob.glob(os.path.join(r, "ep", "*.npz"))):
            z = np.load(p, allow_pickle=False)
            rid = str(z["region_id"])
            rows.append(dict(path=p, robot=str(z["robot"]), run_id=str(z["run_id"]),
                             scene=str(z["scene"]), hazard=str(z["hazard"]), region_id=rid,
                             arena=rid.split("-")[0], success=bool(z["success"])))
    return rows


def split(rows, mode, holdout):
    """mode: robot | run | scene | arena -> (train_rows, test_rows). Never splits a region
    across sets (same place, different action = the ranking unit). 'arena' holds out whole
    arena seeds (comma list, e.g. "a3,a7") for ALL robots -- the right check for a model
    meant to work on every body. 'robot' is the zero-shot new-body test."""
    key = {"robot": "robot", "run": "run_id", "scene": "scene", "arena": "arena"}[mode]
    hold = set(holdout.split(","))
    te = [r for r in rows if r[key] in hold]
    tr = [r for r in rows if r[key] not in hold]
    leak = {r["region_id"] for r in te} & {r["region_id"] for r in tr}
    if leak and mode != "robot":
        raise ValueError(f"{len(leak)} regions appear in both train and test for {mode}={holdout}")
    return tr, te


try:
    import torch
    from torch.utils.data import Dataset

    class EpisodeDataset(Dataset):
        def __init__(self, rows, img=224, hist=T_HIST, augment=False):
            self.rows, self.img, self.hist, self.augment = rows, img, hist, augment
            self._desc_cache = {}
            self._rng = np.random.default_rng()

        def _color_aug(self, rgb):
            """Kill COLOUR as a cue, keep LIGHTNESS structure. The interim model swung
            P(success) 0.02 -> 0.97 on the same Carter gap when only Replicator's block colours
            changed -- it had latched onto hue. Channel permutation / greyscale / per-channel
            gain remove hue information; brightness+contrast jitter stays mild so a dark glossy
            oil patch is still darker than the floor around it. Depth is never touched."""
            r = self._rng
            x = rgb.astype(np.float32) / 255.0
            if r.random() < 0.5:
                x = x[..., r.permutation(3)]
            if r.random() < 0.25:
                x = np.repeat(x.mean(-1, keepdims=True), 3, -1)
            x = x * r.uniform(0.85, 1.15, 3)                         # per-channel gain
            m = x.mean()
            x = (x - m) * r.uniform(0.75, 1.25) + m * r.uniform(0.7, 1.3)
            return (np.clip(x, 0, 1) * 255).astype(np.uint8)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            d = load_episode(self.rows[i]["path"])
            if d["desc"] not in self._desc_cache:
                self._desc_cache[d["desc"]] = featurize_descriptor(json.loads(d["desc"]))
            js, body = self._desc_cache[d["desc"]]
            T = self.hist
            pb, pj = d["prop_base"][-T:], d["prop_joints"][-T:]
            if len(pb) < T:                              # left-pad short histories by repeating
                pad = T - len(pb)
                pb = np.concatenate([np.repeat(pb[:1], pad, 0), pb])
                pj = np.concatenate([np.repeat(pj[:1], pad, 0), pj])
            act = np.zeros((K_MAX, 7), np.float32)            # [vx,vy,vz,wz,dh,pitch,roll]; 4-d -> default posture
            a = d["action"][:K_MAX]; act[:len(a), :a.shape[1]] = a
            rgb = self._color_aug(d["rgb"]) if self.augment else d["rgb"]
            return dict(
                rgb=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))),
                depth=torch.from_numpy(d["depth"][None].astype(np.float32)),
                base=torch.from_numpy(pb.astype(np.float32)),
                joints=torch.from_numpy(pj.astype(np.float32)),
                jstatic=torch.from_numpy(js), body=torch.from_numpy(body),
                action=torch.from_numpy(act), n_act=len(a),
                skill=int(d["skill"]),
                y_success=float(d["success"]), y_fall=float(d["fell"]),
                y_collision=float(d["collision"]), y_stall=float(d["stall"]),
                y_progress=float(d["progress"]), y_max_tilt=float(d["max_tilt"]),
                region=self.rows[i]["region_id"], robot=self.rows[i]["robot"],
                hazard=self.rows[i]["hazard"])

    def collate(batch):
        """Pad joints to the batch max J; mask marks real joints."""
        J = max(b["joints"].shape[1] for b in batch)
        out = {}
        for k in ("rgb", "depth", "base", "body", "action"):
            out[k] = torch.stack([b[k] for b in batch])
        B, T = len(batch), batch[0]["joints"].shape[0]
        joints = torch.zeros(B, T, J, batch[0]["joints"].shape[2])
        jstatic = torch.zeros(B, J, N_JOINT_STATIC)
        jmask = torch.zeros(B, J, dtype=torch.bool)
        for i, b in enumerate(batch):
            n = b["joints"].shape[1]
            joints[i, :, :n] = b["joints"]; jstatic[i, :n] = b["jstatic"]; jmask[i, :n] = True
        out.update(joints=joints, jstatic=jstatic, jmask=jmask,
                   skill=torch.tensor([b["skill"] for b in batch]))
        for k in batch[0]:
            if k.startswith("y_"):
                out[k] = torch.tensor([b[k] for b in batch], dtype=torch.float32)
        out["region"] = [b["region"] for b in batch]
        out["robot"] = [b["robot"] for b in batch]
        out["hazard"] = [b["hazard"] for b in batch]
        return out
except ImportError:          # Isaac's python writes episodes without needing torch here
    pass


def write_synthetic(root, n=240, seed=0):
    """Fake runs with a PLANTED rule so the trainer can be checked end to end:
    fail iff speed * (1 / mu) * (mass / 50) > 3, where mu is shown as image brightness.
    Three robots with different joint counts; region = (robot-agnostic place id)."""
    rng = np.random.default_rng(seed)
    robots = {"spot": (12, 31.6, "legged"), "h1": (19, 51.6, "legged"), "carter": (4, 76.2, "wheeled")}
    os.makedirs(os.path.join(root, "ep"), exist_ok=True)
    k = 0
    for reg in range(n // 6):
        mu = float(rng.uniform(0.1, 0.9))
        for rb, (J, m, fam) in robots.items():
            desc = dict(kind=rb, family=fam, joint_names=[f"j{i}" + ("_wheel" if fam == "wheeled" and i < 2 else "")
                                                          for i in range(J)],
                        joint_lower=[-1.0] * J, joint_upper=[1.0] * J, joint_vmax=[10.0] * J,
                        joint_tau_max=[50.0] * J, mass_kg=m, bbox_xyz_m=[1, .5, .6],
                        max_cmd=dict(vx_fwd=1.2, vx_rev=0.6, vy=0.3, vz=0.0, wz=1.0))
            for v in (0.3, 0.9):
                fail = v / mu * (m / 50.0) > 3.0
                np.savez_compressed(
                    os.path.join(root, "ep", f"{k:06d}.npz"),
                    rgb=np.full((64, 64, 3), int(255 * mu), np.uint8),
                    depth=np.full((64, 64), 2.0, np.float32),
                    prop_base=rng.normal(0, 0.1, (T_HIST, 13)).astype(np.float32),
                    prop_joints=rng.normal(0, 0.1, (T_HIST, J, 7)).astype(np.float32),
                    desc=json.dumps(desc), action=np.tile([v, 0, 0, 0], (10, 1)).astype(np.float32),
                    action_dt=0.2, skill=0, success=not fail, fell=fail, collision=False, stall=False,
                    progress=0.2 if fail else 1.0, max_tilt=1.0 if fail else 0.1, failure_tag="synthetic",
                    robot=rb, hazard="synthetic", scene="synthetic", run_id=f"r{reg % 4}", value=mu,
                    region_id=f"g{reg}", variant="{}")
                k += 1
    return k
