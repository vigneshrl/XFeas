#!/usr/bin/env python
"""
gen5 closed-loop diagnostic video (gsam2 env): chase (recording camera) | head camera, with the decision in force overlaid
(read from data/gen5_closed/diag/<jid>/decisions5_t<k>.jsonl, matched by recorder frame index). Recording only.
    python closed_loop/g5_video.py <jid> <kind> [--trial 0] [--scene SC2A_O1]
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import argparse, json, os, subprocess, glob
import numpy as np
import cv2

G5 = xfeas_paths.ROOT + "/data/gen5_closed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jid"); ap.add_argument("kind"); ap.add_argument("--trial", type=int, default=0); ap.add_argument("--scene", default="SC2A_O1")
    a = ap.parse_args()
    fd = os.path.join(G5, "videos", "frames", f"{a.kind}_{a.scene}_{a.jid}_t{a.trial}")
    hd = os.path.join(G5, "diag", a.jid, f"head_t{a.trial}")
    dec = [json.loads(l) for l in open(os.path.join(G5, "diag", a.jid, f"decisions5_t{a.trial}.jsonl"))]
    out = os.path.join(G5, "videos", f"{a.kind}_{a.scene}_{a.jid}_t{a.trial}_chase_head.mp4")
    tmp = out + ".frames"; os.makedirs(tmp, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(fd, "*.jpg")))
    last_head = None; j = 0
    for i, f in enumerate(fs):
        n = int(os.path.basename(f)[:-4])
        c = cv2.imread(f)
        hp = os.path.join(hd, f"{n:06d}.jpg")
        if os.path.exists(hp):
            last_head = cv2.imread(hp)
        h = last_head if last_head is not None else np.zeros((224, 224, 3), np.uint8)
        h = cv2.resize(h, (720, 720))
        while j + 1 < len(dec) and dec[j + 1].get("frame") is not None and dec[j + 1]["frame"] <= n:
            j += 1
        img = np.hstack([c, h])
        d = dec[j] if dec and dec[j].get("frame") is not None and dec[j]["frame"] <= n else None
        cv2.rectangle(img, (0, 0), (img.shape[1], 70), (20, 20, 20), -1)
        if d:
            e = d["executed"]
            t1 = (f"{d['arm']}  t {d['t']:.2f}s  x {d['x']:.2f}  to lintel {d['d_to_overhang']:+.2f} m  body top {d['top']} m (lintel {d['overhang_under']:.2f})")
            t2 = (f"SENT vx {e.get('lin_vel_x')} h {e.get('pelvis_height')} pitch {e.get('torso_pitch')}  | model pick {d['chosen']['cls']} "
                  f"U {d['chosen']['U']:.2f} (stock U {d['U_stock']:.2f})")
            cv2.putText(img, t1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)
            cv2.putText(img, t2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (120, 230, 255), 2)
        cv2.imwrite(os.path.join(tmp, f"{i:06d}.jpg"), img)
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate", "30", "-i", os.path.join(tmp, "%06d.jpg"), "-vf", "scale=1600:-2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", out], check=True)
    for f in glob.glob(os.path.join(tmp, "*.jpg")):
        os.remove(f)
    os.rmdir(tmp)
    print(out)


if __name__ == "__main__":
    main()
