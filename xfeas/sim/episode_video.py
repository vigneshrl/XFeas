#!/usr/bin/env python
"""Overlay + mp4 for one episode_runner.py episode: episode_video.py <frame_dir> <out.mp4>"""

import os, sys  # noqa: E401  (standalone process, possibly in an isolated env: find the repo root itself)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import xfeas  # noqa: E402, F401  (puts every XFeas code folder on sys.path)
import json, os, sys
import cv2

THIS = os.path.dirname(os.path.abspath(__file__))
import chase as CH
from video import write_mp4

NICE = dict(success="REACHED GOAL", fall="FELL", stuck="STUCK", gave_up="GAVE UP", timeout="TIMED OUT")
fdir, out = sys.argv[1], sys.argv[2]
m = json.load(open(os.path.join(fdir, "meta.json")))
frames, F = CH.read_frames(fdir), m["frames"]
res = []
for i, im in enumerate(frames):
    tt, inf, vx = F[min(i, len(F) - 1)] if F else (0, "", 0)
    CH.panel(im, 0, 0, im.shape[1], 70, 0.6)
    CH.text(im, m["label"], 14, 30, 0.85, (255, 225, 120), 2)
    CH.text(
        im,
        f"{m['robot'].upper()}  {m['scene']}: {m['desc']}  |  no map, relative goal only  |  t={tt:5.1f}s",
        14,
        58,
        0.55,
        (230, 230, 230),
        1,
    )
    if inf:
        CH.panel(im, 0, im.shape[0] - 52, min(im.shape[1], 14 + 13 * len(inf)), 52, 0.6)
        CH.text(im, inf, 14, im.shape[0] - 18, 0.7, (140, 255, 140), 2)
    res.append(im)
lab = NICE.get(m["outcome"], m["outcome"].upper())
for im in res[-45:]:
    (tw, _), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
    W, H = im.shape[1], im.shape[0]
    CH.panel(im, W // 2 - tw // 2 - 18, H // 2 - 40, tw + 36, 64, 0.6)
    CH.text(
        im, lab, W // 2 - tw // 2, H // 2 + 8, 1.3, (100, 240, 100) if m["outcome"] == "success" else (245, 90, 80), 3
    )
print("[video]", write_mp4(res, out, fps=30))
