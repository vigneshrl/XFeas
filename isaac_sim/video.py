#!/usr/bin/env python
"""MP4 writer for the demos (runs in the gsam2 env, which bundles ffmpeg via imageio_ffmpeg;
plain `imageio` is not installed, so this drives imageio_ffmpeg.write_frames directly).
H.264 + yuv420p so the files play in browsers, slide decks and phones; frames are padded to
even dimensions because yuv420p requires it."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import numpy as np


def write_mp4(frames, path, fps=30, crf=16):
    import imageio_ffmpeg
    h, w = np.asarray(frames[0]).shape[:2]
    H, W = h + (h % 2), w + (w % 2)
    gen = imageio_ffmpeg.write_frames(path, (W, H), fps=fps, codec="libx264", pix_fmt_in="rgb24",
                                      pix_fmt_out="yuv420p", macro_block_size=1,
                                      output_params=["-crf", str(crf), "-preset", "slow", "-movflags", "+faststart"])
    gen.send(None)
    prev, run, worst = None, 0, 0
    for f in frames:
        f = np.asarray(f, np.uint8)[..., :3]
        if f.shape[:2] != (H, W):
            p = np.zeros((H, W, 3), np.uint8); p[:f.shape[0], :f.shape[1]] = f; f = p
        small = f[::8, ::8].astype(np.int16)
        run = run + 1 if prev is not None and np.abs(small - prev).mean() < 0.05 else 0
        worst, prev = max(worst, run), small
        gen.send(np.ascontiguousarray(f).tobytes())
    gen.close()
    # a stale render path shows up as long runs of IDENTICAL frames (not just a still robot,
    # which still has lighting noise / camera easing). Flag anything over 1 s.
    if worst > fps:
        print(f"[video] WARNING {path}: {worst} identical frames in a row ({worst / fps:.1f} s) "
              f"-- check the capture path", flush=True)
    return path
