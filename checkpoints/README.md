# Checkpoints

| File | What it is |
|---|---|
| `xfeas_all_g1s_v4.pt` | **Current XFeas feasibility model** (used by the `g5sm` arm). Trained on all controllers including GEAR-SONIC. |

**`xfeas_all_g1s_v4.pt`**
* Format `xfeas_gen5_v1_g5b`, saved 2026-11-02, 129 MB, sha256 `6acae0a1b12885b44d58d6fee9f0909082c64d203a14e883f1f731fab87177a2`.
* Contents: the full network weights (frozen DINOv3-S vision encoder + context encoder + 5 ensemble members), the
  training config (`cfg`), the input normalisation (`norm`) and the loss history. Every weight the model uses comes from
  this file. Note: the model server builds the network with `pretrained=True` before loading the file, so its first run
  downloads the DINOv3-S weights through `timm` from Hugging Face (they are then overwritten by the checkpoint).
* Trained with `train.py` on the v4 manifest (side-by-side branches + gen4 side-by-side options; school scenes excluded
  from training), 25 epochs, batch 16, ensemble 5, initialised from the gen4 context encoder. Selection score
  `U = mean P(success) − 0.5·std − 0.5·P(fall)`.
* Held out of training: the Isaac Sim Spot flat-terrain policy (`spotf`) only.

**Download.** Checkpoints are stored with [Git LFS](https://git-lfs.com). Run `git lfs install` once before cloning;
in an existing clone, `git lfs pull` fetches them.

**Use.**
```bash
python evaluate.py model --ckpt checkpoints/xfeas_all_g1s_v4.pt --device cpu      # offline check
python evaluate.py school demo sonic g5sm 1                                       # g5sm uses this file by default
```
Running an episode also needs the controller specs and self-test probes in `data/` (see the main README, "Data").
