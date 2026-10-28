"""
URMA ("One Policy to Run Them All", Bohlinger et al., CoRL 2024) policy forward pass in plain numpy.

A line-by-line port of one_policy_to_run_them_all/algorithms/uni_ppo/ppo/policy.py (flax) so that the RELEASED
weights (experiments/pre_trained_model, exported once to urma_params.npz by export_urma.py) can be evaluated inside
Isaac Sim's python without JAX. Deterministic action = the policy mean (what the release's test() uses).
Validated against the flax module on recorded MuJoCo observations (export_urma.py: max |diff| reported there).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import numpy as np

EPS_LN = 1e-6            # flax.linen.LayerNorm default epsilon
STAB = 1e-8              # config_algorithm.stability_epsilon (checkpoint)
T_MIN = 0.015            # softmax_temperature_min (checkpoint)
MEAN_CLIP = 10.0         # policy_mean_abs_clip (checkpoint)


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


class URMAPolicy:
    def __init__(self, npz_path):
        z = np.load(npz_path)
        self.p = {k: z[k].astype(np.float64) for k in z.files}

    def _d(self, name, x):
        return x @ self.p[name + "/kernel"] + self.p[name + "/bias"]

    def _ln(self, name, x):
        m = x.mean(-1, keepdims=True)
        v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) / np.sqrt(v + EPS_LN) * self.p[name + "/scale"] + self.p[name + "/bias"]

    def _attend(self, desc, state, d0, ln, d1, ds, temp):
        m = self._d(d1, _elu(self._ln(ln, self._d(d0, desc))))
        m = np.clip(np.tanh(m), -1.0 + STAB, 1.0 - STAB)
        lat = _elu(self._d(ds, state))                                   # (N, 4)
        e = np.exp(m / (np.exp(self.p[temp][0]) + T_MIN))
        w = e / (e.sum(-1, keepdims=True) + STAB)                       # softmax over the 64 / 32 slots, per token
        out = (w[:, :, None] * lat[:, None, :]).reshape(len(desc), -1)   # (N, slots*4)
        return out.sum(0), lat

    def __call__(self, jdesc, jstate, fdesc, fstate, general):
        """jdesc (J,23) jstate (J,3) fdesc (F,10) fstate (F,2) general (16,) -> action (J,)"""
        jl, latj = self._attend(jdesc, jstate, "Dense_0", "LayerNorm_0", "Dense_1", "Dense_2", "joint_log_softmax_temperature")
        fl, _ = self._attend(fdesc, fstate, "Dense_3", "LayerNorm_1", "Dense_4", "Dense_5", "foot_log_softmax_temperature")
        x = np.concatenate([jl, fl, general])
        a = _elu(self._ln("LayerNorm_2", self._d("Dense_6", x)))
        a = _elu(self._d("Dense_7", a))
        a = self._d("Dense_8", a)                                        # (128,)
        dl = self._d("Dense_10", _elu(self._ln("LayerNorm_3", self._d("Dense_9", jdesc))))   # (J,128)
        c = np.concatenate([np.broadcast_to(a, (len(jdesc), a.shape[0])), latj, dl], -1)
        mu = self._d("Dense_12", _elu(self._ln("LayerNorm_4", self._d("Dense_11", c))))[:, 0]
        return np.clip(mu, -MEAN_CLIP, MEAN_CLIP)
