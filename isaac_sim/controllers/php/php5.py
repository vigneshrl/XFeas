"""PHP (isaac_sim/controllers/php/g1_php.py, released depth student, NOT modified) for the gen3/gen5 EnvSim / GenSim harness: the
adapter was written for the episode runners and lacks two RobotAdapter attributes the harness descriptor reads
(_effort_cap, _base_masses); they are defaulted here (None = nominal limits / unknown mass). Nothing else changes; PHP's
own D435i depth rig (release mount, 106 x 60, 10 Hz) keeps driving its policy."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import xfeas_paths  # noqa: E402  (repo root: puts every XFeas code folder on sys.path, machine paths)
import g1_php as GP


class G1PhpAdapter5(GP.G1PhpAdapter):
    def __init__(self, kind, prim_path, position, yaw=0.0):
        self._effort_cap = None
        self._base_masses = None
        GP.G1PhpAdapter.__init__(self, kind, prim_path, position, yaw)

    def descriptor(self, stage=None):
        try:
            return GP.G1PhpAdapter.descriptor(self, stage)
        except Exception as e:                       # noqa: BLE001
            return dict(kind=self.kind, controller="PHP depth student (released, D435i)", descriptor_error=repr(e)[:200],
                        joint_names=list(self._art.dof_names) if getattr(self._art, "dof_names", None) is not None else [])


def make_g1p5(kind, prim_path, position, yaw=0.0):
    GP.register_local_nav(kind)
    return G1PhpAdapter5(kind, prim_path, position, yaw)


make_g1p5.SPEC = GP.make_g1p.SPEC
