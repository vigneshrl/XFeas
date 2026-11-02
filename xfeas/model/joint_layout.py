#!/usr/bin/env python
"""One shared joint layout for all robots (G1 29 joints, Spot 19, Jackal wheels 4, Iris rotors 4), so a single
network reads every robot's body state. Import it before building a model or a dataset.

Details:
gen5 canonical joint list = gen4 (G1 29 + Spot 19) + Jackal wheels (4) + Iris rotors (4). Patches gen4_common IN PLACE
(CANON_JOINTS, NJ, _JIDX, F_FRAME) so dataset_gen4.build_context, the gen4 encoders and feasibility_model all see the same sizes.
Import this BEFORE building any model / context (dataset5 and feasibility_model do). G1 slots 0-28 and Spot 29-47 are unchanged, so
gen4 weights load; only the h(t) / probe-trace patch MLPs grow (q / dq: +2 x 8 per frame, see feasibility_model.load_gen4).
"""

import xfeas  # noqa: F401  (puts every XFeas code folder on sys.path)
import os, sys

THIS = os.path.dirname(os.path.abspath(__file__))
import gen4_common as G4C

XBODY_JOINTS = (
    "front_left_wheel_joint",
    "front_right_wheel_joint",
    "rear_left_wheel_joint",
    "rear_right_wheel_joint",
    "rotor0_joint",
    "rotor1_joint",
    "rotor2_joint",
    "rotor3_joint",
)
NJ_GEN4 = len(G4C.G1_JOINTS) + len(G4C.SPOT_JOINTS)
F_FRAME_GEN4 = 7 + 7 + 5 + 1 + 6 + 12 + 1 + 2 * NJ_GEN4
QDQ_OFF = F_FRAME_GEN4 - 2 * NJ_GEN4  # where the q block starts inside one h(t) frame

if XBODY_JOINTS[0] not in G4C._JIDX:
    G4C.CANON_JOINTS = G4C.CANON_JOINTS + XBODY_JOINTS
    G4C.NJ = len(G4C.CANON_JOINTS)
    G4C._JIDX.update({n: i for i, n in enumerate(G4C.CANON_JOINTS)})
    G4C.F_FRAME += 2 * len(XBODY_JOINTS)  # q / dq slots per joint in every h(t) / probe frame
NJ, F_FRAME = G4C.NJ, G4C.F_FRAME


def frame_index_map():
    """old (gen4) frame feature index -> new (gen5) index"""
    m = []
    for f in range(F_FRAME_GEN4):
        if f < QDQ_OFF + NJ_GEN4:
            m.append(f)  # head features + q block (new q slots appended after it)
        else:
            m.append(f + len(XBODY_JOINTS))  # dq block shifts by the 8 new q slots
    return m
