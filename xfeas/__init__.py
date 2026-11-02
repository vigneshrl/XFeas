"""XFeas: a learned feasibility layer on top of off-the-shelf locomotion controllers.

Importing this package puts every XFeas code folder on sys.path, so modules import each other by plain name
(`import feasibility_model`, `import episode_runner`). Every script in the repo starts with `import xfeas` for that.
Machine-specific locations (Isaac Sim, Python environments, third-party code, assets) live in `xfeas.paths`.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIRS = (
    # the method
    "xfeas/model",
    "xfeas/data_collection",
    "xfeas/train",
    "xfeas/closed_loop",
    # simulation: robots, controllers, scenes
    "xfeas/sim",
    "xfeas/sim/controllers",
    "xfeas/sim/controllers/sonic",
    "xfeas/sim/controllers/php",
    "xfeas/sim/controllers/gr00t",
    "xfeas/sim/controllers/urma",
    "xfeas/sim/controllers/isaaclab",
    "xfeas/scenes",
    "xfeas/scenes/terrain",
    # earlier generations the method builds on
    "xfeas/_support/model",
    "xfeas/_support/data_collection",
    "xfeas/_support/train",
    "xfeas/_support/closed_loop",
    # comparison methods and utilities
    "baselines",
    "tools",
)
for _d in reversed(CODE_DIRS):
    _p = os.path.join(_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from xfeas import paths  # noqa: E402
