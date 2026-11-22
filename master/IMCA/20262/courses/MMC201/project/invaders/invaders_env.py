"""
Compatibility entry point. The environment now lives in the package
`mini_invaders/` (one file per component; start with mini_invaders/__init__.py
for the reading order). Older scripts keep working:

    from invaders_env import Invaders, heuristic, render, ACTIONS
"""
from mini_invaders import *                  # noqa: F401,F403
from mini_invaders import Invaders, heuristic, render  # noqa: F401
