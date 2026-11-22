"""
Mini Space Invaders: a non-stationary grid game for actor-critic agents.

Free-moving aliens of several kinds, bunkers, diagonal fire, a ship that moves
in 2D, three maps, limited ammunition and bonuses that fall under fire. The
hidden phase (rain / sniper / trap / smart) changes how the aliens fire and is
never observed. An episode ends only when the ship has no lives.

Suggested reading order, one component per file:

  1. config.py      every constant: grid, maps, alien kinds, rewards, ammo
  2. env.py         the Invaders class: state, reset(), step() as six sub-steps,
                    and the map to Zhao's MDP (Ch. 1)
  3. arena.py       ship zone and bunkers                      (step 1)
  4. weapons.py     ammunition, shared launcher, grenade blast,
                    damage-based reward                        (steps 1-2)
  5. aliens.py      waves, kinds, movement, kamikazes,
                    shields, teleports                         (step 3)
  6. alien_fire.py  alien fire and the hidden phase: the
                    non-stationary part of p(s'|s,a)           (steps 4-5)
  7. bonuses.py     bonuses released by the game               (step 6)
  8. features.py    the observation phi(s) in R^311 the agent sees
  9. scripted.py    the Greedy / Cautious baselines
 10. render.py      drawing (no effect on the dynamics)

Usage:
    from mini_invaders import Invaders, heuristic, render, ACTIONS
    env = Invaders(seed=0, phase="smart", map_name="fortress")
    phi = env.reset()
    phi, r, done = env.step(5)                 # 5 = light shot
"""
from .config import *                        # noqa: F401,F403  (all constants)
from .env import Invaders
from .features import BLOCKS, N_FEAT, features
from .render import KIND_COLOR, render
from .scripted import heuristic
