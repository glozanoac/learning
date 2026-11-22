"""
The environment class: state, reset, and one transition.

How the class maps to the MDP of Zhao, "Mathematical Foundations of RL", Ch. 1:
  state s          the attributes of Invaders (ship, aliens, bullets, bunkers,
                   bonus, ammunition, lives, wave, t, vx, ...)        [Zhao 1.2]
  action a         config.ACTIONS (7)                                 [Zhao 1.2]
  transition       step(): p(s'|s,a) is the composition of the six
                   sub-steps below, random only through self.rng      [Zhao 1.3]
  reward           r returned by step(), the sum of the sub-steps'
                   rewards (config.R_*)                               [Zhao 1.5]
  episode          ends only when lives == 0 (absorbing terminal
                   state); there is no time limit                     [Zhao 1.6]
  Markov property  s' depends only on (s, a)                          [Zhao 1.7, (1.4)]
  features phi(s)  features.py: what the agent sees                   [Zhao 8.1, 9.1]
The agent never reads these rules: everything it knows comes from samples
(s_t, a_t, r_{t+1}, s_{t+1}), i.e. it is model-free                   [Zhao Ch. 5, 7]

Each component of the game lives in its own module and is mixed into the class:
  arena.py       ship zone, bunkers            ArenaMixin
  weapons.py     ammunition, launcher, damage  WeaponsMixin
  aliens.py      waves, kinds, movement        AliensMixin
  alien_fire.py  alien fire, hidden phase      AlienFireMixin
  bonuses.py     falling bonuses               BonusMixin
"""
import numpy as np

from .alien_fire import AlienFireMixin
from .aliens import AliensMixin
from .arena import ArenaMixin
from .bonuses import BonusMixin
from .config import (BONUSES, KINDS, MOVES, R_HIT, SHIP_ROW, START_AMMO,
                     START_GRN, START_LIVES, W)
from .features import N_FEAT, features
from .weapons import WeaponsMixin


class Invaders(ArenaMixin, WeaponsMixin, AliensMixin, AlienFireMixin, BonusMixin):
    N_FEAT = N_FEAT                          # dimension of phi(s) (= 311)

    def __init__(self, seed=None, phase="rain", map_name="fortress"):
        self.rng = np.random.default_rng(seed)
        self.phase = phase                   # hidden mode xi, see alien_fire.py
        self.map = map_name
        self.reset()

    # --------------------------------------------------------------- reset --
    def reset(self):
        """Initial state s_0; returns phi(s_0)."""
        self.sx, self.sy = W // 2, SHIP_ROW
        self.lives = START_LIVES
        self.t = 0
        self.bullets = []            # alien bullets: [x, y, dx]
        self.pbullets = []           # player bullets: [x, y, damage]
        self.ammo_l, self.ammo_g = START_AMMO, START_GRN
        self.blasts = []             # grenade explosions to draw: [x, y, frames left]
        self.pu = None               # bonus: [x, y(float), kind]
        self.gun = 0                 # steps of gun power-up left
        self.vx = 0                  # ship's last horizontal move (smart aliens read it)
        self.ammo = 0.0              # shots saved by smart aliens while the ship hides
        self._build_bunkers()        # walls are built once per episode (not per wave)
        self.wave = 0
        self._spawn_wave()
        self.stats = dict(ret=0.0, kills=0, dropped=0, collected=0, hits=0,
                          own_blocks=0, dodges=0, shots_light=0, shots_grenade=0,
                          alien_hits=0, dry=0, empty_steps=0, rammed=0, crashed=0,
                          damage=0, shield_blocks=0, shields_cast=0,
                          **{f"got_{b}": 0 for b in BONUSES},
                          **{f"drop_{b}": 0 for b in BONUSES},
                          **{f"kill_{k}": 0 for k in KINDS})
        self.last_hit = False
        return self.features()

    def features(self):
        """phi(s) in R^311, see features.py  [Zhao 8.2.3, 8.1, 9.1]."""
        return features(self)

    # ---------------------------------------------------------------- step --
    def step(self, a):
        """One transition (s, a) -> (r, s')  [Zhao 1.3, 1.5]. Returns (phi(s'), r, done)."""
        r, hit = 0.0, False

        # 1. the ship moves and/or fires; walking into a bullet or a diver = hit
        r += self._move_ship(a)
        r += self._pull_trigger(a)                    # weapons.py
        hit |= self._bullet_on_ship()                 # alien_fire.py
        if self._diver_on_ship():                     # aliens.py
            hit = True
            self.stats["rammed"] += 1

        # 2. the ship's projectiles fly: bunkers, aliens, grenade blasts
        r += self._move_player_bullets()              # weapons.py

        # 3. the aliens move, use their abilities, maybe ram; new wave if none left
        if self._aliens_turn():                       # aliens.py
            hit = True
            self.stats["rammed"] += 1

        # 4. alien fire (depends on the hidden phase and on each alien's kind)
        self._fire()                                  # alien_fire.py

        # 5. alien bullets fall (maybe diagonally)
        hit |= self._move_alien_bullets()             # alien_fire.py

        self.last_hit = hit
        if hit:
            r += R_HIT
            self.lives = max(0, self.lives - 1)
            self.stats["hits"] += 1

        # 6. bonuses: released by the game, fall, collected in the ship's cell
        r += self._bonus_turn(hit)                    # bonuses.py

        self.stats["empty_steps"] += self.ammo_l == 0 and self.ammo_g == 0
        self.t += 1
        self.stats["ret"] += r
        done = self.lives <= 0                        # no time limit: only when out of lives
        return self.features(), r, done

    def _move_ship(self, a):
        """Move one cell if the target is free (actions 0..3; 4..6 stay)."""
        dx, dy = MOVES[a]
        if self.free(self.sx + dx, self.sy + dy):
            self.sx, self.sy = self.sx + dx, self.sy + dy
            self.vx = dx
        else:
            self.vx = 0
        return 0.0
