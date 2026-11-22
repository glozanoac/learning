"""
Bonuses: released by the GAME (not the aliens) from the top of the screen at a
random column; they fall 0.5 rows per step and bunkers do not stop them. The
ship collects one by being in its cell (+R_BONUS; the real value is the effect):

  life      +1 life (up to MAX_LIVES)
  gun       for GUN_STEPS steps: DUAL SHOT (two parallel bullets), faster,
            up to GUN_PAIRS pairs in the air
  repair    rebuilds the bunker blocks near the ship (useless on 'open')
  ammo      +AMMO_CRATE light shots
  grenades  +GRN_CRATE grenades

Supply drops: crates of a type fall more often when the ship is low on it.
"""
import numpy as np

from .config import (AMMO_CRATE, BONUS_DROP_P, BONUS_P, BONUSES, BUNKER_ROWS,
                     GRN_CRATE, GUN_STEPS, MAPS, MAX_AMMO, MAX_GRN, MAX_LIVES,
                     R_BONUS, SHIP_ROW, W)


class BonusMixin:
    def _bonus_turn(self, hit):
        """Step 6 of the transition: maybe release a bonus, let it fall, collect it.
        A ship hit this step cannot collect. Returns the reward."""
        r = 0.0
        if self.pu is None and self.rng.random() < BONUS_DROP_P:
            p = np.array(BONUS_P)
            if self.ammo_l < 10:
                p[BONUSES.index("ammo")] *= 2.5
            if self.ammo_g < 2:
                p[BONUSES.index("grenades")] *= 2.0
            kind = BONUSES[self.rng.choice(len(BONUSES), p=p / p.sum())]
            self.pu = [int(self.rng.integers(W)), 0.0, kind]    # [x, y (float), kind]
            self.stats["dropped"] += 1
            self.stats[f"drop_{kind}"] += 1
        for move in (0.0, 0.5):                       # check before and after it falls
            if self.pu is None:
                break
            self.pu[1] += move
            if self.pu[0] == self.sx and int(self.pu[1]) == self.sy and not hit and self.lives > 0:
                self._apply_bonus(self.pu[2])
                r += R_BONUS
                self.pu = None
            elif self.pu[1] > SHIP_ROW:
                self.pu = None
        return r

    def _apply_bonus(self, kind):
        self.stats["collected"] += 1
        self.stats[f"got_{kind}"] += 1
        if kind == "life":
            self.lives = min(MAX_LIVES, self.lives + 1)
        elif kind == "gun":
            self.gun = GUN_STEPS
        elif kind == "ammo":
            self.ammo_l = min(MAX_AMMO, self.ammo_l + AMMO_CRATE)
        elif kind == "grenades":
            self.ammo_g = min(MAX_GRN, self.ammo_g + GRN_CRATE)
        else:                                         # repair nearby blocks
            m = MAPS[self.map]
            for x in m["cols"]:
                if abs(x - self.sx) <= 3:
                    for y in BUNKER_ROWS:
                        if not (x == self.sx and y == self.sy):
                            self.bunkers[(x, y)] = m["hp"]
