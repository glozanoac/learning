"""
The ship's weapons: limited ammunition of two types, a shared launcher, the
flight of the ship's projectiles, grenade blasts, and the damage-based reward.

  light shot  damage 1, start with 25 (max 60); costs R_SHOT = -0.02.
  grenade     PROXIMITY FUSE: explodes when it touches an alien OR passes
              next to one (within 1 cell), in a round blast of radius 2:
              3 damage at the centre, 2 next to it, 1 at the edge.
              Start with 4 (max 10); costs R_GRENADE = -0.1.
  SHARED LAUNCHER: only one projectile in the air at a time (a light shot or a
              grenade; with the gun bonus, up to GUN_PAIRS dual shots), so each
              grenade is a real choice instead of extra fire.
  Dry trigger: firing with no ammo of that type does nothing and costs R_DRY.
  Projectiles rise 2 rows per step (light shots 3 with the gun bonus). Hitting
  one's own bunker costs R_OWN_BLOCK.

Reward [Zhao 1.5]: +R_DMG per point of damage actually dealt (overkill earns
nothing) and a kill bonus so that destroying an alien of kind k is worth
KIND_REWARD[k] in total, whichever weapon did it (the report proves this by a
telescoping sum).
"""
import numpy as np

from .config import (GRN_DMG, GRN_RADIUS, GUN_PAIRS, KIND_HP, KIND_REWARD,
                     R_DMG, R_DRY, R_GRENADE, R_OWN_BLOCK, R_SHOT, W)


class WeaponsMixin:
    # ------------------------------------------------------------ launcher --
    def can_fire_light(self):
        """Shared launcher: no grenade in the air, and room for a light shot."""
        if any(pb[2] == GRN_DMG for pb in self.pbullets):
            return False
        n = len(self.pbullets)
        room = n <= 2 * (GUN_PAIRS - 1) if self.gun else n == 0
        return self.ammo_l > 0 and room

    def can_fire_grenade(self):
        """Shared launcher: a grenade only when nothing at all is in the air."""
        return self.ammo_g > 0 and not self.pbullets

    def _pull_trigger(self, a):
        """Actions 5 (light shot) and 6 (grenade). Returns the reward."""
        r = 0.0
        if a == 5:                                            # light shot
            if self.ammo_l <= 0:
                self.stats["dry"] += 1
                r += R_DRY
            elif self.can_fire_light():
                if self.gun:                                  # dual shot, still 1 round
                    lx = self.sx if self.sx < W - 1 else self.sx - 1
                    self.pbullets += [[lx, self.sy, 1], [lx + 1, self.sy, 1]]
                else:
                    self.pbullets.append([self.sx, self.sy, 1])
                self.ammo_l -= 1
                r += R_SHOT
                self.stats["shots_light"] += 1
        elif a == 6:                                          # grenade
            if self.ammo_g <= 0:
                self.stats["dry"] += 1
                r += R_DRY
            elif self.can_fire_grenade():
                self.pbullets.append([self.sx, self.sy, GRN_DMG])
                self.ammo_g -= 1
                r += R_GRENADE
                self.stats["shots_grenade"] += 1
        return r

    # -------------------------------------------------------------- flight --
    def _move_player_bullets(self):
        """Player projectiles rise; they hit bunkers, aliens, or explode.
        Returns the reward earned this step."""
        r = 0.0
        self.blasts = [[bx, by, k - 1] for bx, by, k in self.blasts if k > 1]
        alive = []
        for pb in self.pbullets:
            speed = 3 if (self.gun and pb[2] == 1) else 2
            for _ in range(speed):
                pb[1] -= 1
                x, y, dmg = pb
                if y < 0:
                    pb = None
                    break
                if self._hit_bunker(x, y, dmg):               # breaking your own cover
                    r += R_OWN_BLOCK
                    self.stats["own_blocks"] += 1
                    if dmg == GRN_DMG:                        # grenade still explodes
                        r += self._explode(x, y)
                    pb = None
                    break
                if dmg == GRN_DMG:                            # proximity fuse
                    if any(max(abs(a["x"] - x), abs(a["y"] - y)) <= 1 for a in self.aliens):
                        r += self._explode(x, y)
                        pb = None
                        break
                    continue
                got = self._damage_alien(x, y, dmg)
                if got is not None:
                    r += got
                    pb = None
                    break
            if pb is not None:
                alive.append(pb)
        self.pbullets = alive
        if self.gun:
            self.gun -= 1
        return r

    # -------------------------------------------------------------- damage --
    def _explode(self, x, y):
        """Grenade blast at (x, y): a round area of radius GRN_RADIUS.
        Damage 3 at the centre, 2 within 1.5 cells, 1 out to the edge."""
        self.blasts.append([x, y, 3])
        r, hit_any = 0.0, False
        for a in list(self.aliens):
            if a not in self.aliens:                          # removed by an earlier split
                continue
            dist = np.hypot(a["x"] - x, a["y"] - y)
            if dist <= GRN_RADIUS + 0.01:
                dmg = GRN_DMG if dist == 0 else (2 if dist <= 1.5 else 1)
                got = self._damage_alien(a["x"], a["y"], dmg, direct=False)
                r += got or 0.0
                hit_any = True
        self.stats["alien_hits"] += hit_any                   # counts as a hit for accuracy
        return r

    def _damage_alien(self, x, y, dmg=1, direct=True):
        """Damage to the alien in (x, y): returns the reward, or None if no alien."""
        for a in self.aliens:
            if a["x"] == x and a["y"] == y:
                self.stats["alien_hits"] += direct
                if a["shield"]:                               # the shield absorbs the hit
                    a["shield"] = False
                    self.stats["shield_blocks"] += 1
                    return 0.0
                dealt = min(dmg, a["hp"])                     # overkill is wasted
                self.stats["damage"] += dealt
                a["hp"] -= dealt
                r = R_DMG * dealt
                if a["hp"] <= 0:
                    self.aliens.remove(a)
                    self.stats["kills"] += 1
                    self.stats[f"kill_{a['kind']}"] += 1
                    r += KIND_REWARD[a["kind"]] - R_DMG * KIND_HP[a["kind"]]   # kill bonus
                    if a["kind"] == "splitter":               # splits into two minis
                        self._split(a)
                return r
        return None
