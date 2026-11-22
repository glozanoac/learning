"""
Alien fire, and the danger it creates for the ship. This is where the hidden
phase xi acts: it is the NON-STATIONARY part of the transition kernel.

Every shot starts right below the alien that fires it. Alien bullets fall 1 row
per step and may drift 1 column per step (diagonal); bunkers stop them.

Phase (never part of the observation):
  rain    aliens fire at random, some shots diagonal.
  sniper  aliens aim at the ship's current cell (straight or diagonal).
  trap    while a bonus is falling, the aliens fire volleys into its column.
  smart   the aliens hunt the ship together:
            * they LEAD their shots (aim where the ship is heading);
            * PINCER fire: shots at both sides of the ship at once;
            * they CUT ESCAPE ROUTES: they aim at the cells the ship could
              dodge to (cells not already under fire);
            * they BREAK COVER: while the ship hides they shoot the bunker
              block above it, and save shots for a burst when it comes out;
            * divers go kamikaze more often when the ship is boxed in
              (aliens.py reads escape_cells());
            * ESCALATION: every new wave is more aggressive (up to MAX_AGGRO).

Because the phase changes p(s'|s,a) but not the observation, the agent faces a
POMDP [report, remark on partial observability]; the smart aliens also read the
ship's last move vx, which is why vx is part of the state s (Markov property
[Zhao 1.7, (1.4)]).
"""
import numpy as np

from .config import (BUNKER_ROWS, ESCALATION, KIND_FIRE, MAX_AGGRO, MOVES,
                     SHIP_ROW, TITAN_VOLLEY_P, W, ZONE_TOP)


class AlienFireMixin:
    # ------------------------------------------------------------- firing --
    def _shot_at(self, tx, ty):
        """Aimed shot: it always starts right below a living alien. Among all
        (alien, direction) pairs, pick the one whose path passes closest to
        the target cell (tx, ty); fire only if it misses by at most 1 column.
        Gunners are preferred when several aliens can make the shot."""
        best, best_miss = [], 2
        for al in self.aliens:
            xa, y0 = al["x"], al["y"] + 1
            k = ty - y0
            if k < 0:
                continue
            for dx in (-1, 0, 1):
                xf = xa + dx * k
                if not 0 <= xf < W:
                    continue                         # would leave the screen first
                miss = abs(xf - tx) - (0.5 if al["kind"] == "gunner" else 0)
                if miss < best_miss:
                    best, best_miss = [(xa, y0, dx)], miss
                elif miss == best_miss:
                    best.append((xa, y0, dx))
        if best and best_miss <= 1:
            xa, y0, dx = best[self.rng.integers(len(best))]
            self.bullets.append([xa, y0, dx])

    def _fire(self):
        """Step 4 of the transition: every alien may fire, depending on the phase."""
        if not self.aliens:
            return
        rng = self.rng
        p_rand = {"rain": 0.06, "sniper": 0.015, "trap": 0.025, "smart": 0.01}[self.phase]
        for al in self.aliens:                                # random shots
            if rng.random() < p_rand * KIND_FIRE[al["kind"]]:
                drift = [-1, 1] if al["kind"] == "gunner" else [-1, 0, 0, 1]
                self.bullets.append([al["x"], al["y"] + 1, int(rng.choice(drift))])
        for al in self.aliens:                                # titans: one straight shot
            if al["kind"] == "titan" and rng.random() < TITAN_VOLLEY_P:
                self.bullets.append([al["x"], al["y"] + 1, 0])
        if self.phase == "sniper" and rng.random() < 0.35:    # aimed at the ship
            self._shot_at(self.sx, self.sy)
        if self.phase == "smart":
            self._smart_fire()
        if self.phase == "trap" and self.pu is not None:      # volley on the bait
            for _ in range(len(self.aliens)):
                if rng.random() < 0.35:
                    ty = int(rng.integers(max(ZONE_TOP, int(self.pu[1]) + 1), SHIP_ROW + 1)) \
                        if self.pu[1] < SHIP_ROW else SHIP_ROW
                    self._shot_at(self.pu[0], ty)

    def aggression(self):
        """Escalation of the smart phase: grows with the wave number, capped."""
        return min(MAX_AGGRO, 1 + ESCALATION * (self.wave - 1))

    def _smart_fire(self):
        rng = self.rng
        ag = self.aggression()
        near_y = max(a["y"] for a in self.aliens)
        k = self.sy - near_y                                  # steps until a shot arrives
        lead = int(np.clip(self.sx + self.vx * k, 0, W - 1))  # where the ship is heading
        close = float(np.clip(1 - (self.sy - near_y) / (SHIP_ROW - 1), 0, 1))
        if self.covered():
            self.ammo = min(3.0, self.ammo + 0.15 * ag)       # save shots for later
            if rng.random() < 0.2 * ag:                       # BREAK COVER: shoot the block
                blocks = [y for y in BUNKER_ROWS if (self.sx, y) in self.bunkers and y < self.sy]
                if blocks:
                    self._shot_at(self.sx, min(blocks))
            return
        p = min(0.95, 0.3 * ag * (1 + 2 * close))             # exposed: fire, more when close
        if rng.random() < p:
            self._shot_at(lead, self.sy)
        if rng.random() < min(0.9, 0.35 * ag):                # PINCER: both sides at once
            for side in (-1, 1):
                self._shot_at(int(np.clip(lead + side, 0, W - 1)), self.sy)
        if rng.random() < min(0.9, 0.3 * ag):                 # CUT ESCAPE ROUTES
            for ex, ey in self.escape_cells()[:2]:
                self._shot_at(ex, ey)
        while self.ammo >= 1:                                 # burst of saved shots
            self.ammo -= 1
            self._shot_at(int(np.clip(lead + rng.integers(-1, 2), 0, W - 1)), self.sy)

    # ------------------------------------------------------------- flight --
    def _move_alien_bullets(self):
        """Step 5 of the transition: bullets fall (maybe diagonally).
        Returns True if one reaches the ship."""
        hit = False
        alive = []
        for b in self.bullets:
            b[0] += b[2]
            b[1] += 1
            if not 0 <= b[0] < W or b[1] > SHIP_ROW:
                continue
            if self._hit_bunker(b[0], b[1]):
                continue
            if b[0] == self.sx and b[1] == self.sy:
                hit = True
                continue
            alive.append(b)
        self.bullets = alive
        return hit

    def _bullet_on_ship(self):
        """The ship walked into a bullet."""
        for b in self.bullets:
            if b[0] == self.sx and b[1] == self.sy:
                self.bullets.remove(b)
                return True
        return False

    # ------------------------------------------------------------- danger --
    def threat_at(self, x, y, steps):
        """Will an alien bullet or a kamikaze diver be in (x, y) now or within
        `steps` moves? (a kamikaze can shift 1 column per row as it homes in)"""
        for bx, by, bdx in self.bullets:
            for k in range(steps + 1):
                if bx + bdx * k == x and by + k == y:
                    return True
        for a in self.aliens:
            if a.get("kamikaze"):
                for k in range(steps + 1):
                    if a["y"] + k == y and abs(a["x"] - x) <= k:
                        return True
        return False

    bullet_at = threat_at                             # backwards-compatible name

    def escape_cells(self):
        """Cells the ship could move to next step that are not under fire."""
        out = []
        for dx, dy in MOVES[:5]:
            x, y = self.sx + dx, self.sy + dy
            if self.free(x, y) and not self.threat_at(x, y, 2):
                out.append((x, y))
        return out
