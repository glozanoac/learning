"""
The aliens: waves, kinds, movement and special abilities.

Aliens move FREELY and individually in the alien zone (rows ALIEN_TOP..
ALIEN_BOTTOM); they never enter the player zone, except kamikaze divers.
On its turn each alien
  - DODGES if one of the ship's bullets is coming up its column (how often
    depends on its kind, KIND_EVADE);
  - BACKS OFF upwards if the ship gets right under it;
  - otherwise follows the plan of its kind (_alien_goal):
  gunner   1 hit,  +1.5  keeps a firing line on the ship from a distance
  splitter 2 hits, +2    when destroyed it SPLITS into two fast minis (+0.5 each)
  healer   2 hits, +2.5  hangs back and puts a SHIELD on a nearby ally every few
                         steps (the shield blocks the next hit)
  phantom  1 hit,  +2    TELEPORTS every few steps (it flickers just before)
  diver    1 hit,  +1.5  swoops down above the ship; close to it, it may go
                         KAMIKAZE: crosses into the player zone and homes in to
                         ram the ship (-10, a life lost); it wrecks a bunker block
                         it crashes into, and loops back to the top if it misses
  titan    6 hits, +5    armoured turtle: slow, hovers over the ship, fires single
                         straight bullets; at most one per wave, never dodges
A new wave of N_ALIENS appears at the top when all are destroyed.

Part of the state s [Zhao 1.2]: the list of alien dicts. All randomness of the
moves comes from self.rng, so p(s'|s,a) [Zhao 1.3] is well defined.
"""
import numpy as np

from .config import (ALIEN_BOTTOM, ALIEN_TOP, HEAL_EVERY, HEAL_RANGE, KAMIKAZE_P,
                     KIND_EVADE, KIND_HP, KIND_MOVE_EVERY, KIND_P, MAPS, N_ALIENS,
                     PHANTOM_EVERY, SHIP_ROW, SPAWN_KINDS, W)


class AliensMixin:
    # --------------------------------------------------------------- waves --
    def _spawn_wave(self):
        kinds = [str(k) for k in self.rng.choice(SPAWN_KINDS, size=N_ALIENS, p=KIND_P)]
        for unique in ("titan", "healer"):                    # at most one of each
            idx = [j for j, k in enumerate(kinds) if k == unique]
            for j in idx[1:]:
                kinds[j] = "splitter"
        cols = self.rng.choice(W, size=N_ALIENS, replace=False)
        self.next_id = N_ALIENS
        self.wave = getattr(self, "wave", 0) + 1
        self.aliens = [self._new_alien(int(c), ALIEN_TOP, k, i)
                       for i, (c, k) in enumerate(zip(cols, kinds))]

    def _new_alien(self, x, y, kind, ident):
        return dict(x=x, y=y, kind=kind, hp=KIND_HP[kind], id=ident, diving=False,
                    shield=False, timer=int(self.rng.integers(PHANTOM_EVERY)))

    @property
    def alien_xs(self):
        return [a["x"] for a in self.aliens]

    def _aliens_turn(self):
        """Step 3 of the transition: move, use abilities, respawn.
        Returns True if a kamikaze rammed the ship."""
        self.rammed_now = False
        self._move_aliens()
        self._abilities()
        if not self.aliens:
            self._spawn_wave()
        return self.rammed_now

    # ----------------------------------------------------------- abilities --
    def _split(self, a):
        """A destroyed splitter leaves two minis next to it."""
        taken = {(q["x"], q["y"]) for q in self.aliens}
        for dx in (-1, 1):
            nx = a["x"] + dx
            for cx in (nx, a["x"]):
                if 0 <= cx < W and (cx, a["y"]) not in taken:
                    self.aliens.append(self._new_alien(cx, a["y"], "mini", self.next_id))
                    self.next_id += 1
                    taken.add((cx, a["y"]))
                    break

    def _abilities(self):
        """Healer shields, phantom teleports."""
        taken = {(q["x"], q["y"]) for q in self.aliens}
        for a in self.aliens:
            if a["kind"] == "healer" and self.t % HEAL_EVERY == a["id"] % HEAL_EVERY:
                allies = [q for q in self.aliens if q is not a and not q["shield"]
                          and abs(q["x"] - a["x"]) + abs(q["y"] - a["y"]) <= HEAL_RANGE]
                if allies:
                    tgt = max(allies, key=lambda q: (q["kind"] == "titan", q["hp"]))
                    tgt["shield"] = True
                    self.stats["shields_cast"] += 1
            if a["kind"] == "phantom" and not a.get("kamikaze"):
                a["timer"] += 1
                if a["timer"] >= PHANTOM_EVERY:
                    a["timer"] = 0
                    for _ in range(20):
                        nx = int(self.rng.integers(W))
                        ny = int(self.rng.integers(ALIEN_TOP, ALIEN_BOTTOM + 1))
                        if (nx, ny) not in taken:
                            taken.discard((a["x"], a["y"]))
                            a["x"], a["y"] = nx, ny
                            taken.add((nx, ny))
                            break

    # ------------------------------------------------------------ movement --
    def _alien_goal(self, a):
        """Where this alien wants to go (its kind's plan)."""
        sx, sy = self.sx, self.sy
        k = a["kind"]
        if k in ("splitter", "phantom"):               # wander in the middle rows
            if self.rng.random() < 0.3:
                return a["x"] + int(self.rng.integers(-2, 3)), int(self.rng.integers(2, 6))
            return a["x"], a["y"]
        if k == "mini":                                # swarm: jitter around the ship's column
            return sx + int(self.rng.integers(-2, 3)), int(self.rng.integers(4, ALIEN_BOTTOM + 1))
        if k == "healer":                              # hang back, away from the ship's column,
            others = [q for q in self.aliens if q is not a]   # close to its allies
            cx = int(np.mean([q["x"] for q in others])) if others else a["x"]
            far = [x for x in range(W) if abs(x - sx) >= 3] or [a["x"]]
            return min(far, key=lambda x: abs(x - cx)), ALIEN_TOP
        if k == "gunner":                              # keep a firing line from afar
            gy = max(ALIEN_TOP, sy - 7)
            d = sy - gy
            lines = [x for x in (sx, sx - d, sx + d) if 0 <= x < W]
            return min(lines, key=lambda x: abs(x - a["x"])), gy
        if k == "titan":                               # hover over the ship, mid-height
            return sx, 4
        # diver: swoop to the edge of the player zone above the ship, then back
        if a["diving"] and a["y"] >= ALIEN_BOTTOM:
            a["diving"] = False
        elif not a["diving"] and a["y"] <= ALIEN_TOP + 1:
            a["diving"] = True
        return (sx, ALIEN_BOTTOM) if a["diving"] else (a["x"], ALIEN_TOP)

    def _kamikaze_step(self, a, taken):
        """Dive one row into the player zone, homing in on the ship's column."""
        x, y = a["x"], a["y"] + 1
        if x != self.sx and self.rng.random() < 0.7:
            x += 1 if self.sx > x else -1
        taken.discard((a["x"], a["y"]))
        if y > SHIP_ROW:                                      # missed: loop back to the top
            a.update(x=int(self.rng.integers(W)), y=ALIEN_TOP, kamikaze=False, diving=False)
        elif (x, y) in self.bunkers:                          # crashes into a block
            self.bunkers.pop((x, y))
            self.aliens.remove(a)
            self.stats["crashed"] += 1
            return
        else:
            a["x"], a["y"] = x, y
        if (a["x"], a["y"]) == (self.sx, self.sy):            # rams the ship
            self.aliens.remove(a)
            self.rammed_now = True
            return
        taken.add((a["x"], a["y"]))

    def _diver_on_ship(self):
        """The ship flew into a kamikaze diver."""
        for a in self.aliens:
            if a.get("kamikaze") and (a["x"], a["y"]) == (self.sx, self.sy):
                self.aliens.remove(a)
                return True
        return False

    def _move_aliens(self):
        speed_up = MAPS[self.map]["speed"]
        taken = {(a["x"], a["y"]) for a in self.aliens}
        for a in list(self.aliens):
            if a not in self.aliens:
                continue
            every = max(1, KIND_MOVE_EVERY[a["kind"]] - speed_up)
            if (self.t + a["id"]) % every:
                continue
            x, y = a["x"], a["y"]
            boxed = self.phase == "smart" and len(self.escape_cells()) <= 1
            if a["kind"] == "diver" and not a.get("kamikaze") \
                    and y >= ALIEN_BOTTOM - (4 if boxed else 2) and abs(x - self.sx) <= 3 \
                    and self.rng.random() < KAMIKAZE_P:
                a["kamikaze"] = True                          # commit to ramming the ship
            if a.get("kamikaze"):
                self._kamikaze_step(a, taken)
                continue
            opts = [(x + dx, y + dy) for dx, dy in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
                    if 0 <= x + dx < W and ALIEN_TOP <= y + dy <= ALIEN_BOTTOM
                    and ((x + dx, y + dy) == (x, y) or (x + dx, y + dy) not in taken)]
            in_line = lambda cx, cy: any(pb[0] == cx and pb[1] > cy for pb in self.pbullets)
            threatened = any(pb[0] == x and 0 < pb[1] - y <= 5 for pb in self.pbullets)
            if threatened and self.rng.random() < KIND_EVADE[a["kind"]]:
                safe = [o for o in opts if not in_line(*o)]           # dodge
                if safe:
                    nx, ny = safe[self.rng.integers(len(safe))]
                    self.stats["dodges"] += (nx, ny) != (x, y)
                else:
                    nx, ny = x, y
            elif a["kind"] not in ("titan", "phantom") and x == self.sx and self.sy - y <= 6 \
                    and (x, y - 1) in opts and self.rng.random() < 0.6:
                nx, ny = x, y - 1                                     # back off
            else:
                gx, gy = self._alien_goal(a)
                nx, ny = min(opts, key=lambda o: abs(o[0] - gx) + abs(o[1] - gy)
                             + 0.01 * self.rng.random())
            taken.discard((x, y))
            taken.add((nx, ny))
            a["x"], a["y"] = nx, ny
