"""
The arena: where the ship may stand, and the bunkers.

  * The ship moves left / right / up / down inside the player zone
    (rows ZONE_TOP..SHIP_ROW) and cannot enter bunker blocks.
  * Bunkers stop bullets from both sides and wear out. They are built once
    at the start of an episode; damage persists across waves (only the
    'repair' bonus restores them, see bonuses.py).

Part of the state s [Zhao 1.2]: the ship position (sx, sy) and the dict
bunkers {(x, y): strength}.
"""
from .config import BUNKER_ROWS, MAPS, SHIP_ROW, W, ZONE_TOP


class ArenaMixin:
    def _build_bunkers(self):
        m = MAPS[self.map]
        self.bunkers = {(x, y): m["hp"] for x in m["cols"] for y in BUNKER_ROWS}

    def set_map(self, name):
        """Change the map; bunkers are rebuilt immediately."""
        self.map = name
        self._build_bunkers()
        if (self.sx, self.sy) in self.bunkers:
            self.sy = SHIP_ROW

    def free(self, x, y):
        """Can the ship stand in (x, y)?"""
        return 0 <= x < W and ZONE_TOP <= y <= SHIP_ROW and (x, y) not in self.bunkers

    def covered(self):
        """Is there a bunker block above the ship (between it and the aliens)?"""
        return any((self.sx, y) in self.bunkers for y in BUNKER_ROWS if y < self.sy)

    def _hit_bunker(self, x, y, dmg=1):
        """A bullet reaches (x, y): wear the block down. True if there was one."""
        if (x, y) in self.bunkers:
            self.bunkers[(x, y)] -= dmg
            if self.bunkers[(x, y)] <= 0:
                del self.bunkers[(x, y)]
            return True
        return False
