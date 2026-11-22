"""
Every number of the game in one place: the grid, the maps, the alien kinds,
the rewards, the ammunition and the action set.

MDP view [Zhao 1.2, 1.5]: ACTIONS is the action set A (|A| = 7); the R_*
constants and KIND_REWARD define the reward r(s, a, s'); everything else
parametrises the transition p(s' | s, a).
"""

# --------------------------------------------------------------- the grid --
W, H = 11, 15                       # columns x rows, row 0 is the top
SHIP_ROW = H - 1                    # bottom row
ZONE_TOP = H - 6                    # ship may move in rows ZONE_TOP..SHIP_ROW
ALIEN_TOP, ALIEN_BOTTOM = 1, ZONE_TOP - 1   # aliens live in these rows
BUNKER_ROWS = (SHIP_ROW - 4, SHIP_ROW - 3)
BUNKER_HP = 3                       # max block strength (for features / drawing)

# --------------------------------------------------------------- the maps --
# fortress: three wide bunkers, normal alien speed
# open:     no bunkers at all, aliens move faster
# pillars:  six thin, weaker pillars with narrow gaps to shoot through
MAPS = {
    "fortress": dict(cols=(1, 2, 5, 8, 9), hp=3, speed=0),
    "open":     dict(cols=(), hp=3, speed=1),            # aliens move faster
    "pillars":  dict(cols=(0, 3, 4, 6, 7, 10), hp=2, speed=0),
}

# ---------------------------------------------- hidden phase and actions --
# The phase is the hidden mode xi: it changes how the aliens fire and is
# never part of the observation (see alien_fire.py).
PHASES = ("rain", "sniper", "trap", "smart")
ACTIONS = ("left", "right", "up", "down", "stay", "shoot", "grenade")
MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0), (0, 0), (0, 0))

# ------------------------------------------------------------- the aliens --
N_ALIENS = 5                        # aliens per wave
KINDS = ("gunner", "diver", "titan", "splitter", "healer", "phantom", "mini")
SPAWN_KINDS = ("gunner", "diver", "titan", "splitter", "healer", "phantom")
KIND_P = (0.2, 0.2, 0.15, 0.2, 0.1, 0.15)   # wave composition probabilities
KIND_HP = {"gunner": 1, "diver": 1, "titan": 6, "splitter": 2, "healer": 2,
           "phantom": 1, "mini": 1}
MAX_HP = 6
KIND_REWARD = {"gunner": 1.5, "diver": 1.5, "titan": 5.0, "splitter": 2.0, "healer": 2.5,
               "phantom": 2.0, "mini": 0.5}
KIND_FIRE = {"gunner": 2.5, "diver": 0.7, "titan": 0.0, "splitter": 1.0, "healer": 0.0,
             "phantom": 1.0, "mini": 0.5}
KIND_MOVE_EVERY = {"gunner": 2, "diver": 1, "titan": 4, "splitter": 2, "healer": 2,
                   "phantom": 2, "mini": 1}
KIND_EVADE = {"gunner": 0.7, "diver": 0.8, "titan": 0.0, "splitter": 0.3, "healer": 0.6,
              "phantom": 0.0, "mini": 0.6}
HEAL_EVERY, HEAL_RANGE = 12, 6          # healer: shield an ally within range every N steps
PHANTOM_EVERY = 6                       # phantom: teleport every N steps
KAMIKAZE_P = 0.8                        # chance a diver near the ship goes kamikaze
ESCALATION, MAX_AGGRO = 0.15, 2.5       # each wave +15 % aggression (smart phase), capped
TITAN_VOLLEY_P = 0.3                    # chance per step that a titan fires a straight shot

# ------------------------------------------------------------- the reward --
# Damage-based scoring: R_DMG per point of damage actually dealt, plus a kill
# bonus KIND_REWARD[k] - R_DMG * KIND_HP[k], so destroying an alien is worth
# exactly KIND_REWARD[k] whatever weapon was used (see weapons.py).
R_HIT, R_OWN_BLOCK = -10.0, -0.2        # ship hit; own bullet hits own bunker
R_DMG = 0.5                             # per point of damage actually dealt
R_BONUS = 0.5                           # small nudge; the real value is the effect
R_SHOT, R_GRENADE, R_DRY = -0.02, -0.1, -0.05   # ammunition has a price

# ---------------------------------------------------- bonuses and ammunition --
BONUSES = ("life", "gun", "repair", "ammo", "grenades")
BONUS_P = (0.15, 0.15, 0.15, 0.35, 0.20)
BONUS_DROP_P = 0.025                    # chance per step that the game releases a bonus
START_AMMO, MAX_AMMO, AMMO_CRATE = 25, 60, 15          # light shots
START_GRN, MAX_GRN, GRN_CRATE, GRN_DMG = 4, 10, 3, 3    # grenades (centre damage)
GRN_RADIUS = 2                                          # blast radius (cells)
START_LIVES, MAX_LIVES = 3, 9
GUN_STEPS, GUN_PAIRS = 120, 2           # dual-shot bonus: duration, pairs in the air
