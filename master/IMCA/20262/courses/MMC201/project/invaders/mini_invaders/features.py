"""
The observation phi(s) in R^311: what the agent sees.

This plays the role of the feature vector phi(s) of [Zhao 8.2.3]; the critic
v(s, w) and the policy pi(a|s, theta) are functions of phi(s) [Zhao 8.1, 9.1].
phi is ego-centric (everything relative to the ship) and drops information:
the hidden phase, far bullets, the aliens' timers. States with equal features
are treated alike; see the report, remark on partial observability, and its
appendix "Feature map" for the table of the blocks A..M below.
"""
import numpy as np

from .config import (BONUSES, BUNKER_HP, GRN_DMG, GUN_PAIRS, GUN_STEPS, KINDS,
                     MAX_AMMO, MAX_GRN, MAX_HP, MAX_LIVES, MOVES, N_ALIENS,
                     SHIP_ROW, W, ZONE_TOP)

NK = len(KINDS)
BLOCKS = (                                   # (block, size), in order
    ("A bullets near the ship", 105), ("B look-ahead danger", 10),
    ("C bullets further up", 5), ("D ship position", 17),
    ("E bunkers near the ship", 30), ("F bonus position", 18),
    ("G nearest alien", 11), ("H cover, own bullets, bias", 3),
    ("I bonus type", 5), ("I lives", 6), ("I gun", 2), ("I ammunition", 5),
    ("J aliens by column", 7 * (4 + NK)), ("K nearest alien kind", NK + 2),
    ("L wave composition", NK), ("M wave number", 1),
)
N_FEAT = sum(n for _, n in BLOCKS)           # = 311


def features(env):
    """phi(s) for the current state of `env` (a float32 vector of length N_FEAT)."""
    f = np.zeros(N_FEAT, dtype=np.float32)
    sx, sy = env.sx, env.sy
    i = 0
    # A. bullets around the ship: cols -2..2, rows -5..+1, one channel per drift (105)
    far = np.zeros(5)
    for bx, by, bdx in env.bullets:
        ddx, ddy = bx - sx, by - sy
        if -2 <= ddx <= 2:
            if -5 <= ddy <= 1:
                f[i + ((ddx + 2) * 7 + (ddy + 5)) * 3 + (bdx + 1)] = 1.0
            elif ddy < -5:
                far[ddx + 2] += 1
    i += 105
    # B. look-ahead danger per move target x (1, 2 steps) (10)
    for m in range(5):
        tx, ty = sx + MOVES[m][0], sy + MOVES[m][1]
        if not env.free(tx, ty):
            tx, ty = sx, sy                              # blocked move = stay
        f[i + 2 * m] = float(env.threat_at(tx, ty, 1))
        f[i + 2 * m + 1] = float(env.threat_at(tx, ty, 2))
    i += 10
    # C. bullets further up in the 5 nearby columns (5)
    f[i:i + 5] = np.minimum(far, 3) / 3
    i += 5
    # D. ship position: column (11) and row within the zone (6)
    f[i + sx] = 1.0
    f[i + W + (sy - ZONE_TOP)] = 1.0
    i += 17
    # E. bunkers / walls around the ship: cols -2..2, rows -3..+2 (30)
    for ddx in range(-2, 3):
        for ddy in range(-3, 3):
            x, y = sx + ddx, sy + ddy
            v = env.bunkers.get((x, y), 0) / BUNKER_HP
            if not 0 <= x < W or y > SHIP_ROW:
                v = 1.0
            f[i + (ddx + 2) * 6 + (ddy + 3)] = v
    i += 30
    # F. bonus: present, relative column (11), relative row (6)
    if env.pu is not None:
        f[i] = 1.0
        f[i + 1 + int(np.clip(env.pu[0] - sx, -5, 5)) + 5] = 1.0
        rel = int(env.pu[1]) - sy
        b = 0 if rel < -8 else 1 if rel < -5 else 2 if rel < -2 else 3 if rel < 0 else 4 if rel == 0 else 5
        f[i + 12 + b] = 1.0
    i += 18
    # G. nearest alien: relative column (7) and how far up it is (4)
    near = None
    if env.aliens:
        near = min(env.aliens, key=lambda al: (abs(al["x"] - sx), sy - al["y"]))
        f[i + int(np.clip(near["x"] - sx, -3, 3)) + 3] = 1.0
        gap = sy - near["y"]
        f[i + 7 + min(max(gap - 1, 0) // 4, 3)] = 1.0
    i += 11
    # H. own bunker above me, own bullets in flight, bias (3)
    f[i] = float(env.covered())
    f[i + 1] = len(env.pbullets) / (2 * GUN_PAIRS)
    f[i + 2] = 1.0
    i += 3
    # I. bonus type (5), lives (one-hot up to 5, plus scaled count), gun (2),
    #    ammunition: light level, light empty, grenade level, grenade empty,
    #    grenade in the air (5)
    if env.pu is not None:
        f[i + BONUSES.index(env.pu[2])] = 1.0
    i += 5
    f[i + min(max(env.lives, 1), 5) - 1] = 1.0
    f[i + 5] = env.lives / MAX_LIVES
    i += 6
    f[i] = float(env.gun > 0)
    f[i + 1] = env.gun / GUN_STEPS
    i += 2
    f[i] = env.ammo_l / MAX_AMMO
    f[i + 1] = float(env.ammo_l == 0)
    f[i + 2] = env.ammo_g / MAX_GRN
    f[i + 3] = float(env.ammo_g == 0)
    f[i + 4] = float(any(pb[2] == GRN_DMG for pb in env.pbullets))
    i += 5
    # J. aliens by column, cols -3..+3 around the ship (7 x (4 + kinds)):
    #    present, how far up, armour, shielded, kind (one-hot)
    for ddx in range(-3, 4):
        col = [al for al in env.aliens if al["x"] == sx + ddx]
        if col:
            al = max(col, key=lambda q: q["y"])           # the lowest one in that column
            j = i + (ddx + 3) * (4 + NK)
            f[j] = 1.0
            f[j + 1] = (sy - al["y"]) / SHIP_ROW
            f[j + 2] = al["hp"] / MAX_HP
            f[j + 3] = float(al["shield"])
            f[j + 4 + KINDS.index(al["kind"])] = 1.0
    i += 7 * (4 + NK)
    # K. kind of the nearest alien (one-hot), its armour, shielded
    if near is not None:
        f[i + KINDS.index(near["kind"])] = 1.0
        f[i + NK] = near["hp"] / MAX_HP
        f[i + NK + 1] = float(near["shield"])
    i += NK + 2
    # L. wave composition: how many of each kind are left
    for al in env.aliens:
        f[i + KINDS.index(al["kind"])] += 1 / N_ALIENS
    i += NK
    # M. wave number (later waves are more aggressive) (1)
    f[i] = min(env.wave, 12) / 12
    assert i + 1 == N_FEAT
    return f
