"""
Scripted players, used as baselines in the report (Greedy and Cautious).
They read the true state, including bullets beyond the agent's view.
"""
import numpy as np

from .config import BUNKER_ROWS, MOVES, SHIP_ROW


def heuristic(env, go_for_powerups):
    """Avoid cells that a bullet will reach in the next 2 steps, chase a
    target, shoot when an alien is in a clear line of fire.

    go_for_powerups=True  -> 'Greedy': goes for every falling bonus and,
                             when it is about to land, commits even under fire.
    go_for_powerups=False -> 'Cautious': ignores bonuses, stays low.
    """
    sx, sy = env.sx, env.sy
    xs = env.alien_xs
    commit = False
    if go_for_powerups and env.pu is not None:
        tx, ty = env.pu[0], SHIP_ROW
        commit = SHIP_ROW - env.pu[1] <= 3
    else:
        def clear(x):
            return not any((x, y) in env.bunkers for y in BUNKER_ROWS)
        cand = [x for x in xs if clear(x)] or xs or [sx]
        tx, ty = min(cand, key=lambda x: abs(x - sx)), SHIP_ROW
    best, best_score = 4, np.inf
    for m in range(5):
        nx, ny = sx + MOVES[m][0], sy + MOVES[m][1]
        if not env.free(nx, ny):
            continue
        danger = 0 if commit else 100 * env.threat_at(nx, ny, 2)
        score = danger + abs(nx - tx) + abs(ny - ty) + 0.1 * (m != 4)
        if score < best_score:
            best, best_score = m, score
    if best == 4 and sx in xs and not env.covered():
        target = max((al for al in env.aliens if al["x"] == sx), key=lambda q: q["y"])
        if target["hp"] >= 2 and env.can_fire_grenade():      # grenades for armour
            return 6
        if env.can_fire_light():
            return 5
    return best
