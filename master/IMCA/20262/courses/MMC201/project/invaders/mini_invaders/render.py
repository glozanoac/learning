"""
Drawing a frame with PIL: used by the viewer (watch.py) and the GIF maker.
Nothing here affects the dynamics or what the agent sees.
"""
import numpy as np

from .config import GRN_DMG, GRN_RADIUS, H, MAPS, MAX_HP, PHANTOM_EVERY, W, ZONE_TOP

KIND_COLOR = {"gunner": (231, 76, 60), "diver": (243, 156, 18), "titan": (46, 139, 87),
              "splitter": (236, 112, 175), "healer": (72, 201, 216), "phantom": (210, 200, 255),
              "mini": (245, 160, 200)}


def _draw_alien(d, cx, cy, kind, hp, diving=False, cell=32):
    col = KIND_COLOR[kind]
    bg = (12, 14, 32)
    u = cell / 32                                      # scale with the cell size
    if kind == "titan":                                # big armoured turtle + HP bar
        shell, dark, skin = (46, 139, 87), (20, 80, 45), (163, 196, 91)
        w2, h2 = 18 * u, 11 * u
        for lx, ly in ((-14, 6), (14, 6), (-12, -7), (12, -7)):   # four flippers
            d.ellipse([cx + (lx - 5) * u, cy + (ly - 4) * u, cx + (lx + 5) * u, cy + (ly + 4) * u],
                      fill=skin)
        d.ellipse([cx - 5 * u, cy + 8 * u, cx + 5 * u, cy + 17 * u], fill=skin)  # head (facing down)
        d.ellipse([cx - 3 * u, cy + 12 * u, cx - 1 * u, cy + 14 * u], fill=bg)  # eyes
        d.ellipse([cx + 1 * u, cy + 12 * u, cx + 3 * u, cy + 14 * u], fill=bg)
        d.ellipse([cx - w2, cy - h2, cx + w2, cy + h2], fill=shell, outline=dark, width=2)
        for k in (-1, 0, 1):                           # shell plates
            d.regular_polygon((cx + 9 * k * u, cy - (0 if k else 2) * u, 4.5 * u), 6,
                              fill=dark if k == 0 else (36, 110, 65), outline=dark)
        bw = 2 * w2
        d.rectangle([cx - w2, cy - h2 - 8 * u, cx + w2, cy - h2 - 4 * u], fill=(60, 20, 20))
        d.rectangle([cx - w2, cy - h2 - 8 * u, cx - w2 + bw * hp / MAX_HP, cy - h2 - 4 * u],
                    fill=(46, 204, 113))
        return
    if kind == "diver":
        if diving:                                     # nose-down dart
            d.polygon([(cx, cy + 12), (cx - 11, cy - 8), (cx, cy - 3), (cx + 11, cy - 8)], fill=col)
        else:
            d.polygon([(cx, cy + 9), (cx - 12, cy - 7), (cx + 12, cy - 7)], fill=col)
            d.rectangle([cx - 2, cy - 4, cx + 2, cy], fill=bg)
        return
    if kind == "splitter":                             # pink blob with a seam
        d.ellipse([cx - 12, cy - 10, cx + 12, cy + 10], fill=col)
        d.line([cx, cy - 10, cx, cy + 10], fill=(150, 50, 100), width=2)
        d.ellipse([cx - 7, cy - 3, cx - 3, cy + 1], fill=bg)
        d.ellipse([cx + 3, cy - 3, cx + 7, cy + 1], fill=bg)
        if hp < 2:
            d.line([cx - 9, cy - 6, cx - 2, cy + 6], fill=bg, width=2)
        return
    if kind == "mini":                                 # small pink blob
        d.ellipse([cx - 7, cy - 6, cx + 7, cy + 6], fill=col)
        d.ellipse([cx - 3, cy - 2, cx - 1, cy], fill=bg)
        d.ellipse([cx + 1, cy - 2, cx + 3, cy], fill=bg)
        return
    if kind == "healer":                               # cyan diamond with a cross + halo
        d.ellipse([cx - 9, cy - 16, cx + 9, cy - 12], outline=(255, 255, 160), width=2)
        d.polygon([(cx, cy - 12), (cx + 12, cy), (cx, cy + 12), (cx - 12, cy)], fill=col)
        d.rectangle([cx - 2, cy - 7, cx + 2, cy + 7], fill=(255, 255, 255))
        d.rectangle([cx - 7, cy - 2, cx + 7, cy + 2], fill=(255, 255, 255))
        if hp < 2:
            d.line([cx + 3, cy - 9, cx + 9, cy + 1], fill=bg, width=2)
        return
    if kind == "phantom":                              # ghost (round top, wavy bottom)
        flicker = diving                               # 'diving' carries the flicker flag
        c = (120, 110, 170) if flicker else col
        d.pieslice([cx - 11, cy - 12, cx + 11, cy + 10], 180, 360, fill=c)
        d.rectangle([cx - 11, cy - 1, cx + 11, cy + 7], fill=c)
        for k in range(4):
            x0 = cx - 11 + k * 5.5
            d.polygon([(x0, cy + 7), (x0 + 5.5, cy + 7), (x0 + 2.75, cy + 12)], fill=c)
        d.ellipse([cx - 6, cy - 5, cx - 2, cy], fill=bg)
        d.ellipse([cx + 2, cy - 5, cx + 6, cy], fill=bg)
        return
    d.rectangle([cx - 11, cy - 7, cx + 11, cy + 7], fill=col)
    d.rectangle([cx - 6, cy - 3, cx - 3, cy], fill=bg)
    d.rectangle([cx + 3, cy - 3, cx + 6, cy], fill=bg)
    d.line([cx - 11, cy + 7, cx - 14, cy + 12], fill=col, width=2)
    d.line([cx + 11, cy + 7, cx + 14, cy + 12], fill=col, width=2)
    if kind == "gunner":                               # two cannons
        d.rectangle([cx - 9, cy + 7, cx - 6, cy + 13], fill=(255, 200, 200))
        d.rectangle([cx + 6, cy + 7, cx + 9, cy + 13], fill=(255, 200, 200))


def render(env, title="", cell=32):
    from PIL import Image, ImageDraw, ImageFont
    try:
        f1, f2 = ImageFont.load_default(size=15), ImageFont.load_default(size=13)
    except TypeError:                                  # old Pillow
        f1 = f2 = ImageFont.load_default()
    hud = 46
    img = Image.new("RGB", (W * cell, H * cell + hud), (12, 14, 32))
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(7)
    for _ in range(40):                                # stars
        x, y = rng.integers(0, W * cell), rng.integers(hud, H * cell + hud)
        d.point((x, y), fill=(90, 90, 120))
    oy = hud
    zy = ZONE_TOP * cell + oy                          # player zone
    d.rectangle([0, zy, W * cell, H * cell + oy], fill=(18, 22, 46))
    for x0 in range(0, W * cell, 12):
        d.line([x0, zy, x0 + 6, zy], fill=(60, 70, 110))
    for al in env.aliens:                              # aliens (anywhere in their zone)
        flag = (al["timer"] >= PHANTOM_EVERY - 1) if al["kind"] == "phantom" \
            else (al.get("diving", False) or al.get("kamikaze", False))
        _draw_alien(d, al["x"] * cell + cell // 2, al["y"] * cell + cell // 2 + oy,
                    al["kind"], al["hp"], flag, cell)
        if al["shield"]:                               # shield bubble
            cx, cy = al["x"] * cell + cell // 2, al["y"] * cell + cell // 2 + oy
            d.ellipse([cx - cell * 0.55, cy - cell * 0.55, cx + cell * 0.55, cy + cell * 0.55],
                      outline=(120, 200, 255), width=2)
        if al.get("kamikaze"):                         # flame trail
            cx, cy = al["x"] * cell + cell // 2, al["y"] * cell + oy
            d.polygon([(cx - 5, cy), (cx + 5, cy), (cx, cy - 12)], fill=(255, 90, 30))
    for (bx, by), hp in env.bunkers.items():           # bunkers
        x0, y0 = bx * cell + 1, by * cell + oy + 1
        shade = [(40, 90, 80), (60, 150, 120), (80, 200, 160)][min(hp, 3) - 1]
        d.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=shade)
        for _ in range(MAPS[env.map]["hp"] - hp):      # cracks
            px, py = x0 + rng.integers(4, cell - 6), y0 + rng.integers(4, cell - 6)
            d.rectangle([px, py, px + 4, py + 4], fill=(12, 14, 32))
    for bx, by, bdx in env.bullets:                    # alien bullets (slanted if diagonal)
        cx, cy = bx * cell + cell // 2, by * cell + cell // 2 + oy
        d.line([cx - 5 * bdx, cy - 7, cx + 5 * bdx, cy + 7],
               fill=(231, 76, 60) if bdx == 0 else (255, 105, 180), width=4)
    for px_, py_, dmg in env.pbullets:                 # player bullets / grenades
        cx, cy = px_ * cell + cell // 2, py_ * cell + cell // 2 + oy
        if dmg == GRN_DMG:                             # grenade: just a dot
            rr = cell * 0.2
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(170, 220, 120))
        else:
            d.rectangle([cx - 2, cy - 8, cx + 2, cy + 8],
                        fill=(255, 140, 0) if env.gun else (241, 196, 15))
    for bx, by, k in env.blasts:                       # grenade blast area (radius 2 cells)
        cx, cy = bx * cell + cell // 2, by * cell + cell // 2 + oy
        rad = cell * (GRN_RADIUS + 0.5) * (1.15 - 0.25 * k)
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=(255, 160, 40), width=4)
        d.ellipse([cx - rad / 2, cy - rad / 2, cx + rad / 2, cy + rad / 2], fill=(255, 220, 120))
    if env.pu is not None:                             # bonus
        cx, cy = env.pu[0] * cell + cell // 2, int(env.pu[1] * cell) + cell // 2 + oy
        col, ch = {"life": ((236, 64, 122), "L"), "gun": ((255, 152, 0), "G"),
                   "repair": ((38, 198, 218), "R"), "ammo": ((212, 172, 13), "A"),
                   "grenades": ((39, 84, 47), "Gr")}[env.pu[2]]
        d.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=col, outline=(255, 255, 255))
        d.text((cx - 4 * len(ch), cy - 8), ch, fill=(255, 255, 255), font=f2)
    cx, cy = env.sx * cell + cell // 2, env.sy * cell + cell // 2 + oy
    if env.last_hit:                                   # explosion flash
        d.ellipse([cx - 20, cy - 20, cx + 20, cy + 20], fill=(255, 120, 40))
    ship_col = (231, 76, 60) if env.last_hit else (46, 204, 113)
    d.polygon([(cx, cy - 12), (cx - 13, cy + 10), (cx + 13, cy + 10)], fill=ship_col)
    if env.gun:                                        # twin cannons while the gun is active
        d.rectangle([cx - 10, cy - 6, cx - 7, cy + 4], fill=(255, 140, 0))
        d.rectangle([cx + 7, cy - 6, cx + 10, cy + 4], fill=(255, 140, 0))
    d.text((6, 4), title, fill=(255, 255, 255), font=f1)
    s = env.stats
    gun = f"  2xGUN {env.gun}" if env.gun else ""
    low = (231, 76, 60) if env.ammo_l == 0 else (200, 200, 220)
    d.text((6, 24), f"score {s['ret']:+.0f}  lives {env.lives}  ammo {env.ammo_l}  "
                    f"gren {env.ammo_g}  wave {env.wave}{gun}", fill=low, font=f2)
    return img
