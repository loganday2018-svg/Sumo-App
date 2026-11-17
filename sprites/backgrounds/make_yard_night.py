"""Stable yard at night, 240x160 GBA-style background for the teppo mini-game."""
import os
import numpy as np
from PIL import Image

W, H = 240, 160
OUT = os.path.dirname(os.path.abspath(__file__))
img = np.zeros((H, W, 3), dtype=np.uint8)

def C(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

# palette
SKY = [C(x) for x in ('050918', '0a1030', '0f1844', '152356', '1c2e66', '253c74', '2f4a82', '3a5890', '456699')]
HILL1, HILL2 = C('172052'), C('0e1638')
ROOF = [C('0a0e1e'), C('121a32'), C('1c2846'), C('4a6090')]
WALL = [C('2c3658'), C('333e62'), C('3e4a72')]
WALLSH = C('1f2842')
BEAM = [C('121828'), C('1c2440')]
SHOJI = C('9a7a58')
SHOJI_L = C('e8c078')
FENCE = [C('161c34'), C('222a48'), C('2c3656')]
LEAF = [C('0c1a2a'), C('122836'), C('1c3c42'), C('4a7a78')]
TRUNK = [C('10142a'), C('222c46')]
LAMP = [C('b8482c'), C('e8802c'), C('f8c860'), C('fff0b0')]
LAMPD = C('3a2030')
GROUND = [C('151c3a'), C('1e2848'), C('28345a'), C('33426c'), C('3c4c78')]
STONE = [C('222a4a'), C('3c4870')]
DARK = SKY[0]
STAR = [C('b8c8e8'), C('6a82b8')]
MOON = [C('f0f4ff'), C('b8c8e8')]
WGLOW = [C('564a5c'), C('7a5c5a')]   # warm light on wall
GGLOW = [C('443f58'), C('5e4c58')]   # warm light on ground

BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])

def px(x, y, c):
    if 0 <= x < W and 0 <= y < H:
        img[y, x] = c

def rect(x0, y0, x1, y1, c):
    img[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)] = c

def dither(x0, y0, x1, y1, ca, cb, t0, t1, vertical=True):
    """Ordered-dither between ca and cb; t ramps 0..1 over the span."""
    for y in range(y0, y1):
        for x in range(x0, x1):
            t = (y - y0) / max(1, y1 - y0 - 1) if vertical else (x - x0) / max(1, x1 - x0 - 1)
            t = t0 + (t1 - t0) * t
            img[y, x] = cb if (BAYER[y % 4, x % 4] + 0.5) / 16 < t else ca

rng = np.random.RandomState(7)

# ---- sky: bands with dither seams ----
bands = [0, 8, 15, 21, 27, 32, 37, 41, 45]
for i in range(len(SKY) - 1):
    y0 = bands[i]
    y1 = bands[i + 1] if i + 1 < len(bands) else 70
    rect(0, y0, W, y1, SKY[i])
    # 3-row dither seam into the next band
    for y in range(y1 - 3, y1):
        for x in range(W):
            if BAYER[y % 4, x % 4] / 16 < (y - (y1 - 3) + 1) / 4.5:
                img[y, x] = SKY[i + 1]
rect(0, 45, W, 100, SKY[-1])
# stars: dense at the top, thinning toward the horizon
for _ in range(70):
    x, y = rng.randint(0, W), rng.randint(2, 40)
    if rng.rand() < (1.0 - y / 46.0):
        px(x, y, STAR[0] if rng.rand() < 0.3 else STAR[1])
# moon: crescent with dithered halo, upper right
MX, MY, MR = 206, 30, 7
for y in range(MY - 14, MY + 15):
    for x in range(MX - 16, MX + 17):
        d = ((x - MX) ** 2 + (y - MY) ** 2) ** 0.5
        if MR < d < 15 and BAYER[y % 4, x % 4] / 16 < (15 - d) / 15.0 * 0.4 and 0 <= y < 45:
            px(x, y, SKY[4])
for y in range(MY - MR, MY + MR + 1):
    for x in range(MX - MR, MX + MR + 1):
        if (x - MX) ** 2 + (y - MY) ** 2 <= MR * MR:
            # cut-out circle shifted up-right leaves a crescent on the lower left
            if (x - MX - 3) ** 2 + (y - MY + 2) ** 2 > (MR - 1) ** 2:
                px(x, y, MOON[0] if (x - MX) ** 2 + (y - MY) ** 2 < (MR - 1) ** 2 else MOON[1])
# thin moonlit cloud streaks
for (cx, cy, l) in [(30, 33, 26), (150, 28, 30), (95, 38, 18), (60, 22, 20)]:
    for i in range(l):
        px(cx + i, cy, SKY[4])
        if 3 < i < l - 3: px(cx + i, cy + 1, SKY[5])
# distant hills
for x in range(W):
    h = int(41 + 4 * np.sin(x / 19.0) + 3 * np.sin(x / 7.0 + 1))
    rect(x, h, x + 1, 100, HILL1)
    if x % 2 == 0 or h > 66:
        pass
for x in range(W):
    h = int(45 + 3 * np.sin(x / 13.0 + 2))
    rect(x, h, x + 1, 100, HILL2)

# ---- building ----
RY0, RY1 = 49, 65  # roof span
for y in range(RY0, RY1):
    # trapezoid widening, full width (building spans screen)
    for x in range(W):
        row = y - RY0
        c = ROOF[1] if row % 4 != 3 else ROOF[0]
        if (x + (row // 4) * 3) % 6 == 0 and row % 4 != 3:
            c = ROOF[2]
        img[y, x] = c
# ridge cap and eave
rect(0, RY0 - 3, W, RY0, ROOF[3])
rect(0, RY0 - 3, W, RY0 - 2, ROOF[2])
for x in range(0, W, 8):
    rect(x, RY0 - 5, x + 5, RY0 - 3, ROOF[2])
rect(0, RY1, W, RY1 + 2, ROOF[0])
rect(0, RY1 + 2, W, RY1 + 3, BEAM[1])
# eave shadow onto wall
dither(0, RY1 + 3, W, RY1 + 9, WALLSH, WALL[0], 0.0, 1.0)
# wall
rect(0, RY1 + 9, W, 90, WALL[0])
for y in range(RY1 + 9, 90):
    for x in range(W):
        if BAYER[y % 4, x % 4] < 2 and (y > 72):
            img[y, x] = WALL[1]
# base stone
rect(0, 88, W, 92, STONE[0])
for x in range(0, W, 12):
    rect(x, 88, x + 1, 92, BEAM[0])
rect(0, 88, W, 89, STONE[1])
# posts
for x in range(8, W, 46):
    rect(x, RY1 + 3, x + 5, 88, BEAM[1])
    rect(x, RY1 + 3, x + 1, 88, BEAM[0])
    rect(x + 4, RY1 + 3, x + 5, 88, BEAM[0])
# shoji windows between posts, kept dim
for x0 in (18, 202):
    x0 += 0
    rect(x0, 71, x0 + 26, 87, BEAM[0])
    rect(x0 + 1, 72, x0 + 25, 86, WALL[2] if x0 in (110, 156) else SHOJI)
    for gx in range(x0 + 1, x0 + 26, 6):
        rect(gx, 72, gx + 1, 86, BEAM[1])
    for gy in range(72, 87, 6):
        rect(x0 + 1, gy, x0 + 25, gy + 1, BEAM[1])
# warm glow in the outer windows only
for x0 in (18, 202):
    rect(x0 + 1, 72, x0 + 25, 86, SHOJI_L)
    for gx in range(x0 + 1, x0 + 26, 6):
        rect(gx, 72, gx + 1, 86, BEAM[1])
    for gy in range(72, 87, 6):
        rect(x0 + 1, gy, x0 + 25, gy + 1, BEAM[1])

# ---- fence ----
rect(0, 90, W, 98, FENCE[0])
for x in range(0, W, 6):
    rect(x, 90, x + 4, 98, FENCE[1])
    rect(x, 90, x + 1, 98, FENCE[2] if x % 12 == 0 else FENCE[1])
    rect(x + 4, 90, x + 6, 98, FENCE[0])
    rect(x, 90, x + 4, 91, FENCE[0])  # pointed top hint
rect(0, 93, W, 94, FENCE[0])
rect(0, 96, W, 98, FENCE[0])

# ---- pine tree on the left ----
def clump(cx, cy, rw, rh):
    for y in range(cy - rh, cy + rh + 1):
        for x in range(cx - rw, cx + rw + 1):
            dx, dy = (x - cx) / rw, (y - cy) / rh
            d = dx * dx + dy * dy
            if d <= 1.0:
                shade = 3 if (dy < -0.35 and dx > -0.2) else 2 if dy < 0.3 else 1
                if d > 0.82 and (x + y) % 2: shade = max(0, shade - 1)
                if (x * 7 + y * 3) % 5 == 0 and shade > 1: shade -= 1
                if shade == 3 and (x + y) % 3 == 0: shade = 2
                px(x, y, LEAF[shade])
rect(30, 40, 34, 98, TRUNK[0])
rect(30, 40, 31, 98, TRUNK[1])
rect(32, 62, 40, 64, TRUNK[0])
rect(25, 72, 31, 74, TRUNK[0])
for (cx, cy, rw, rh) in [(32, 24, 11, 5), (24, 34, 14, 5), (41, 36, 13, 5), (31, 46, 16, 6), (46, 58, 11, 5), (19, 62, 12, 5), (30, 70, 14, 5)]:
    clump(cx, cy, rw, rh)

# ---- lanterns (hung from eave) ----
def lantern(cx, top, lit=True):
    rect(cx, RY1 + 2, cx + 1, top, BEAM[0])
    rect(cx - 4, top, cx + 5, top + 2, LAMPD)
    for y in range(top + 2, top + 12):
        w = 5 - (1 if y in (top + 2, top + 11) else 0)
        rect(cx - w, y, cx + w + 1, y + 1, LAMP[1])
        rect(cx - 2, y, cx + 3, y + 1, LAMP[2])
    rect(cx - 1, top + 5, cx + 2, top + 9, LAMP[3])
    for y in (top + 4, top + 8):
        rect(cx - 5, y, cx + 6, y + 1, LAMP[0])
    rect(cx - 4, top + 12, cx + 5, top + 13, LAMPD)
    rect(cx, top + 13, cx + 1, top + 16, LAMP[0])
    # warm light pool on the wall, dithered by distance
    for y in range(top - 8, 98):
        for x in range(cx - 22, cx + 23):
            if not (0 <= x < W): continue
            d = (((x - cx) / 2.0) ** 2 + (y - (top + 7)) ** 2) ** 0.5
            c = tuple(img[y, x])
            if c in (WALL[0], WALL[1], WALLSH, WALL[2]) and d < 24:
                t = (24 - d) / 24.0
                b = BAYER[y % 4, x % 4] / 16
                if b < t * 1.1: img[y, x] = WGLOW[1] if b < t * 0.45 else WGLOW[0]
lantern(74, 68)
lantern(214, 68)

# ---- ground ----
rect(0, 98, W, H, GROUND[1])
# vertical banding with dithered seams
gb = [(98, GROUND[0]), (104, GROUND[1]), (116, GROUND[2]), (130, GROUND[1]), (142, GROUND[0])]
for i, (y0, c) in enumerate(gb):
    y1 = gb[i + 1][0] if i + 1 < len(gb) else H
    rect(0, y0, W, y1, c)
    if i + 1 < len(gb):
        for y in range(y1 - 2, y1):
            for x in range(W):
                if BAYER[y % 4, x % 4] / 16 < (y - (y1 - 2) + 1) / 3.0:
                    img[y, x] = gb[i + 1][1]
# lit patch around the wrestler zone (low contrast)
for y in range(108, 138):
    for x in range(60, 210):
        dx, dy = (x - 135) / 78.0, (y - 124) / 16.0
        if dx * dx + dy * dy < 1 and BAYER[y % 4, x % 4] < 5 and tuple(img[y, x]) == GROUND[2]:
            img[y, x] = GROUND[3]
# warm light spilling from the lanterns onto the floor (outer zones only)
for lx in (64, 226):
    for y in range(99, 132):
        for x in range(lx - 24, lx + 25):
            if not (0 <= x < W) or 84 <= x < 190: continue
            d = (((x - lx) / 1.0) ** 2 + ((y - 99) * 1.3) ** 2) ** 0.5
            if d < 20 and tuple(img[y, x]) in (GROUND[0], GROUND[1], GROUND[2], GROUND[3]):
                t = (20 - d) / 20.0
                b = BAYER[y % 4, x % 4] / 16
                if b < t * 0.9: img[y, x] = GGLOW[1] if b < t * 0.35 else GGLOW[0]
# moonlight sheen on the floor centre, cool
for y in range(104, 124):
    for x in range(100, 200):
        if BAYER[y % 4, x % 4] == 0 and tuple(img[y, x]) == GROUND[2]:
            img[y, x] = GROUND[3]
# pebbles and rake streaks
for _ in range(150):
    x, y = rng.randint(0, W), rng.randint(100, 140)
    c = tuple(img[y, x])
    if c in (GROUND[1], GROUND[2]):
        px(x, y, GROUND[0] if rng.rand() < 0.5 else GROUND[3])
for y in range(106, 140, 5):
    for x in range(rng.randint(0, 6), W, 9):
        if rng.rand() < 0.6:
            L = rng.randint(3, 7)
            for i in range(L):
                if tuple(img[y, min(x + i, W - 1)]) in (GROUND[1], GROUND[2], GROUND[3]):
                    px(x + i, y, GROUND[0])
# a few stepping stones at the edges
for (sx, sy) in [(12, 112), (222, 118), (40, 128), (200, 134)]:
    rect(sx, sy, sx + 8, sy + 4, STONE[1]); rect(sx, sy + 3, sx + 8, sy + 4, STONE[0])
    rect(sx + 1, sy - 1, sx + 7, sy, STONE[1])
# ground line
rect(0, 98, W, 99, DARK)
rect(0, 99, W, 100, GROUND[0])
# bottom 20 darker: dithered vignette
dither(0, 140, W, H, GROUND[0], DARK, 0.0, 0.85)
# top 30: dark calm veil on sky
for y in range(0, 14):
    for x in range(W):
        if BAYER[y % 4, x % 4] / 16 < (14 - y) / 20.0 and tuple(img[y, x]) != SKY[0]:
            img[y, x] = SKY[0] if y < 8 else SKY[1]

# ---- center backdrop calm: pull contrast down in x 80-190, y 39-124 ----
# flatten wall detail into a hushed tone behind pole/wrestler
for y in range(66, 98):
    for x in range(80, 191):
        c = tuple(img[y, x])
        if c in (SHOJI, WALL[2], BEAM[1], BEAM[0], FENCE[2], SHOJI_L):
            img[y, x] = WALL[0] if y < 88 else FENCE[1]
        if c == FENCE[1] and (x % 6) in (0, 5):
            img[y, x] = FENCE[0]
# fence in center: darker, no warm planks
for y in range(90, 98):
    for x in range(80, 191):
        if tuple(img[y, x]) == FENCE[2]:
            img[y, x] = FENCE[1]

im = Image.fromarray(img, 'RGB')
assert im.size == (240, 160)
im.save(os.path.join(OUT, 'yard_night.png'))
im.resize((W * 4, H * 4), Image.NEAREST).save(os.path.join(OUT, 'yard_night_x4.png'))
print('distinct colours:', len(set(map(tuple, img.reshape(-1, 3)))))
