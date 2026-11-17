"""Stable yard in spring (cherry blossom, bright day), 240x160 GBA-style background."""
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
SKY = [C(x) for x in ('1f3668', '2a4a84', '3a609c', '4c78b4', '6094c8', '7eb0dc', '9cc8ec', 'bcdcf4', 'd6ecfa')]
CLOUD = [C('f2f8fc'), C('c8dcee')]
HILL1, HILL2 = C('8ab4a0'), C('6c9c84')
ROOF = [C('3a4258'), C('48526a'), C('5a6680'), C('72809a')]
WALL = [C('a4acc0'), C('b2bacb')]
WALLSH = C('8a92aa')
BEAM = [C('3e3a52'), C('54506a')]
SHOJI = C('ece4d2')
FENCE = [C('4a4658'), C('625c72'), C('9a7a60')]
PINK = [C('9c4a6a'), C('d4789a'), C('f0a4bc'), C('fcd4e0')]
TRUNK = [C('3a2a34'), C('5a4044')]
LAMP = [C('8a3430'), C('b84c3e'), C('c86a54')]
LAMPD = C('4a2a32')
GROUND = [C('948e94'), C('a8a4a6'), C('bcb8b6'), C('cec8c2'), C('dcd6cc')]
STONE = [C('8a8aa0'), C('a4a4b8')]
DARK = C('1c1a2c')
VIG = C('4c4a62')

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

def in_calm(x, y, m=2):
    """True inside the wrestler/pole zone (plus margin) where no petals may go."""
    return 80 - m <= x < 191 + m and 39 - m <= y <= 124 + m

# ---- sky: bands with dither seams ----
bands = [0, 10, 18, 25, 31, 35, 39, 42, 45]
for i in range(len(SKY) - 1):
    y0 = bands[i]
    y1 = bands[i + 1]
    rect(0, y0, W, y1, SKY[i])
    for y in range(y1 - 3, y1):
        for x in range(W):
            if BAYER[y % 4, x % 4] / 16 < (y - (y1 - 3) + 1) / 4.5:
                img[y, x] = SKY[i + 1]
rect(0, 45, W, 100, SKY[-1])

# soft clouds (puffs of ellipses, lit on top, shaded below)
def cloud(cx, cy, w):
    puffs = [(0, 0, w // 2, 3), (-w // 4, 1, w // 4, 2), (w // 4, 0, w // 3, 3), (w // 2 - 3, 1, w // 5, 2)]
    for (ox, oy, rx, ry) in puffs:
        for y in range(cy + oy - ry, cy + oy + ry + 1):
            for x in range(cx + ox - rx, cx + ox + rx + 1):
                dx, dy = (x - cx - ox) / max(1, rx), (y - cy - oy) / max(1, ry)
                if dx * dx + dy * dy <= 1.0:
                    px(x, y, CLOUD[0] if y < cy + 1 else CLOUD[1])
for (cx, cy, w) in [(36, 34, 30), (150, 33, 34), (208, 36, 26), (100, 36, 20)]:
    cloud(cx, cy, w)

# distant green hills
for x in range(W):
    h = int(41 + 4 * np.sin(x / 19.0) + 3 * np.sin(x / 7.0 + 1))
    rect(x, h, x + 1, 100, HILL1)
for x in range(W):
    h = int(45 + 3 * np.sin(x / 13.0 + 2))
    rect(x, h, x + 1, 100, HILL2)

# ---- building ----
RY0, RY1 = 49, 65  # roof span
for y in range(RY0, RY1):
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
# shoji windows (outer two only), paper lit by daylight
for x0 in (18, 202):
    rect(x0, 71, x0 + 26, 87, BEAM[0])
    rect(x0 + 1, 72, x0 + 25, 86, SHOJI)
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
    rect(x, 90, x + 4, 91, FENCE[0])
rect(0, 93, W, 94, FENCE[0])
rect(0, 96, W, 98, FENCE[0])

# ---- cherry tree in bloom (same trunk and clumps as the pine) ----
def clump(cx, cy, rw, rh):
    for y in range(cy - rh, cy + rh + 1):
        for x in range(cx - rw, cx + rw + 1):
            dx, dy = (x - cx) / rw, (y - cy) / rh
            d = dx * dx + dy * dy
            if d <= 1.0:
                shade = 3 if (dy < -0.2 and dx < 0.3) else 2 if dy < 0.3 else 1
                if d > 0.82 and (x + y) % 2: shade = max(0, shade - 1)
                if (x * 7 + y * 3) % 5 == 0 and shade > 1: shade -= 1
                if (x * 5 + y * 11) % 9 == 0 and shade < 3: shade += 1
                px(x, y, PINK[shade])
rect(30, 40, 34, 98, TRUNK[0])
rect(30, 40, 31, 98, TRUNK[1])
rect(32, 62, 40, 64, TRUNK[0])
rect(25, 72, 31, 74, TRUNK[0])
for (cx, cy, rw, rh) in [(32, 24, 11, 5), (24, 34, 14, 5), (41, 36, 13, 5), (31, 46, 16, 6), (46, 58, 11, 5), (19, 62, 12, 5), (30, 70, 14, 5)]:
    clump(cx, cy, rw, rh)

# ---- lanterns (hung from eave, unlit by day) ----
def lantern(cx, top):
    rect(cx, RY1 + 2, cx + 1, top, BEAM[0])
    rect(cx - 4, top, cx + 5, top + 2, LAMPD)
    for y in range(top + 2, top + 12):
        w = 5 - (1 if y in (top + 2, top + 11) else 0)
        rect(cx - w, y, cx + w + 1, y + 1, LAMP[1])
        rect(cx - 2, y, cx + 3, y + 1, LAMP[2])
    for y in (top + 4, top + 8):
        rect(cx - 5, y, cx + 6, y + 1, LAMP[0])
    rect(cx - 4, top + 12, cx + 5, top + 13, LAMPD)
    rect(cx, top + 13, cx + 1, top + 16, LAMP[0])
lantern(74, 68)
lantern(214, 68)

# ---- ground ----
rect(0, 98, W, H, GROUND[1])
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

# ---- petals: scattered on the floor, drifting in the air (sparse) ----
def petal(x, y, c, wide=True):
    for dx in ((0, 1) if wide else (0,)):
        if not in_calm(x + dx, y):
            px(x + dx, y, c)
# fallen under the tree, a little denser
for _ in range(16):
    petal(rng.randint(8, 66), rng.randint(102, 136), PINK[rng.randint(1, 3)], rng.rand() < 0.5)
# elsewhere on the yard
for _ in range(26):
    x, y = rng.randint(0, W - 2), rng.randint(101, 138)
    petal(x, y, PINK[rng.randint(1, 3)], rng.rand() < 0.5)
# drifting in the air (outside the calm zone, below the text band)
for (x, y) in [(70, 40), (62, 78), (96, 36), (120, 33), (160, 35), (196, 44), (212, 62), (226, 80),
               (232, 52), (52, 90), (66, 56), (14, 96), (200, 90), (234, 104), (74, 100), (88, 33),
               (176, 33), (206, 33), (58, 38), (222, 96)]:
    petal(x, y, PINK[3] if (x + y) % 2 else PINK[2], (x * y) % 3 != 0)

# bottom 20 darker: dithered vignette (white and yellow text sits here)
dither(0, 140, W, 150, GROUND[0], VIG, 0.5, 1.0)
dither(0, 150, W, H, VIG, DARK, 0.0, 0.9)
SKYSET = set(SKY)
# top 30: dark calm veil on sky so text stays legible
for y in range(0, 30):
    for x in range(W):
        if BAYER[y % 4, x % 4] / 16 < (30 - y) / 34.0 and tuple(img[y, x]) in SKYSET:
            img[y, x] = SKY[0] if y < 12 else SKY[1]

# ---- center backdrop calm: pull contrast down in x 80-190, y 39-124 ----
for y in range(66, 98):
    for x in range(80, 191):
        c = tuple(img[y, x])
        if c in (SHOJI, BEAM[1], BEAM[0], FENCE[2]):
            img[y, x] = WALL[0] if y < 88 else FENCE[1]
        if c == FENCE[1] and (x % 6) in (0, 5):
            img[y, x] = FENCE[0]
for y in range(90, 98):
    for x in range(80, 191):
        if tuple(img[y, x]) == FENCE[2]:
            img[y, x] = FENCE[1]

im = Image.fromarray(img, 'RGB')
assert im.size == (240, 160)
im.save(os.path.join(OUT, 'yard_spring.png'))
im.resize((W * 4, H * 4), Image.NEAREST).save(os.path.join(OUT, 'yard_spring_x4.png'))
print('distinct colours:', len(set(map(tuple, img.reshape(-1, 3)))))
