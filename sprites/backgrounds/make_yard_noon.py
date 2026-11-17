"""Stable yard at bright midday, 240x160 GBA-style background for the teppo mini-game."""
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
SKY = [C(x) for x in ('1f5aa8', '2a6cbc', '3a82cc', '54a0dc', '74b8e8', '98cef0', 'b8e0f6', 'd0ecf8', 'e2f4fb')]
CLOUD = [C('ffffff'), C('d8eaf8')]
HILL1, HILL2 = C('8db4c0'), C('76a0ae')
ROOF = [C('3a404c'), C('4a5260'), C('5e6876'), C('7a8494')]
WALL = [C('cfc8b8'), C('dcd5c4'), C('eae4d4')]
WALLSH = C('a8a498')
BEAM = [C('5a4a40'), C('7a6858')]
SHOJI = C('f0ecdf')
SHOJI_L = SHOJI
FENCE = [C('6a665e'), C('847e72'), C('9a9484')]
LEAF = [C('1f5a2a'), C('2d7a35'), C('43963f'), C('62b04c')]
TRUNK = [C('4a3426'), C('6a4c36')]
LAMP = [C('7c3a34'), C('b05448'), C('c26a58'), C('cf7f6a')]
LAMPD = C('4a3030')
GROUND = [C('b09c78'), C('c4b088'), C('d4c29a'), C('e0d0ac'), C('ecdfc0')]
STONE = [C('a09c98'), C('bcb8b0')]
DARK = C('2c2a3a')

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
# small white clouds, puffy tops and a pale shaded underside
def cloud(cx, cy, w):
    rect(cx + w // 4, cy - 2, cx + w // 4 + w // 2, cy - 1, CLOUD[0])
    rect(cx + 1, cy - 1, cx + w - 1, cy, CLOUD[0])
    rect(cx, cy, cx + w, cy + 1, CLOUD[0])
    rect(cx + 2, cy + 1, cx + w - 2, cy + 2, CLOUD[1])
cloud(58, 33, 24)
cloud(176, 30, 28)
cloud(116, 35, 14)
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
# short eave shadow onto wall (sun overhead)
dither(0, RY1 + 3, W, RY1 + 6, WALLSH, WALL[0], 0.0, 1.0)
# wall
rect(0, RY1 + 6, W, 90, WALL[0])
for y in range(RY1 + 6, 90):
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
# shoji windows between posts, plain daylight paper
for x0 in (18, 202):
    x0 += 0
    rect(x0, 71, x0 + 26, 87, BEAM[0])
    rect(x0 + 1, 72, x0 + 25, 86, WALL[2] if x0 in (110, 156) else SHOJI)
    for gx in range(x0 + 1, x0 + 26, 6):
        rect(gx, 72, gx + 1, 86, BEAM[1])
    for gy in range(72, 87, 6):
        rect(x0 + 1, gy, x0 + 25, gy + 1, BEAM[1])
# redraw grid (no glow at midday)
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
                shade = 3 if (dy < -0.2 and dx < 0.3) else 2 if dy < 0.3 else 1
                if d > 0.82 and (x + y) % 2: shade = max(0, shade - 1)
                if (x * 7 + y * 3) % 5 == 0 and shade > 1: shade -= 1
                px(x, y, LEAF[shade])
rect(30, 40, 34, 98, TRUNK[0])
rect(30, 40, 31, 98, TRUNK[1])
rect(32, 62, 40, 64, TRUNK[0])
rect(25, 72, 31, 74, TRUNK[0])
for (cx, cy, rw, rh) in [(32, 24, 11, 5), (24, 34, 14, 5), (41, 36, 13, 5), (31, 46, 16, 6), (46, 58, 11, 5), (19, 62, 12, 5), (30, 70, 14, 5)]:
    clump(cx, cy, rw, rh)

# ---- lanterns (hung from eave, unlit) ----
def lantern(cx, top, lit=False):
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
    rect(sx + 1, sy + 4, sx + 9, sy + 5, GROUND[0])
# ground line
rect(0, 98, W, 99, DARK)
rect(0, 99, W, 100, GROUND[0])
# bottom 20 darker: dithered vignette
dither(0, 140, W, H, GROUND[0], DARK, 0.45, 1.0)
# top 30: calm veil on sky (dithered down into the deep blues) so white text reads
for y in range(0, 30):
    for x in range(W):
        if BAYER[y % 4, x % 4] / 16 < min(1.0, (30 - y) / 20.0):
            c = tuple(img[y, x])
            if c in SKY[2:] or c in CLOUD or c in (HILL1, HILL2):
                img[y, x] = SKY[0] if y < 12 else SKY[1]
            elif c == SKY[1] and y < 12:
                img[y, x] = SKY[0]

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
im.save(os.path.join(OUT, 'yard_noon.png'))
im.resize((W * 4, H * 4), Image.NEAREST).save(os.path.join(OUT, 'yard_noon_x4.png'))
print('distinct colours:', len(set(map(tuple, img.reshape(-1, 3)))))
