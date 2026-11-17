"""Stable yard on a snowy winter afternoon, 240x160 GBA-style background.

Same composition as make_yard.py; only palette, sky, lighting and snow differ.
"""
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
SKY = [C(x) for x in ('2a3446', '34425a', '40506a', '4e607a', '5e7088', '72849a', '8a9cb0', 'a0b0c0', 'b4c2d0')]
HILL1, HILL2 = C('8294ac'), C('7084a0')
ROOF = [C('1c2230'), C('262e40'), C('34405a'), C('46526c')]
WALL = [C('4e5870'), C('5c6882'), C('6e7a96')]
WALLSH = C('3e4860')
BEAM = [C('2a2838'), C('3c3a4c')]
SHOJI = WALL[2]
SHOJI_L = C('d6c08e')
FENCE = [C('2a2634'), C('3a3644'), C('4c4858')]
LEAF = [C('1a2c2e'), C('24403a'), C('34583f'), C('4a7048')]
TRUNK = [C('2a2024'), C('40303a')]
LAMP = [C('b8603c'), C('e0985a'), C('f4c880'), C('fff0c8')]
LAMPD = BEAM[0]
STONE = [C('3c4660'), C('586482')]
DARK = C('10141e')
# snow: bright, light shade, mid (dimmed for text areas), mid shade
SN, SN2, SNM, SNM2 = C('eef3f8'), C('cdd8e4'), C('a2b2c4'), C('8a9bb0')
# earth showing through, muted and cool (not brown)
E = [C('4a4652'), C('5e5864'), C('706a74')]

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

# ---- sky: overcast bands with dither seams ----
bands = [0, 8, 15, 21, 27, 32, 37, 41, 45]
for i in range(len(SKY) - 1):
    y0 = bands[i]
    y1 = bands[i + 1]
    rect(0, y0, W, y1, SKY[i])
    for y in range(y1 - 3, y1):
        for x in range(W):
            if BAYER[y % 4, x % 4] / 16 < (y - (y1 - 3) + 1) / 4.5:
                img[y, x] = SKY[i + 1]
rect(0, 45, W, 100, SKY[-1])
# soft cloud streaks, low contrast, kept below the text rows
for (cx, cy, l) in [(30, 33, 26), (150, 32, 30), (205, 37, 22), (95, 38, 18)]:
    for i in range(l):
        px(cx + i, cy, SKY[6])
        if 3 < i < l - 3:
            px(cx + i, cy + 1, SKY[7])
# distant snowy hills
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

# snow on the roof: ridge, drifts along the top rows, patches, eave tufts
rect(0, RY0 - 3, W, RY0 - 1, SN)
rect(0, RY0 - 1, W, RY0, SN2)
for x in range(0, W, 8):
    rect(x, RY0 - 5, x + 5, RY0 - 4, SN)
    rect(x, RY0 - 4, x + 5, RY0 - 3, SN2)
for x in range(W):
    d = max(1, 2 + int(round(1.5 * np.sin(x / 5.0) + 1.2 * np.sin(x / 2.3 + 1))))
    for r in range(d):
        px(x, RY0 + r, SN if r == 0 else SN2)
for _ in range(16):
    k = rng.randint(1, 4)
    y = RY0 + 4 * k + rng.randint(0, 2)
    x = rng.randint(0, W - 10)
    L = rng.randint(3, 9)
    rect(x + 1, y, x + L - 1, y + 1, SN2)
    rect(x, y + 1, x + L, y + 2, SN2)
x = 0
while x < W:
    L = rng.randint(3, 8)
    if rng.rand() < 0.6:
        rect(x, RY1 - 2, x + L, RY1, SN2)
        rect(x + 1, RY1 - 3, x + L - 1, RY1 - 2, SN2)
    x += L + rng.randint(2, 9)

# eave shadow onto wall
dither(0, RY1 + 3, W, RY1 + 9, WALLSH, WALL[0], 0.0, 1.0)
# icicles under the eave
for x in range(3, W, 11):
    if rng.rand() < 0.7:
        for j in range(rng.randint(1, 4)):
            px(x, RY1 + 3 + j, SN2)
# wall
rect(0, RY1 + 9, W, 90, WALL[0])
for y in range(RY1 + 9, 90):
    for x in range(W):
        if BAYER[y % 4, x % 4] < 2 and (y > 72):
            img[y, x] = WALL[1]
# base stone with a ledge of snow
rect(0, 88, W, 92, STONE[0])
for x in range(0, W, 12):
    rect(x, 88, x + 1, 92, BEAM[0])
rect(0, 88, W, 89, STONE[1])
# posts
for x in range(8, W, 46):
    rect(x, RY1 + 3, x + 5, 88, BEAM[1])
    rect(x, RY1 + 3, x + 1, 88, BEAM[0])
    rect(x + 4, RY1 + 3, x + 5, 88, BEAM[0])
# shoji windows between posts
for x0 in (18, 202):
    rect(x0, 71, x0 + 26, 87, BEAM[0])
    rect(x0 + 1, 72, x0 + 25, 86, SHOJI_L)
    for gx in range(x0 + 1, x0 + 26, 6):
        rect(gx, 72, gx + 1, 86, BEAM[1])
    for gy in range(72, 87, 6):
        rect(x0 + 1, gy, x0 + 25, gy + 1, BEAM[1])
    # snow on the sill
    rect(x0, 87, x0 + 26, 88, SN2)

# ---- fence ----
rect(0, 90, W, 98, FENCE[0])
for x in range(0, W, 6):
    rect(x, 90, x + 4, 98, FENCE[1])
    rect(x, 90, x + 1, 98, FENCE[2] if x % 12 == 0 else FENCE[1])
    rect(x + 4, 90, x + 6, 98, FENCE[0])
    rect(x, 90, x + 4, 91, FENCE[0])
rect(0, 93, W, 94, FENCE[0])
rect(0, 96, W, 98, FENCE[0])
# snow caps on each picket
for x in range(0, W, 6):
    rect(x, 89, x + 4, 90, SN)
    rect(x, 90, x + 4, 91, SN2)
    rect(x + 1, 88, x + 3, 89, SN)

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
# snow resting on the branch pads
leafset = set(LEAF)
snap = img.copy()
for x in range(8, 62):
    for y in range(16, 84):
        if tuple(snap[y, x]) in leafset and tuple(snap[y - 1, x]) not in leafset:
            hi, lo = (SNM, SNM2) if y < 32 else (SN, SN2)
            if (x * 5 + y) % 7 != 0:
                px(x, y, hi)
            if tuple(snap[y + 1, x]) in leafset and (x + y) % 3 != 0:
                px(x, y + 1, lo)
            if y >= 32 and (x % 4) < 2 and tuple(snap[y + 2, x]) in leafset and (x * 3 + y) % 5 < 2:
                px(x, y + 2, lo)

# ---- lanterns (hung from eave), softly lit ----
def lantern(cx, top):
    rect(cx, RY1 + 2, cx + 1, top, BEAM[0])
    rect(cx - 4, top, cx + 5, top + 2, LAMPD)
    rect(cx - 4, top - 1, cx + 5, top, SN)  # snow cap
    rect(cx - 2, top - 2, cx + 3, top - 1, SN2)
    for y in range(top + 2, top + 12):
        w = 5 - (1 if y in (top + 2, top + 11) else 0)
        rect(cx - w, y, cx + w + 1, y + 1, LAMP[1])
        rect(cx - 2, y, cx + 3, y + 1, LAMP[2])
    rect(cx - 1, top + 5, cx + 2, top + 9, LAMP[3])
    for y in (top + 4, top + 8):
        rect(cx - 5, y, cx + 6, y + 1, LAMP[0])
    rect(cx - 4, top + 12, cx + 5, top + 13, LAMPD)
    rect(cx, top + 13, cx + 1, top + 16, LAMP[0])
    # soft halo via sparse dither on the wall
    for y in range(top - 3, top + 20):
        for x in range(cx - 13, cx + 14):
            d = abs(x - cx) + abs(y - (top + 7))
            if 8 < d < 14 and BAYER[y % 4, x % 4] < (3 if d < 12 else 1) and 0 <= y < H \
                    and tuple(img[y, x]) in (WALL[0], WALL[1], WALLSH):
                img[y, x] = WALL[2]
lantern(74, 68)
lantern(214, 68)

# ---- ground: thin snow, earth showing through near the centre ----
rect(0, 98, W, H, SN)
# far strip in the fence shadow, then ramp to bright snow
dither(0, 100, W, 108, SNM2, SN2, 0.0, 1.0)
dither(0, 108, W, 114, SN2, SN, 0.0, 1.0)
# snow drifts: faint shade streaks
for y in range(114, 134, 4):
    for x in range(rng.randint(0, 8), W, 13):
        if rng.rand() < 0.6:
            L = rng.randint(4, 9)
            for i in range(L):
                if tuple(img[y, min(x + i, W - 1)]) == SN:
                    px(x + i, y, SN2)
# earth patches: noisy ellipse centred on the training spot
for y in range(104, 138):
    for x in range(56, 214):
        dx, dy = (x - 135) / 74.0, (y - 122) / 14.0
        f = 1.0 - (dx * dx + dy * dy)
        n = (0.5 * np.sin(x / 5.3 + 1.7 * np.sin(y / 3.1)) + 0.3 * np.sin(x / 2.7 + y / 2.1)
             + 0.25 * np.sin(x / 11.0 - y / 4.0))
        v = f + 0.5 * n + 0.18 * np.sin(x / 1.9) * np.sin(y / 1.7)
        if v > 0.62:
            c = E[1]
            if (x * 3 + y * 5) % 11 == 0: c = E[0]
            elif (x + y * 2) % 13 == 0: c = E[2]
            px(x, y, c)
        elif v > 0.46:
            px(x, y, E[1] if BAYER[y % 4, x % 4] < 9 else E[0])
        elif v > 0.36:
            px(x, y, SN2 if BAYER[y % 4, x % 4] < 8 else E[1])
        elif v > 0.26 and BAYER[y % 4, x % 4] < 6:
            px(x, y, SN2)
# rake streaks across the bare earth
for y in range(112, 134, 4):
    for x in range(rng.randint(0, 6), W, 9):
        if rng.rand() < 0.6:
            L = rng.randint(3, 7)
            for i in range(L):
                if tuple(img[y, min(x + i, W - 1)]) == E[1]:
                    px(x + i, y, E[0])
# a few bare specks near the edges of the yard
for _ in range(14):
    x, y = rng.randint(0, W), rng.randint(106, 134)
    if tuple(img[y, x]) in (SN, SN2) and not (60 < x < 210):
        px(x, y, E[2])
# stepping stones with snow caps
for (sx, sy) in [(12, 112), (222, 118), (40, 128), (200, 134)]:
    rect(sx, sy, sx + 8, sy + 4, STONE[1]); rect(sx, sy + 3, sx + 8, sy + 4, STONE[0])
    rect(sx + 1, sy - 1, sx + 7, sy, STONE[1])
    rect(sx + 1, sy - 1, sx + 7, sy + 1, SN)
    rect(sx + 2, sy + 1, sx + 6, sy + 2, SN2)
# ground line
rect(0, 98, W, 99, DARK)
rect(0, 99, W, 100, SNM2)
# bottom 20 rows: snow falls into deep shade, then dithered vignette
dither(0, 128, W, 136, SN, SNM2, 0.0, 1.0)
dither(0, 136, W, 140, SNM2, SKY[2], 0.0, 1.0)
dither(0, 140, W, H, SKY[2], DARK, 0.0, 0.85)
# keep stones in the bottom zone readable but dim
# top 30: dark calm veil on sky
for y in range(0, 14):
    for x in range(W):
        if BAYER[y % 4, x % 4] / 16 < (14 - y) / 20.0 and tuple(img[y, x]) != SKY[0]:
            img[y, x] = SKY[0] if y < 8 else SKY[1]

# ---- center backdrop calm: pull contrast down in x 80-190, y 39-124 ----
for y in range(66, 98):
    for x in range(80, 191):
        c = tuple(img[y, x])
        if c in (SHOJI, WALL[2], BEAM[1], BEAM[0], FENCE[2], SHOJI_L):
            img[y, x] = WALL[0] if y < 88 else FENCE[1]
        if c == FENCE[1] and (x % 6) in (0, 5):
            img[y, x] = FENCE[0]
for y in range(90, 98):
    for x in range(80, 191):
        if tuple(img[y, x]) == FENCE[2]:
            img[y, x] = FENCE[1]

# dim bright snow behind the pole and wrestler (soft dithered edges)
DIM = {SN: SNM, SN2: SNM2}
for y in range(36, 134):
    wy = 1.0 if y <= 124 else max(0.0, (132 - y) / 8.0)
    for x in range(56, 212):
        wx = min(1.0, (x - 56) / 24.0, (210 - x) / 20.0)
        w = min(wx, wy)
        if w <= 0:
            continue
        c = tuple(img[y, x])
        if c in DIM and (BAYER[y % 4, x % 4] + 0.5) / 16 < w:
            img[y, x] = DIM[c]

# ---- falling snowflakes: sparse, outside the calm zone and text rows ----
placed = 0
for _ in range(400):
    if placed >= 26:
        break
    x, y = rng.randint(0, W), rng.randint(31, 97)
    if 74 < x < 196:
        continue
    bg = img[y, x].astype(int)
    if bg.sum() > 330:
        continue
    px(x, y, SN)
    placed += 1

im = Image.fromarray(img, 'RGB')
assert im.size == (240, 160)
im.save(os.path.join(OUT, 'yard_winter.png'))
im.resize((W * 4, H * 4), Image.NEAREST).save(os.path.join(OUT, 'yard_winter_x4.png'))
print('distinct colours:', len(set(map(tuple, img.reshape(-1, 3)))))
