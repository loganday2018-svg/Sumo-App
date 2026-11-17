#!/usr/bin/env python3
"""Keikoba (sumo stable training room) background for the teppo mini-game.

240x160 GBA-style pixel art. Only flat colours and ordered (Bayer) dithering.
Writes keikoba.png and keikoba_x4.png next to this script.
"""
import os
import random
import numpy as np
from PIL import Image

W, H = 240, 160
OUT = os.path.dirname(os.path.abspath(__file__))
rnd = random.Random(7)


def hx(s):
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


P = {
    # wall (dark indigo-tinted wood, low contrast)
    'w0': hx('1a1522'), 'w1': hx('221b2b'), 'w2': hx('2b2236'), 'w3': hx('352a40'),
    'wg': hx('130f1a'),
    # beams (warm dark wood)
    'b0': hx('2a1c1a'), 'b1': hx('3b2822'), 'b2': hx('4d3429'), 'b3': hx('654431'),
    # tatami
    't0': hx('2c3022'), 't1': hx('3a4029'), 't2': hx('4b5232'), 'te': hx('151210'),
    # floor step / edge
    'e0': hx('3a2a22'), 'e1': hx('57402f'), 'e2': hx('7a5a3e'),
    # clay floor
    'c0': hx('3b2d29'), 'c1': hx('4c3a30'), 'c2': hx('5f4a39'), 'c3': hx('755c44'),
    'c4': hx('8c7050'),
    # plaques
    'p0': hx('6b5a45'), 'p1': hx('8c7656'), 'p2': hx('a99068'), 'pi': hx('1c1614'),
    # banner
    'n0': hx('1b1a3a'), 'n1': hx('26285a'), 'n2': hx('36397a'), 'g1': hx('b9892c'),
    'g2': hx('e0b648'),
    # lantern / light
    'l0': hx('8a3a1c'), 'l1': hx('c8612a'), 'l2': hx('f0a040'), 'l3': hx('ffd878'),
    'h1': hx('2e2024'), 'h2': hx('4a2e2a'), 'h3': hx('6a4430'),
    # shrine
    's0': hx('5a3a2a'), 's1': hx('7e5538'), 'sw': hx('c9bfa6'), 'sr': hx('8a2a24'),
    # window light
    'k0': hx('2c2430'), 'k1': hx('d8b070'), 'k2': hx('f6dc9a'),
    # misc
    'k': hx('0c0910'),
}

img = np.zeros((H, W, 3), dtype=np.uint8)
img[:, :] = P['w1']

BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0


def px(x, y, c):
    if 0 <= x < W and 0 <= y < H:
        img[y, x] = P[c] if isinstance(c, str) else c


def rect(x0, y0, x1, y1, c):
    """Inclusive rect."""
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W - 1, x1), min(H - 1, y1)
    img[y0:y1 + 1, x0:x1 + 1] = P[c]


def dith(x0, y0, x1, y1, a, b, t):
    """Fill rect with a, replacing fraction t of pixels with b (ordered)."""
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            px(x, y, b if t > BAYER[y % 4, x % 4] else a)


def ramp(x, y, t, cols):
    """Pick from a colour ramp by t in [0,1] with ordered dither."""
    n = len(cols) - 1
    v = min(max(t, 0), 1) * n
    i = int(v)
    f = v - i
    if i >= n:
        return cols[n]
    return cols[i + 1] if f > BAYER[y % 4, x % 4] else cols[i]


def glow(cx, cy, rx, ry, cols, strength=1.0, base=None):
    """Additive-ish radial glow: overwrite pixels with ramp colour where t > 0."""
    for y in range(max(0, cy - ry), min(H, cy + ry + 1)):
        for x in range(max(0, cx - rx), min(W, cx + rx + 1)):
            d = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
            if d >= 1:
                continue
            t = (1 - d) * strength
            c = ramp(x, y, t, cols)
            if c != cols[0]:
                px(x, y, c)


# ---------------------------------------------------------------- wall
WALL_END = 92
# vertical planks, 10px wide with dark seams
plank_cols = ['w1', 'w2', 'w1', 'w0', 'w2', 'w1', 'w3', 'w1', 'w0', 'w1', 'w2', 'w1']
for i, x0 in enumerate(range(0, W, 10)):
    c = plank_cols[i % len(plank_cols)]
    rect(x0, 0, x0 + 8, WALL_END, c)
    rect(x0 + 9, 0, x0 + 9, WALL_END, 'wg')
    # grain streaks
    for _ in range(6):
        gx = x0 + rnd.randint(1, 7)
        gy = rnd.randint(2, WALL_END - 14)
        gl = rnd.randint(3, 9)
        for k in range(gl):
            px(gx, gy + k, 'w0' if c != 'w0' else 'w2')
    # nail heads
    px(x0 + 4, 40, 'wg')
    px(x0 + 4, 78, 'wg')

# horizontal rail mid-wall (nuki) behind everything
rect(0, 46, W - 1, 47, 'w0')
rect(0, 48, W - 1, 48, 'wg')

# darken top & lower wall (calm text zones)
for y in range(0, 30):
    t = (30 - y) / 30 * 0.75
    for x in range(W):
        if t > BAYER[y % 4, x % 4]:
            px(x, y, 'wg' if img[y, x].tolist() != list(P['wg']) else 'k')
for y in range(0, 6):
    rect(0, y, W - 1, y, 'k')

# ceiling beam
rect(0, 0, W - 1, 9, 'b0')
rect(0, 10, W - 1, 10, 'k')
rect(0, 7, W - 1, 8, 'b1')
rect(0, 9, W - 1, 9, 'b0')
dith(0, 5, W - 1, 6, 'b0', 'b1', 0.4)
rect(0, 0, W - 1, 2, 'k')

# vertical posts (side only, so centre stays calm)
for x0, x1 in ((0, 11), (228, 239)):
    rect(x0, 11, x1, WALL_END, 'b0')
    rect(x0 + 2, 11, x0 + 3, WALL_END, 'b1')
    rect(x0 + 6, 11, x0 + 6, WALL_END, 'b1')
    rect(x0 + 8, 11, x0 + 8, WALL_END, 'k') if False else None
    for y in range(14, WALL_END, 9):
        px(x0 + 4, y, 'b2')
        px(x0 + 5, y + 1, 'b2')
rect(11, 11, 11, WALL_END, 'k')
rect(228, 11, 228, WALL_END, 'k')
# slim middle post, away from pole, tinted to the wall
for x0 in (74,):
    rect(x0, 11, x0 + 3, WALL_END, 'w0')
    rect(x0, 11, x0, WALL_END, 'wg')
    rect(x0 + 3, 11, x0 + 3, WALL_END, 'w3')

# ---------------------------------------------------------------- window
wx0, wy0, wx1, wy1 = 196, 13, 224, 30
rect(wx0 - 2, wy0 - 2, wx1 + 2, wy1 + 2, 'b0')
rect(wx0 - 1, wy0 - 1, wx1 + 1, wy1 + 1, 'k')
for y in range(wy0, wy1 + 1):
    for x in range(wx0, wx1 + 1):
        t = 0.2 + 0.5 * (x - wx0) / (wx1 - wx0)
        px(x, y, ramp(x, y, t, ['k0', 'k1', 'k2']))
# slats
for y in range(wy0 + 2, wy1, 4):
    rect(wx0, y, wx1, y + 1, 'b0')
    px(wx0 + 5, y, 'b1')
rect(wx0 + 13, wy0, wx0 + 14, wy1, 'b0')
# sill
rect(wx0 - 3, wy1 + 3, wx1 + 3, wy1 + 4, 'b1')

# light shafts from the window (dithered, low contrast), going down-left
def shaft(xa, xb, slope, y0, y1, strength):
    for y in range(y0, y1):
        off = int((y - wy1) * slope)
        for x in range(xa - off, xb - off):
            if 0 <= x < W and y < WALL_END:
                t = strength * (1 - (y - y0) / (y1 - y0))
                if t > BAYER[y % 4, x % 4]:
                    cur = tuple(img[y, x])
                    if cur == P['w1']:
                        px(x, y, 'w2')
                    elif cur == P['w2']:
                        px(x, y, 'w3')
                    elif cur == P['w0']:
                        px(x, y, 'w1')
                    elif cur == P['wg']:
                        px(x, y, 'w0')


shaft(199, 209, 0.55, 33, WALL_END, 0.8)
shaft(213, 221, 0.55, 33, WALL_END, 0.7)

# ---------------------------------------------------------------- lanterns
def lantern(cx, top, ln):
    rect(cx, 11, cx, top - 1, 'k')  # cord
    # halo
    glow(cx, top + 8, 18, 18, ['w1', 'h1', 'h2'], 0.55)
    rect(cx - 4, top, cx + 4, top + 1, 'b0')
    for i in range(ln):
        y = top + 2 + i
        hw = 5 if 2 <= i <= ln - 3 else 4
        rect(cx - hw, y, cx + hw, y, 'l1')
        rect(cx - hw, y, cx - hw, y, 'l0')
        rect(cx + hw, y, cx + hw, y, 'l0')
        rect(cx - 1, y, cx + 2, y, 'l2')
    rect(cx - 1, top + 4, cx + 1, top + ln - 3, 'l3')
    for i in (4, ln - 2 + 2):
        rect(cx - 4, top + i, cx + 4, top + i, 'l0')
    rect(cx - 4, top + 2 + ln, cx + 4, top + 3 + ln, 'b0')
    px(cx, top + 4 + ln, 'l1')
    px(cx - 1, top + 5 + ln, 'l0')
    px(cx + 1, top + 5 + ln, 'l0')


lantern(60, 14, 13)
lantern(186, 14, 13)  # high and out of the pole's way (pole starts y=39)

# ---------------------------------------------------------------- kamidana shelf
kx = 16
rect(kx, 62, kx + 40, 63, 's1')
rect(kx, 64, kx + 40, 64, 's0')
rect(kx + 3, 65, kx + 5, 69, 's0')
rect(kx + 35, 65, kx + 37, 69, 's0')
# shrine house
rect(kx + 12, 50, kx + 28, 61, 's0')
rect(kx + 13, 52, kx + 27, 61, 'h1')
rect(kx + 17, 54, kx + 23, 61, 'k')
rect(kx + 18, 55, kx + 22, 61, 'h2')
# roof
for i in range(5):
    rect(kx + 9 + i, 45 + i, kx + 31 - i, 45 + i, 's1' if i % 2 == 0 else 's0')
rect(kx + 9, 50, kx + 31, 50, 'k')
rect(kx + 10, 49, kx + 30, 49, 's0')
# shimenawa rope with paper streamers
rect(kx + 6, 47, kx + 34, 47, 'p1') if False else None
for x in range(kx + 5, kx + 36):
    px(x, 52 + (1 if x % 4 < 2 else 0) * 0, 'p0')
for x in (kx + 8, kx + 17, kx + 25, kx + 32):
    rect(x, 53, x + 1, 56, 'sw')
    px(x, 57, 'sw')
# sake bottles and offering
rect(kx + 2, 56, kx + 4, 61, 's0')
px(kx + 3, 55, 'sw')
rect(kx + 36, 56, kx + 38, 61, 's0')
px(kx + 37, 55, 'sw')
rect(kx + 9, 60, kx + 11, 61, 'sw')
rect(kx + 29, 60, kx + 31, 61, 'sr')

# ---------------------------------------------------------------- banner (left of wrestler)
bx = 56
rect(bx - 2, 36, bx + 22, 37, 'b1')
rect(bx - 2, 38, bx + 22, 38, 'k')
rect(bx, 39, bx + 20, 82, 'n0')
rect(bx + 1, 39, bx + 19, 78, 'n1')
rect(bx + 2, 41, bx + 3, 76, 'n0')
rect(bx + 17, 41, bx + 18, 76, 'n0')
# fringe
for x in range(bx, bx + 21, 2):
    rect(x, 79, x, 82, 'n0')
    px(x + 1, 79, 'n1')
rect(bx, 78, bx + 20, 78, 'n0')
# crest: a simple flower of dots
ccx, ccy = bx + 10, 56
for dx, dy in ((0, -5), (4, -2), (3, 4), (-3, 4), (-4, -2)):
    rect(ccx + dx - 1, ccy + dy - 1, ccx + dx + 1, ccy + dy + 1, 'g1')
rect(ccx - 1, ccy - 1, ccx + 1, ccy + 1, 'g2')
# vertical kanji-ish strokes beneath the crest
for k, x in enumerate((bx + 6, bx + 10, bx + 14)):
    for y in range(66, 76):
        if (y + k) % 4 != 0:
            px(x, y, 'g1')
    rect(x - 1, 66, x + 1, 66, 'g1')
# gold trim top
rect(bx + 1, 40, bx + 19, 40, 'g1')

# ---------------------------------------------------------------- nafuda plaques (right)
def plaque(x, y, w=9, h=18, tilt=0):
    rect(x, y, x + w - 1, y + h - 1, 'p0')
    rect(x + 1, y + 1, x + w - 2, y + h - 2, 'p1')
    rect(x + 1, y + 1, x + w - 2, y + 1, 'p2')
    rect(x + 1, y + h - 1, x + w, y + h - 1, 'k')  # drop shadow
    rect(x + w, y + 2, x + w, y + h - 1, 'k')
    px(x + w // 2, y + 2, 'pi')  # hang hole
    # fake brushed characters
    cx = x + w // 2
    for yy in range(y + 5, y + h - 2, 3):
        rect(cx - 1, yy, cx + 1, yy, 'pi')
        px(cx + (1 if yy % 2 else -1), yy + 1, 'pi')
    for yy in range(y + 4, y + h - 2):
        if (yy + x) % 3 == 0:
            px(cx, yy, 'pi')


plank_x = [200, 211, 222]
for i in range(3):
    plaque(196 + i * 11, 51 + (i % 2) * 2, 9, 15)
    plaque(196 + i * 11, 74 + ((i + 1) % 2) * 2, 9, 14) if False else None
# second row of small plaques, fewer so it stays calm
for i in range(3):
    plaque(196 + i * 11, 69 + (i % 2) * 2, 9, 10)
# rail the plaques hang from
rect(193, 47, 233, 48, 'b0')
px(193, 47, 'b1')

# left lower name board (pushed away from the pole)
for i in range(2):
    plaque(16 + i * 11 + 20, 72, 9, 14) if False else None

# ---------------------------------------------------------------- tatami platform at back
rect(0, 79, W - 1, 80, 'te')
for y in range(81, 87):
    rect(0, y, W - 1, y, 't0')
# tatami mat seams and weave
for y in range(81, 87):
    for x in range(W):
        if (x + y) % 6 == 0:
            px(x, y, 't1')
for x in range(0, W, 48):
    rect(x, 81, x + 1, 86, 't2' if False else 'te')
dith(0, 81, W - 1, 82, 't0', 't1', 0.5)
# cloth border on mats (heri)
rect(0, 81, W - 1, 81, 'n0')
# wooden edge face of raised platform
rect(0, 87, W - 1, 91, 'e0')
rect(0, 87, W - 1, 87, 'e1')
rect(0, 88, W - 1, 88, 'e0')
for x in range(0, W, 24):
    rect(x, 88, x, 91, 'te')
rect(0, 91, W - 1, 91, 'te')
dith(0, 89, W - 1, 90, 'e0', 'te', 0.3)

# ---------------------------------------------------------------- floor line + clay
# step lip
rect(0, 92, W - 1, 92, 'e2')
rect(0, 93, W - 1, 93, 'e1')
rect(0, 94, W - 1, 94, 'e0')
rect(0, 95, W - 1, 95, 'te')
# shadow under the lip fading into clay
for y in range(96, 104):
    t = 1 - (y - 96) / 8
    for x in range(W):
        px(x, y, 'c0' if t > BAYER[y % 4, x % 4] else 'c1')
# base clay
for y in range(104, H):
    for x in range(W):
        px(x, y, 'c1')
# clay lightening toward the centre, darkening to the bottom edge
for y in range(104, H):
    for x in range(W):
        # distance from a soft centre
        dx = (x - 125) / 130
        dy = (y - 118) / 40
        t = 0.55 - 0.55 * (dx * dx + dy * dy)
        if y > 140:
            t -= (y - 140) / 20 * 0.7
        px(x, y, ramp(x, y, t + 0.25, ['c0', 'c1', 'c2']))
# window light pool on the clay (warm, soft), right of centre
for y in range(100, 146):
    for x in range(120, 238):
        # parallelogram slanting like the shafts
        xs = x + int((y - 100) * 0.55)
        u = (xs - 190) / 40.0
        if 0 <= u < 1:
            t = 0.9 * (1 - abs(u * 2 - 1)) * (1 - (y - 100) / 46)
            cur = tuple(img[y, x])
            if t > BAYER[y % 4, x % 4]:
                if cur == P['c1']:
                    px(x, y, 'c2')
                elif cur == P['c2']:
                    px(x, y, 'c3')
                elif cur == P['c0']:
                    px(x, y, 'c1')
# lantern glow reflected on floor edge
glow(60, 100, 40, 8, ['c1', 'c2', 'c3'], 0.35)
# clay speckles, scuffs and sweep marks
for _ in range(210):
    x = rnd.randint(0, W - 1)
    y = rnd.randint(100, 155)
    cur = tuple(img[y, x])
    if cur in (P['c1'], P['c2'], P['c3']):
        px(x, y, 'c3' if cur != P['c3'] else 'c4') if y < 140 else px(x, y, 'c0')
for _ in range(60):
    x = rnd.randint(0, W - 8)
    y = rnd.randint(104, 150)
    for k in range(rnd.randint(3, 7)):
        cur = tuple(img[y, x + k])
        if cur in (P['c1'], P['c2']):
            px(x + k, y, 'c0' if y > 132 else 'c2')
# broom sweep arcs
for r in (150, 168):
    for a in range(0, 90):
        import math
        ang = math.radians(200 + a * 0.35)
        x = int(135 + r * math.cos(ang))
        y = int(260 + r * math.sin(ang)) - 0
        if 0 <= x < W and 104 <= y < 148 and (a % 3):
            if tuple(img[y, x]) in (P['c1'], P['c2'], P['c3']):
                px(x, y, 'c0')

for y in range(106, 140, 5):
    for x in range(0, W):
        if (x // 6 + y) % 3 == 0 and tuple(img[y, x]) in (P['c1'], P['c2']):
            px(x, y, 'c0' if tuple(img[y, x]) == P['c1'] else 'c1')

# bottom 20 px calm: heavy dither to dark
for y in range(140, H):
    t = (y - 140) / 20
    for x in range(W):
        if t * 0.9 > BAYER[y % 4, x % 4]:
            px(x, y, 'c0')
rect(0, H - 2, W - 1, H - 1, 'te')

# ---------------------------------------------------------------- vignette on the sides
for y in range(0, H):
    for x in list(range(0, 14)) + list(range(W - 14, W)):
        d = (14 - x) / 14 if x < 14 else (x - (W - 15)) / 14
        if y > 95 and d * 0.6 > BAYER[y % 4, x % 4]:
            cur = tuple(img[y, x])
            if cur == P['c1']:
                px(x, y, 'c0')
            elif cur == P['c2']:
                px(x, y, 'c1')
            elif cur == P['c3']:
                px(x, y, 'c2')

# ---------------------------------------------------------------- save
out = Image.fromarray(img, 'RGB')
assert out.size == (W, H)
out.save(os.path.join(OUT, 'keikoba.png'))
out.resize((W * 4, H * 4), Image.NEAREST).save(os.path.join(OUT, 'keikoba_x4.png'))
n = len(set(map(tuple, img.reshape(-1, 3))))
print('colours:', n)
assert n <= 48, n
