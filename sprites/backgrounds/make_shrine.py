"""Haunted shrine at night: 240x160 GBA-style background for the teppo mini-game.
Pure Pillow + numpy, hard pixels, ordered dithering only. Run: python3 make_shrine.py
"""
import math
import os
import numpy as np
from PIL import Image

W, H = 240, 160
OUT = os.path.dirname(os.path.abspath(__file__))


def hx(s):
    s = s.lstrip("#")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


P = {k: hx(v) for k, v in dict(
    s0="070a1c", s1="0b1030", s2="11183e", s3="181f4a", s4="212a58",
    star1="5a6498",
    m0="e6dfae", m1="c4bd90", m2="a8a07a", h1="232c5c", h2="2e3a72",
    f0="141a3c", f1="1a1c40",
    rf0="0c1028", rf1="1a1c40", w0="0f1230", w1="1a1c40",
    c0="05080f", c1="0b1828", c2="142838", c3="1e3c4a",
    k0="12101c", r0="3a0c16", r1="64141f", r2="8e202c", r3="b43438",
    st0="181c30", st1="262b44", st2="373d58", st3="4c5472",
    g0="7a5a2c", g1="d0a448", g2="f2da88", ha="4a3a3c", hb="6e5538",
    ro0="86764e", ro1="b4a474", p0="cdd2e4", p1="969cba",
    mi0="2a3a72", mi1="364d88", mg="232f5a",
    gr0="0d1022", gr1="141a2e", gr2="1c2240", gr3="28304e", gj="080a18",
    moss="1c3a3c", sp="4a78b0", 
).items()}

img = np.zeros((H, W, 3), np.uint8)
BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0


def put(x, y, c):
    if 0 <= x < W and 0 <= y < H:
        img[y, x] = P[c] if isinstance(c, str) else c


def get(x, y):
    return tuple(int(v) for v in img[y, x])


def rect(x0, y0, x1, y1, c):
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            put(x, y, c)


def dmix(x, y, t, a, b):
    """ordered dither: fraction t of pixels use b, others a"""
    return b if BAYER[y % 4, x % 4] < t else a


def hn(x, y, s=0):
    v = (x * 73856093) ^ (y * 19349663) ^ (s * 83492791)
    v = (v ^ (v >> 13)) * 1274126177
    return ((v ^ (v >> 16)) & 0xFFFF) / 65535.0


# ---------------- sky ----------------
ramp = ["s0", "s1", "s2", "s3", "s4"]
for y in range(0, 93):
    f = (y / 92.0) ** 1.15 * (len(ramp) - 1)
    i = min(int(f), len(ramp) - 2)
    for x in range(W):
        put(x, y, dmix(x, y, f - i, ramp[i], ramp[i + 1]))

# stars (sparse, dim)
for k in range(26):
    sx, sy = int(hn(k, 1) * 236) + 2, int(hn(k, 2) * 50) + 4
    put(sx, sy, "star1")

# ---------------- moon (upper right) ----------------
MX, MY, MR = 206, 45, 13
for y in range(MY - 28, MY + 29):
    for x in range(MX - 28, MX + 29):
        d = math.hypot(x - MX, y - MY)
        if 0 <= x < W and 0 <= y < 93 and MR < d < 27:
            if d < 17:
                t = 0.62
                c = "h2"
            elif d < 21:
                t, c = 0.5, "h1"
            else:
                t, c = 0.22, "h1"
            if BAYER[y % 4, x % 4] < t:
                put(x, y, c)
for y in range(MY - MR, MY + MR + 1):
    for x in range(MX - MR, MX + MR + 1):
        d = math.hypot(x - MX, y - MY)
        if d <= MR:
            d2 = math.hypot(x - (MX - 5), y - (MY - 4))
            c = "m0"
            if d2 > 12:
                c = dmix(x, y, 0.5, "m0", "m1") if d2 < 14 else "m1"
            if d2 > 16:
                c = dmix(x, y, 0.5, "m1", "m2") if d2 < 18 else "m2"
            put(x, y, c)
for cx, cy, r in [(MX - 4, MY - 3, 3), (MX + 4, MY + 3, 2), (MX - 2, MY + 6, 2), (MX + 5, MY - 6, 1)]:
    for y in range(cy - r, cy + r + 1):
        for x in range(cx - r, cx + r + 1):
            if math.hypot(x - cx, y - cy) <= r + 0.3 and math.hypot(x - MX, y - MY) < MR - 1:
                put(x, y, dmix(x, y, 0.5, "m1", "m2"))
# thin dark cloud strips crossing the moon
for (cy, x0, x1) in [(MY + 5, MX - 22, MX + 8), (MY + 9, MX - 6, MX + 24)]:
    for x in range(x0, x1):
        wob = int(round(math.sin(x * 0.45) * 0.8))
        for dy in range(2):
            y = cy + wob + dy
            put(x, y, "s2" if dy == 0 else dmix(x, y, 0.5, "s2", "s1"))

# ---------------- far tree line ----------------
for x in range(W):
    ridge = 74 + int(4 * math.sin(x / 17.0) + 3 * math.sin(x / 7.0 + 1)) + (3 if x % 6 < 2 else 0) - (2 if x % 6 == 3 else 0)
    for y in range(ridge, 93):
        put(x, y, dmix(x, y, min(1.0, (y - ridge) / 14.0), "f0", "f1") if y - ridge > 3 else "f0")
        if y - ridge < 2:
            put(x, y, "f0")

# ---------------- distant shrine hall ----------------
HX = 132
for y in range(72, 93):
    for x in range(HX - 36, HX + 37):
        put(x, y, "w0")
for x in range(HX - 36, HX + 37, 12):
    rect(x, 72, x + 1, 92, "w1")
rect(HX - 6, 78, HX + 6, 92, "s0")          # dark doorway
rect(HX - 7, 76, HX + 7, 77, "w1")
for x in range(HX - 36, HX + 37):
    put(x, 73, "s1"); put(x, 74, dmix(x, 74, 0.5, "w0", "s1"))
for y in range(55, 73):
    t = (y - 55) / 17.0
    hw = int(22 + 28 * t ** 1.4)
    for x in range(HX - hw, HX + hw + 1):
        c = "rf1" if (y - 55) % 3 == 0 and (x + y) % 2 == 0 else "rf0"
        put(x, y, c)
for x in range(HX - 24, HX + 25):
    put(x, 54, "rf1")
for sgn in (-1, 1):
    for k in range(3):
        put(HX + sgn * (50 + k), 71 - k, "rf0")
        put(HX + sgn * (22 + 0), 53, "rf1")
rect(HX - 3, 66, HX + 3, 69, "w1")           # gable plaque shadow

# ---------------- cedar helper ----------------
def cedar(cx, top, base, hw, cols, trunk_w, side):
    c0, c1, c2, c3 = cols
    th = 8
    for y in range(top, base + 1):
        k = (y - top) // th
        dy = (y - top) % th
        env = hw * (((k + 1) * th) / float(base - top)) ** 0.8
        env = min(env, hw)
        w = env * (0.35 + 0.65 * dy / (th - 1)) + 1
        if dy == th - 1:
            w += 1.5
        for x in range(int(cx - w - 1), int(cx + w + 2)):
            jag = (hn(x, y, 5) - 0.5) * 1.6
            u = (x - cx) / max(w, 1)
            if abs(x - cx) > w + jag:
                continue
            c = c1
            if u * side > 0.4 and dy < 5:
                c = dmix(x, y, 0.6, c1, c2)
            if dy >= 6:
                c = dmix(x, y, 0.3 * (dy - 5), c1, c0)
            if abs(u) > 0.8 and dy < 4 and u * side > 0:
                c = c3 if hn(x, y, 9) > 0.4 else c2
            put(x, y, c)
    for y in range(base - 10, base + 12):
        for x in range(cx - trunk_w // 2, cx + trunk_w // 2 + 1):
            c = dmix(x, y, 0.4, c1, c0) if (x - cx) * side < 0 else c1
            if (y + x) % 5 == 0 and c == c1:
                c = c0
            put(x, y, c)


cedar(54, 20, 84, 13, (P["c0"], P["c1"], P["c2"], P["c3"]), 3, 1)

# ---------------- mist band A (behind curb) ----------------
def mist(yc, amp, th, phase, k, x0=0, x1=W, protect=True, dark=False):
    for x in range(x0, x1):
        thick = th * (0.55 + 0.45 * math.sin(x / k + phase * 1.7)) 
        if thick < 1.2:
            continue
        off = int(round(amp * math.sin(x / (k * 1.6) + phase)))
        top = int(yc + off - thick / 2)
        n = int(thick)
        for r in range(n):
            y = top + r
            edge = r == 0 or r == n - 1
            c = "mi0"
            if not edge and n >= 4 and 1 <= r <= n - 2 and not (protect and 160 <= x <= 190):
                c = "mi1"
            if edge:
                if BAYER[y % 4, x % 4] < 0.5:
                    continue
            put(x, y, c)


mist(84, 2, 7, 0.3, 11.0)
mist(77, 2, 5, 2.1, 8.0)
mist(88, 1, 5, 4.0, 14.0)

# ---------------- ground line (stone curb) ----------------
for x in range(W):
    put(x, 92, "st3" if (x // 16) % 2 == 0 else "st2")
    for y in range(93, 98):
        c = "st2" if y < 95 else "st1"
        if y == 97:
            c = "st0"
        put(x, y, c)
    if x % 16 == 0:
        for y in range(92, 98):
            put(x, y, "st0")
    if hn(x, 3, 2) > 0.8:
        put(x, 93, "st3")

# ---------------- paving ----------------
rows = [98, 105, 113, 123, 135, 148, 160]
widths = [14, 18, 24, 32, 44, 60]
for i in range(6):
    y0, y1 = rows[i], rows[i + 1]
    w = widths[i]
    off = (i % 2) * (w // 2)
    sx = -off
    n = 0
    while sx < W:
        base = "gr2" if hn(n, i, 3) > 0.5 else "gr1"
        for y in range(y0, y1):
            for x in range(max(sx, 0), min(sx + w, W)):
                c = base
                if y == y0 or x == sx:
                    c = "gj"
                elif y == y0 + 1:
                    c = "gr3" if base == "gr2" else "gr2"
                else:
                    r = hn(x, y, 11)
                    if r > 0.93:
                        c = "gr3"
                    elif r < 0.07:
                        c = "gr0"
                if y >= y1 - 1 and c != "gj":
                    c = "gr0"
                put(x, y, c)
        # moss tufts in joints, cracks
        if hn(n, i, 7) > 0.7:
            mx = sx + 1 + int(hn(n, i, 8) * (w - 3))
            for d in range(3):
                put(mx + d, y0 + 1 + (d % 2), "moss")
        if hn(n, i, 6) > 0.75:
            cx0 = sx + 3 + int(hn(n, i, 4) * (w - 8))
            cy0 = y0 + 3
            for d in range(4):
                put(cx0 + d, cy0 + d // 2, "gr0")
        sx += w
        n += 1
# gravel border strip under curb
for x in range(W):
    for y in (98, 99):
        pass

# ---------------- lanterns ----------------
def halo(cx, cy, rad):
    for y in range(cy - rad, cy + rad + 1):
        for x in range(cx - rad, cx + rad + 1):
            d = math.hypot((x - cx), (y - cy) * 1.05)
            if d < rad and 0 <= y < H:
                t = (1 - d / rad) * 0.62
                if BAYER[y % 4, x % 4] < t:
                    put(x, y, "hb" if d < rad * 0.45 else "ha")


def toro(cx, base):
    halo(cx, base - 18, 15)
    # ground glow
    for y in range(base + 1, base + 7):
        for x in range(cx - 14, cx + 15):
            d = abs(x - cx) / 14.0 + (y - base) / 6.0
            if d < 1.0 and BAYER[y % 4, x % 4] < (1 - d) * 0.5:
                put(x, y, "ha")
    rect(cx - 6, base - 2, cx + 6, base, "st1"); rect(cx - 6, base - 2, cx + 6, base - 2, "st2")
    rect(cx - 2, base - 12, cx + 2, base - 3, "st2")
    rect(cx - 2, base - 12, cx - 2, base - 3, "st1"); rect(cx + 2, base - 12, cx + 2, base - 3, "st3")
    rect(cx - 5, base - 14, cx + 5, base - 13, "st3"); rect(cx - 5, base - 13, cx + 5, base - 13, "st2")
    rect(cx - 4, base - 22, cx + 4, base - 15, "st1")
    rect(cx - 4, base - 22, cx - 4, base - 15, "st0")
    rect(cx - 2, base - 21, cx + 2, base - 16, "g0")
    rect(cx - 1, base - 20, cx + 1, base - 17, "g1")
    put(cx, base - 19, "g2"); put(cx, base - 18, "g2")
    rect(cx - 6, base - 24, cx + 6, base - 23, "st3"); rect(cx - 6, base - 23, cx + 6, base - 23, "st2")
    for j, hw in enumerate([8, 7, 6, 5, 3, 2]):
        y = base - 25 - j
        for x in range(cx - hw, cx + hw + 1):
            put(x, y, "st3" if (x > cx and j < 4) else "st2")
        put(cx - hw, y, "st1")
    put(cx - 8, base - 25, "st3"); put(cx + 8, base - 25, "st3")
    rect(cx - 1, base - 33, cx + 1, base - 31, "st3")
    put(cx, base - 34, "st3")


toro(63, 95)
toro(199, 95)

# ---------------- torii (left) ----------------
def pillar(x0, x1, ytop, ybot):
    for y in range(ytop, ybot + 1):
        for x in range(x0, x1 + 1):
            u = x - x0
            c = ["r0", "r1", "r2", "r2", "r3", "r2"][min(u, 5)] if x1 - x0 == 5 else "r2"
            if u == x1 - x0 - 1 and y % 7 != 0:
                c = "r3"
            if y > ybot - 6:
                c = "k0" if u > 0 else "k0"
            put(x, y, c)
    for x in range(x0 - 1, x1 + 2):
        put(x, ybot + 1, "st0")
        put(x, ybot, "k0")


pillar(7, 12, 40, 93)
pillar(35, 40, 40, 93)
# nuki
for y in range(51, 55):
    for x in range(5, 43):
        put(x, y, {51: "r3", 52: "r2", 53: "r1", 54: "r0"}[y])
# shimaki and kasagi
for x in range(3, 45):
    put(x, 38, "r2"); put(x, 39, "r1"); put(x, 40, "r0")
for y in (34, 35):
    for x in range(0, 48):
        put(x, y, "k0")
for x in range(0, 48):
    put(x, 36, "r2" if x % 9 else "r1"); put(x, 37, "r1")
for x in (0, 1, 2, 45, 46, 47):
    put(x, 33, "k0")
put(0, 32, "k0"); put(47, 32, "k0")
rect(21, 41, 26, 49, "k0"); rect(22, 43, 25, 47, "r0"); put(23, 45, "g0"); put(24, 45, "g0"); put(23, 44, "g0"); put(24, 46, "g0")
# shimenawa rope with shide
for x in range(13, 35):
    sag = int(round(3 * math.sin((x - 13) / 21.0 * math.pi)))
    y = 58 + sag
    for dy in range(3):
        put(x, y + dy, "ro1" if (x + dy) % 3 == 0 else "ro0")
    put(x, y + 3, "k0") if x % 2 else None
for x, w in [(16, 0), (20, 0), (25, 0), (30, 0)]:
    sag = int(round(3 * math.sin((x - 13) / 21.0 * math.pi)))
    y0 = 62 + sag
    for r in range(8):
        zx = x + (1 if (r // 2) % 2 else 0)
        put(zx, y0 + r, "p0"); put(zx + 1, y0 + r, "p1" if r % 2 else "p0")
for y in (58, 59):
    put(12, y, "ro1"); put(35, y, "ro1")

# ---------------- foreground cedar (right edge) ----------------
cedar(233, 12, 86, 24, (P["c0"], P["c1"], P["c2"], P["c3"]), 6, -1)

# ---------------- mist band B (over ground) ----------------
for yc, ph, kk in [(104, 1.0, 13.0), (118, 3.3, 17.0)]:
    for x in range(W):
        thick = 6 * (0.5 + 0.5 * math.sin(x / kk + ph))
        if thick < 1.5:
            continue
        top = int(yc + 2 * math.sin(x / (kk * 1.7) + ph))
        for r in range(int(thick)):
            y = top + r
            if BAYER[y % 4, x % 4] < (0.5 if 0 < r < int(thick) - 1 else 0.2):
                if not (160 <= x <= 190):
                    put(x, y, "mg")
                elif BAYER[y % 4, x % 4] < 0.2:
                    put(x, y, "mg")

# ---------------- hitodama (dim spirit flames, away from the pole) ----------------
def hito(cx, cy):
    for dy, hwd in enumerate([0, 1, 1, 2, 1]):
        for x in range(cx - hwd, cx + hwd + 1):
            put(x, cy + dy, "sp")
    put(cx, cy + 2, "sp")
    put(cx, cy - 1, "mi1")
    put(cx + 1, cy - 2, "mi0")


hito(86, 40)
hito(222, 72)
hito(24, 74)
hito(100, 88)
hito(150, 30)

# ---------------- vignette: keep top and bottom calm ----------------
for y in range(H):
    if y >= 138:
        t = min(0.85, (y - 138) / 22.0)
        for x in range(W):
            if BAYER[y % 4, x % 4] < t:
                put(x, y, "gr0")
    if y < 26:
        t = (26 - y) / 26.0 * 0.5
        for x in range(W):
            if BAYER[y % 4, x % 4] < t and get(x, y) not in (P["m0"], P["m1"], P["m2"]):
                put(x, y, "s0")

# ---------------- write ----------------
im = Image.fromarray(img, "RGB")
assert im.size == (W, H)
im.save(os.path.join(OUT, "shrine.png"))
im.resize((W * 4, H * 4), Image.NEAREST).save(os.path.join(OUT, "shrine_x4.png"))
n = len({tuple(c) for c in img.reshape(-1, 3)})
print("distinct colours:", n)
