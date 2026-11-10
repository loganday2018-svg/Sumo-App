"""Classic rikishi sprite set in the style of the Kaiju Yokozuna concept.

Every design is a parameter dict: body shape, palette, hair, face and gear.
Run:  python sprites/make_rikishi_set.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 56, 62
OX = 4                      # design coords are on a 48-wide grid, centre 23.5
OY = 0                      # vertical shift applied by px (used by the walk cycle)
CX = 23.5 + OX
OUT = Path(__file__).parent / "concepts" / "rikishi_set"
LIGHT = np.array([-0.45, -0.7, 0.55])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
YY, XX = np.mgrid[0:H, 0:W].astype(float)


def hexc(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)


class Sprite:
    def __init__(self, outline):
        self.part = np.full((H, W), -1, dtype=int)
        self.z = np.full((H, W), -1.0)
        self.col = np.zeros((H, W, 4), dtype=np.uint8)
        self.ramps, self.hard, self.group = {}, {}, {}
        self.outline = hexc(outline)
        self.n = 0

    def add(self, mask, normals, ramp, z, hard=True, group=None):
        pid = self.n
        self.n += 1
        self.ramps[pid] = [hexc(c) for c in ramp]
        self.hard[pid] = hard
        self.group[pid] = group or str(pid)
        lv = np.digitize(np.clip(normals @ LIGHT, 0, 1), [0.18, 0.45, 0.78])
        for y, x in zip(*np.nonzero(mask)):
            if z >= self.z[y, x]:
                self.z[y, x] = z
                self.part[y, x] = pid
                self.col[y, x] = self.ramps[pid][lv[y, x]]

    def group_at(self, x, y):
        p = self.part[y, x]
        return None if p < 0 else self.group[p]

    def finish(self):
        img = self.col.copy()
        for y in range(H):
            for x in range(W):
                p = self.part[y, x]
                if p < 0:
                    continue
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W:
                        q = self.part[yy, xx]
                        if q >= 0 and q != p and self.z[yy, xx] < self.z[y, x] \
                                and self.group[q] != self.group[p]:
                            img[y, x] = self.outline if self.hard[p] else self.ramps[p][0]
                            break
        filled = self.part >= 0
        out = img.copy()
        for y in range(H):
            for x in range(W):
                if not filled[y, x]:
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < H and 0 <= xx < W and filled[yy, xx]:
                            out[y, x] = self.outline
                            break
        return out


def ellipse(cx, cy, rx, ry, rot=0.0):
    c, s = math.cos(rot), math.sin(rot)
    dx, dy = XX + 0.5 - cx, YY + 0.5 - cy
    u = (dx * c + dy * s) / rx
    v = (-dx * s + dy * c) / ry
    d = u * u + v * v
    nz = np.sqrt(np.clip(1 - d, 0, 1))
    n = np.stack([u * c - v * s, u * s + v * c, nz * 1.1], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return d <= 1.0, n


def cyl(mask, rx):
    u = np.clip((XX + 0.5 - CX) / rx, -1, 1)
    n = np.stack([u, np.full_like(u, -0.15), np.sqrt(1 - u * u) + 0.2], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return mask, n


def flat(mask):
    n = np.zeros((H, W, 3)); n[..., 2] = 1.0; n[..., 0] = -0.1
    return mask, n / np.linalg.norm(n, axis=-1, keepdims=True)


def px(img, pts, color):
    c = hexc(color) if isinstance(color, str) else tuple(color)
    for x, y in pts:
        x = int(x) + OX
        y = int(y) + OY
        if 0 <= x < W and 0 <= y < H:
            img[y, x] = c


def mirror(pts):
    return pts + [(47 - x, y) for x, y in pts]


BODIES = {
    "classic": dict(trx=16.5, try_=15.5, tcy=33, arm=16, arx=5.5, ary=10.5, hand=18, hr=4.3, leg=8, lrx=6.2),
    "heavy":   dict(trx=19.0, try_=16.5, tcy=33, arm=18.5, arx=5.8, ary=10.5, hand=20.5, hr=4.5, leg=9, lrx=7.0),
    "lean":    dict(trx=14.0, try_=14.0, tcy=33, arm=14.8, arx=6.2, ary=11.0, hand=16.8, hr=4.7, leg=7.5, lrx=6.0),
    "small":   dict(trx=15.0, try_=13.5, tcy=35, arm=14.6, arx=5.0, ary=9.0, hand=16.2, hr=3.9, leg=7, lrx=5.8),
}


def build(d):
    b = BODIES[d.get("body", "classic")]
    s = Sprite(d["outline"])
    sk, mw = d["skin"], d["mawashi"]
    for side in (-1, 1):
        m, n = ellipse(CX + side * b["leg"], 52, b["lrx"], 7.5)
        s.add(m, n, sk, 1, group=f"leg{side}")
    m, n = ellipse(CX, b["tcy"], b["trx"], b["try_"])
    s.add(m, n, sk, 2, group="torso")
    top = b["tcy"] + 6
    band = (YY >= top) & (YY <= top + 5) & ellipse(CX, b["tcy"], b["trx"] + 1, b["try_"] + 1)[0]
    s.add(*cyl(band, b["trx"] + 1), mw, 3, hard=False, group="mawashi")
    front = (np.abs(XX + 0.5 - CX) <= 3.5) & (YY >= top + 5) & (YY <= 51)
    s.add(*cyl(front, 5), mw, 3, hard=False, group="mawashi")
    apron = d.get("apron")
    if apron:
        am = (np.abs(XX + 0.5 - CX) <= 6.5) & (YY >= top + 6) & (YY <= 58)
        s.add(*flat(am), APRONS[apron]["ramp"], 3.5, hard=True, group="apron")
    for side in (-1, 1):
        m, n = ellipse(CX + side * b["arm"], b["tcy"] - 2, b["arx"], b["ary"], rot=side * -0.22)
        s.add(m, n, sk, 4, group=f"arm{side}")
        m, n = ellipse(CX + side * b["hand"], b["tcy"] + 8, b["hr"], b["hr"] - 0.3)
        s.add(m, n, sk, 5, group=f"arm{side}")
    m, n = ellipse(CX, 14, 9.0, 8.5)
    s.add(m, n, sk, 6, group="head")
    HAIR[d.get("hairstyle", "chonmage")](s, d)
    return s, b


# ---------------------------------------------------------------- hair
def _cap(s, d, low=10):
    m, n = ellipse(CX, 14, 9.1, 8.6)
    side = m & (np.abs(XX + 0.5 - CX) >= 7.6) & (YY <= 15)
    s.add((m & (YY <= low)) | side, n, d["hair"], 7, hard=False, group="head")


def hair_chonmage(s, d):
    _cap(s, d)
    m, n = ellipse(CX, 4, 3.6, 2.6)
    s.add(m, n, d["hair"], 8, group="knot")
    m, n = ellipse(CX + 3.5, 3.5, 3.0, 1.8, rot=-0.3)
    s.add(m, n, d["hair"], 8, group="knot")


def hair_oicho(s, d):
    """Ginkgo-leaf topknot worn by top-division wrestlers."""
    _cap(s, d)
    m, n = ellipse(CX, 5.0, 1.8, 2.0)
    s.add(m, n, d["hair"], 8, group="knot")
    m, n = ellipse(CX, 3.8, 7.2, 3.4)
    s.add(m & (YY <= 3.8), n, d["hair"], 9, group="fan")


def hair_zanbara(s, d):
    """Loose short hair of a young wrestler who can't tie a topknot yet."""
    _cap(s, d, low=9)
    for dx, cy in ((-7, 8), (7, 8), (-3.5, 5.6), (3.5, 5.6), (0, 5.2)):
        m, n = ellipse(CX + dx, cy, 3.2, 2.6)
        s.add(m, n, d["hair"], 7, hard=False, group="head")


def hair_bald(s, d):
    m, n = ellipse(CX, 14, 9.1, 8.6)
    side = m & (np.abs(XX + 0.5 - CX) >= 7.8) & (YY >= 10) & (YY <= 15)
    s.add(side, n, d["hair"], 7, hard=False, group="head")


HAIR = dict(chonmage=hair_chonmage, oicho=hair_oicho, zanbara=hair_zanbara, bald=hair_bald)


# ---------------------------------------------------------------- aprons
def _rows(rows, palette):
    return [(x, y, palette[ch]) for y, r in enumerate(rows) for x, ch in enumerate(r) if ch != "."]


APRONS = {
    "sun": dict(ramp=["#b9b2a4", "#e6e0d2", "#fbf8f0", "#ffffff"], motif=_rows([
        "............",
        ".....rr.....",
        "...rrrrrr...",
        "..rrrRRrrr..",
        "..rrRRRRrr..",
        "..rrrRRrrr..",
        "...rrrrrr...",
        ".....rr.....",
        "............",
        "gggggggggggg",
    ], {"r": "#d62828", "R": "#ff5a4a", "g": "#d9a520"})),
    "wave": dict(ramp=["#0a2a4a", "#12406e", "#1a5a96", "#2a74b8"], motif=_rows([
        "............",
        ".ww.....ww..",
        "w..w...w..w.",
        "..wbw.....wb",
        ".wbbbw...wbb",
        "wbbbbbwwwbbb",
        "bbbbbbbbbbbb",
        "..ww....ww..",
        ".w..w..w..w.",
        "gggggggggggg",
    ], {"w": "#e8f4ff", "b": "#5aa0e0", "g": "#d9a520"})),
    "dragon": dict(ramp=["#1a0a2a", "#2c1048", "#3e1866", "#522284"], motif=_rows([
        ".......gG...",
        "......gGGg..",
        "..gg..gy.g..",
        ".gGGg.gg....",
        ".g..gGg.....",
        "....gGGg....",
        "...g..gGg...",
        "..gg...gGg..",
        ".......ggg..",
        "yyyyyyyyyyyy",
    ], {"g": "#1f9a4a", "G": "#5ae08a", "y": "#ffd23b"})),
    "sakura": dict(ramp=["#3a0a18", "#5a1228", "#7a1a38", "#962446"], motif=_rows([
        "............",
        ".p.p....p...",
        "pPyPp..pPp..",
        ".pPp....p...",
        "........p.p.",
        "..p....pPyPp",
        ".pPp....pPp.",
        "pPyPp...p...",
        ".p.p........",
        "gggggggggggg",
    ], {"p": "#ff8fb8", "P": "#ffd0e2", "y": "#ffe27a", "g": "#d9a520"})),
    "mountain": dict(ramp=["#2a2418", "#3e3624", "#544a32", "#6a5e40"], motif=_rows([
        "............",
        "...rr.......",
        "..rRRr......",
        ".....w......",
        "....wWw.....",
        "...wWWWw....",
        "..bbWWWbb...",
        ".bbbbbbbbbb.",
        "bbbbbbbbbbbb",
        "gggggggggggg",
    ], {"r": "#d62828", "R": "#ff6a4a", "w": "#ffffff", "W": "#d8e6f2", "b": "#3a6ea8", "g": "#d9a520"})),
}


# ---------------------------------------------------------------- face
def face(img, d):
    o = d["outline"]
    pupil = d.get("eye", "#1a1a2e")
    brow = d.get("brow_color", o)
    eyes = d.get("eyes", "normal")
    if eyes == "normal":
        px(img, mirror([(19, 14), (20, 14), (21, 14)]), o)
        px(img, mirror([(19, 15), (20, 15)]), "#ffffff")
        px(img, mirror([(21, 15)]), pupil)
    elif eyes == "wide":
        px(img, mirror([(19, 13), (20, 13), (21, 13)]), o)
        px(img, mirror([(19, 14), (20, 14), (21, 14), (19, 15), (21, 15)]), "#ffffff")
        px(img, mirror([(20, 15)]), pupil)
        px(img, mirror([(18, 14), (18, 15), (22, 14), (22, 15)]), o)
    elif eyes == "closed":
        px(img, mirror([(19, 15), (20, 15), (21, 15)]), o)
    elif eyes == "happy":
        px(img, mirror([(19, 15), (20, 14), (21, 15)]), o)
    elif eyes == "glare":
        px(img, mirror([(19, 14), (20, 14), (21, 14), (22, 14)]), o)
        px(img, mirror([(20, 15)]), "#ffffff")
        px(img, mirror([(21, 15)]), pupil)
    br = d.get("brow", "angry")
    if br == "angry":
        px(img, mirror([(18, 11), (19, 11), (20, 12), (21, 12), (22, 13)]), brow)
    elif br == "thick":
        px(img, mirror([(18, 11), (19, 11), (20, 12), (21, 12), (22, 13),
                        (18, 12), (19, 12), (20, 13), (21, 13)]), brow)
    elif br == "flat":
        px(img, mirror([(18, 12), (19, 12), (20, 12), (21, 12)]), brow)
    elif br == "worried":
        px(img, mirror([(18, 13), (19, 12), (20, 12), (21, 11)]), brow)
    elif br == "bushy":
        px(img, mirror([(17, 12), (18, 11), (19, 11), (20, 11), (21, 12),
                        (17, 13), (18, 12), (19, 12), (20, 12)]), brow)
    sk = d["skin"]
    px(img, [(24, 16), (24, 17)], sk[0])
    px(img, [(23, 17)], sk[1])
    mo = d.get("mouth", "flat")
    if mo == "grit":
        px(img, [(21, 18), (26, 18), (22, 18), (23, 18), (24, 18), (25, 18)], o)
        px(img, [(22, 19), (23, 19), (24, 19), (25, 19)], "#f4f1e8")
        px(img, [(22, 20), (23, 20), (24, 20), (25, 20)], o)
    elif mo == "flat":
        px(img, [(22, 19), (23, 19), (24, 19), (25, 19)], o)
    elif mo == "smirk":
        px(img, [(21, 19), (22, 19), (23, 19), (24, 19), (25, 18)], o)
    elif mo == "smile":
        px(img, [(21, 18), (22, 19), (23, 19), (24, 19), (25, 19), (26, 18)], o)
    elif mo == "grin":
        px(img, [(21, 18), (22, 18), (23, 18), (24, 18), (25, 18), (26, 18), (22, 20), (23, 20), (24, 20), (25, 20)], o)
        px(img, [(22, 19), (23, 19), (24, 19), (25, 19)], "#f4f1e8")
    elif mo == "shout":
        px(img, [(22, 18), (23, 18), (24, 18), (25, 18), (21, 19), (26, 19), (21, 20), (26, 20), (22, 21), (23, 21), (24, 21), (25, 21)], o)
        px(img, [(22, 19), (23, 19), (24, 19), (25, 19)], "#f4f1e8")
        px(img, [(22, 20), (25, 20)], "#5a1020")
        px(img, [(23, 20), (24, 20)], "#c0303a")
    elif mo == "frown":
        px(img, [(21, 20), (22, 19), (23, 19), (24, 19), (25, 19), (26, 20)], o)
    if d.get("blush"):
        px(img, mirror([(17, 17), (18, 17)]), d["blush"])


# ---------------------------------------------------------------- gear
def gear(img, s, d, b):
    o = d["outline"]
    top = b["tcy"] + 6
    trx = b["trx"]
    lo, hi = int(23.5 - trx - 2), int(23.5 + trx + 3)

    def on(x, y, groups=("torso", "mawashi")):
        X = int(x) + OX
        return 0 <= X < W and 0 <= y < H and s.group_at(X, int(y)) in groups

    for g in d.get("gear", []):
        if g == "tsuna":
            rope = ["#8a8478", "#cfc9bb", "#f7f4ec", "#ffffff"]
            y0 = top - 1
            for x in range(lo, hi):
                if not on(x, y0):
                    continue
                px(img, [(x, y0 - 2)], o)
                px(img, [(x, y0 - 1)], rope[3 if (x // 2) % 2 else 2])
                px(img, [(x, y0)], rope[2 if (x // 2) % 2 else 1])
                px(img, [(x, y0 + 1)], rope[1 if (x // 2) % 2 else 0])
                px(img, [(x, y0 + 2)], o)
            tab = ["WW.", ".WW", "WW.", ".WW", "WW."]
            for sx in (11, 19, 26, 34):
                for ry, row in enumerate(tab):
                    for rx, ch in enumerate(row):
                        if ch == "W":
                            px(img, [(sx + rx, y0 + 3 + ry)], "#ffffff" if rx < 2 else "#c9c4b8")
                for ry, row in enumerate(tab):
                    first, last = row.index("W"), row.rindex("W")
                    px(img, [(sx + first - 1, y0 + 3 + ry), (sx + last + 1, y0 + 3 + ry)], o)
        elif g == "sagari":
            c = d.get("sagari", d["mawashi"][1])
            for x in range(int(23.5 - trx + 5), int(23.5 + trx - 4), 2):
                if abs(x - 23.5) < 5:
                    continue
                for y in range(top + 6, top + 11):
                    px(img, [(x, y)], c)
                px(img, [(x, top + 11)], o)
        elif g == "knee_tape":
            for x in range(int(23.5 - b["leg"] - 4), int(23.5 - b["leg"] + 4)):
                px(img, [(x, 53)], "#f2f2f2"); px(img, [(x, 54)], "#c8c8c8")
        elif g == "arm_tape":
            ax = 23.5 + b["arm"] + 1
            for x in range(int(ax - 4), int(ax + 4)):
                for y, c in ((b["tcy"] + 2, "#ffffff"), (b["tcy"] + 3, "#d8d8d8"), (b["tcy"] + 4, "#ffffff")):
                    if on(x, y, (f"arm1",)):
                        px(img, [(x, y)], c)
        elif g == "sideburns":
            px(img, mirror([(15, 15), (16, 15), (15, 16), (16, 16), (15, 17)]), d["hair"][1])
        elif g == "stubble":
            for x, y in ((20, 20), (22, 21), (24, 21), (26, 20), (21, 21), (25, 21), (23, 20), (27, 19), (19, 19)):
                px(img, [(x, y)], d["skin"][0])
        elif g == "scar":
            px(img, [(26, 12), (27, 13), (27, 14), (28, 16), (28, 17)], "#b85a4a")
        elif g == "nose_tape":
            px(img, [(23, 16), (24, 16), (22, 17), (25, 17)], "#ffffff")
            px(img, [(23, 17), (24, 17)], "#e6e0d0")
        elif g == "black_eye":
            px(img, [(26, 16), (27, 16), (28, 16), (29, 15), (29, 14)], "#6a3a8a")
        elif g == "beads":
            for i, x in enumerate(range(13, 35)):
                yy = 23 + int(3.5 * math.sin(math.pi * (x - 13) / 22))
                px(img, [(x, yy)], "#5a2a10" if i % 2 else "#c97a3a")
        elif g == "chest_hair":
            for x, y in ((22, 26), (25, 26), (23, 27), (24, 28), (23, 29)):
                px(img, [(x, y)], d["hair"][1])
        elif g == "salt":
            hx = 23.5 - b["hand"]
            for x, y in ((hx - 3, b["tcy"] + 3), (hx - 1, b["tcy"] + 1), (hx + 1, b["tcy"] + 2), (hx - 4, b["tcy"]), (hx, b["tcy"] - 1)):
                px(img, [(x, y)], "#ffffff")
    apron = d.get("apron")
    if apron:
        for x, y, c in APRONS[apron]["motif"]:
            px(img, [(18 + x, top + 7 + y)], c)
        for x in range(17, 31):
            px(img, [(x, 58)], "#ffd23b" if x % 2 else "#b88a10")
            px(img, [(x, 59)], "#b88a10" if x % 2 == 0 else o)


def details(img, d, b):
    sk = d["skin"]
    px(img, [(23, b["tcy"] + 4), (24, b["tcy"] + 4)], sk[0])
    px(img, mirror([(17, b["tcy"] - 8), (18, b["tcy"] - 7), (19, b["tcy"] - 7), (20, b["tcy"] - 7), (21, b["tcy"] - 8)]), sk[1])
    hx = 23.5 - b["hand"]
    px(img, mirror([(int(hx - b["hr"] + 1.5), b["tcy"] + 8), (int(hx - b["hr"] + 1.5), b["tcy"] + 9)]), sk[0])
    lx = 23.5 - b["leg"]
    px(img, mirror([(int(lx) - 2, 59), (int(lx), 59)]), sk[0])


def render(d):
    s, b = build(d)
    img = s.finish()
    details(img, d, b)
    face(img, d)
    gear(img, s, d, b)
    return img


def aura(img, d, frame):
    h, w = img.shape[:2]
    cur = img[:, :, 3] > 0
    layers = []
    for _ in range(6):
        nxt = cur.copy()
        nxt[1:, :] |= cur[:-1, :]; nxt[:-1, :] |= cur[1:, :]
        nxt[:, 1:] |= cur[:, :-1]; nxt[:, :-1] |= cur[:, 1:]
        nxt[:-2, :] |= cur[2:, :]
        layers.append(nxt & ~cur)
        cur = nxt
    out = img.copy()
    cols = [hexc(c) for c in d["aura"]]
    for i, ring in enumerate(layers):
        for y, x in zip(*np.nonzero(ring)):
            if i <= 1:
                c, a = cols[0], 235
            elif i <= 3:
                c, a = cols[1], 200
            else:
                if ((y + frame * 3) // 2 + x) % 4 == 0:
                    continue
                c, a = cols[2], 170
            out[y, x] = (*c[:3], a)
    for x in range(4, w - 4, 4):
        t = np.nonzero(cur[:, x])[0]
        if len(t) == 0:
            continue
        hh = 2 + (x * 3 + frame * 2) % 5
        for k in range(hh):
            yy = t[0] - k
            for xx in (x, x + 1):
                if 0 <= yy < h and out[yy, xx, 3] == 0 and not (k == hh - 1 and xx == x + 1):
                    out[yy, xx] = (*cols[2][:3], 170)
    return out


# ---------------------------------------------------------------- palettes
SKIN = {
    "fair":   ["#9c5a3c", "#d9895f", "#f2b48a", "#ffd9b8"],
    "pale":   ["#a06a52", "#dfa385", "#f7cdb2", "#fff0e2"],
    "tan":    ["#8d4f2c", "#c97b4b", "#eaa877", "#ffd2a6"],
    "olive":  ["#6e4628", "#a8703e", "#cf9a62", "#f0c48c"],
    "brown":  ["#4e2c18", "#7e4a2a", "#a86e44", "#d0966a"],
    "deep":   ["#2e1a10", "#52301c", "#78492c", "#a06a46"],
}
HAIRC = {
    "black": ["#0d0d18", "#1d1d30", "#2f2f4a", "#4b4b6e"],
    "brown": ["#2a1408", "#4a2812", "#6e4020", "#94602e"],
    "gray":  ["#5a5a64", "#8a8a94", "#b8b8c4", "#e0e0ea"],
    "white": ["#8a8a94", "#bcbcc6", "#e4e4ec", "#ffffff"],
    "blue":  ["#0a1028", "#16204a", "#24346e", "#38508e"],
}


def mw(base):
    pals = {
        "purple": ["#2a1440", "#4b2475", "#6e3aa6", "#9c6bd1"],
        "black":  ["#0a0a10", "#1a1a26", "#2c2c3e", "#46465e"],
        "red":    ["#4a0a0a", "#8a1616", "#c82828", "#f05a4a"],
        "gold":   ["#6a4a00", "#a87a00", "#e0aa10", "#ffd860"],
        "navy":   ["#0b1a3d", "#173a7a", "#2a63c4", "#64a0ff"],
        "teal":   ["#053a3a", "#0a6464", "#14948c", "#4ad0c0"],
        "green":  ["#0d3b22", "#16693a", "#22a35a", "#6ee09a"],
        "pink":   ["#5a1030", "#9a2458", "#d84a8a", "#ff8ec0"],
        "orange": ["#7a2c00", "#c75200", "#ff7a1a", "#ffb15c"],
        "silver": ["#3a3e48", "#6a707e", "#a0a8b8", "#dce2ee"],
        "maroon": ["#2a0410", "#500a20", "#7a1432", "#a82a4c"],
        "brown":  ["#2a1608", "#4e2c12", "#7a4a22", "#a87038"],
        "sky":    ["#123a5a", "#1e6a9a", "#3a9ad8", "#8ad0ff"],
        "white":  ["#8a8478", "#cfc9bb", "#f0ece2", "#ffffff"],
    }
    return pals[base]


AURA = {
    "gold": ["#fff7c2", "#ffe066", "#ffb627"],
    "red": ["#ffd1c2", "#ff6a3d", "#c41e1e"],
    "blue": ["#e8f7ff", "#7fdcff", "#2d8cff"],
    "green": ["#f2fff4", "#9dffb8", "#2bd46a"],
    "purple": ["#f6e8ff", "#c28aff", "#7a2ad8"],
    "pink": ["#fff0f6", "#ff9ec8", "#e0408a"],
    "white": ["#ffffff", "#e6eef8", "#9ab0c8"],
}


def D(key, name, blurb, skin, mawashi, hair="black", outline="#1f1420", aura="gold", **kw):
    return dict(key=key, name=name, blurb=blurb, skin=SKIN[skin], mawashi=mw(mawashi),
                hair=HAIRC[hair], outline=outline, aura=AURA[aura], **kw)


DESIGNS = [
    D("kaiju_yokozuna", "Kaiju Yokozuna", "The original. Tsuna, topknot, purple silk.",
      "fair", "purple", mouth="flat", blush="#e8836a", gear=["tsuna"]),
    D("akagane", "Akagane", "Copper yokozuna. Ginkgo topknot, red silk.",
      "tan", "red", hairstyle="oicho", mouth="grit", gear=["tsuna"], aura="red"),
    D("kurogane", "Kurogane", "Iron wall. Heavy build, black mawashi.",
      "brown", "black", body="heavy", mouth="frown", brow="thick", gear=["sagari"], sagari="#d9a520"),
    D("aonami", "Aonami", "Blue wave apron. Calm and patient.",
      "fair", "navy", hairstyle="oicho", eyes="closed", brow="flat", apron="wave", aura="blue"),
    D("kinboshi", "Kinboshi", "Golden star. Loves the crowd.",
      "pale", "gold", mouth="grin", eyes="happy", brow="flat", blush="#f08a7a", gear=["tsuna"]),
    D("momo", "Momo", "Rookie with no topknot yet. Pink mawashi.",
      "pale", "pink", body="small", hairstyle="zanbara", hair="brown", eyes="wide", brow="worried",
      mouth="smile", blush="#ff8a9a", aura="pink"),
    D("yamakaze", "Yamakaze", "Mountain wind. Huge belly, sideburns.",
      "olive", "brown", body="heavy", mouth="smirk", gear=["sideburns", "sagari"]),
    D("hayabusa", "Hayabusa", "Falcon. Lean, fast, taped knee.",
      "tan", "teal", body="lean", eyes="glare", mouth="flat", gear=["knee_tape", "sagari"], aura="blue"),
    D("tetsuo", "Tetsuo", "Veteran brawler. Scar, taped arm, silver.",
      "brown", "silver", eyes="glare", mouth="grit", gear=["scar", "arm_tape", "sagari"], aura="white"),
    D("hinode", "Hinode", "Rising sun ceremonial apron.",
      "fair", "red", hairstyle="oicho", mouth="flat", apron="sun", aura="red"),
    D("ryujin", "Ryujin", "Dragon apron. Deep purple silk.",
      "deep", "purple", hairstyle="oicho", eyes="glare", mouth="smirk", apron="dragon", aura="purple"),
    D("oyakata", "Oyakata", "Retired stablemaster. Gray hair, prayer beads.",
      "pale", "green", hair="gray", eyes="closed", brow="bushy", brow_color="#b8b8c4", mouth="frown",
      gear=["beads"], aura="green"),
    D("koishi", "Koishi", "Little pebble. Small, round and eager.",
      "tan", "sky", body="small", eyes="wide", brow="flat", mouth="shout", blush="#e88060",
      gear=["salt"], aura="blue"),
    D("ikazuchi", "Ikazuchi", "Thunder yokozuna. Gold silk, fierce glare.",
      "deep", "gold", hairstyle="oicho", eyes="glare", brow="thick", mouth="grit", gear=["tsuna"]),
    D("sakura", "Sakuragawa", "Cherry blossom apron. Graceful technician.",
      "pale", "maroon", hairstyle="oicho", hair="brown", mouth="smile", eyes="happy", brow="flat",
      apron="sakura", aura="pink"),
    D("tora", "Tora", "Tiger. Stubble, bandaged nose, orange silk.",
      "olive", "orange", eyes="glare", mouth="smirk", gear=["stubble", "nose_tape", "sagari"], aura="red"),
    D("fujiyama", "Fujiyama", "Mount Fuji apron. Massive and unmovable.",
      "fair", "white", body="heavy", hairstyle="oicho", mouth="flat", brow="thick", apron="mountain", aura="white"),
    D("kumo", "Kumo", "Cloud spider. Lean grappler, black eye.",
      "pale", "black", body="lean", hair="blue", mouth="grin", gear=["black_eye", "sagari"], aura="purple"),
    D("bonzan", "Bonzan", "Shaved monk wrestler. Beads and calm.",
      "olive", "orange", hairstyle="bald", eyes="closed", brow="flat", mouth="smile",
      gear=["beads"], aura="gold"),
    D("shiroi", "Shiroi", "White-rope yokozuna in navy. Silent and cold.",
      "brown", "navy", hairstyle="oicho", eyes="glare", brow="angry", mouth="flat",
      gear=["tsuna"], aura="white"),
]


def up(a, k):
    im = Image.fromarray(a, "RGBA")
    return im.resize((im.width * k, im.height * k), Image.NEAREST)


def pad(a, p=8):
    o = np.zeros((a.shape[0] + 2 * p, a.shape[1] + 2 * p, 4), dtype=np.uint8)
    o[p:p + a.shape[0], p:p + a.shape[1]] = a
    return o


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rendered = []
    for d in DESIGNS:
        idle = render(d)
        Image.fromarray(idle, "RGBA").save(OUT / f"{d['key']}.png")
        up(idle, 6).save(OUT / f"{d['key']}_x6.png")
        frames = [up(aura(pad(idle), d, f), 6) for f in range(3)]
        frames[0].save(OUT / f"{d['key']}_powerup_x6.gif", save_all=True,
                       append_images=frames[1:], duration=110, loop=0, disposal=2)
        rendered.append((d, idle))
    sheet(rendered)


def sheet(items, cols=5, k=4):
    sw, sh = W * k, H * k
    cell_w, cell_h = sw + 40, sh + 110
    rows = math.ceil(len(items) / cols)
    img = Image.new("RGBA", (cols * cell_w + 40, rows * cell_h + 90), (24, 26, 44, 255))
    dr = ImageDraw.Draw(img)
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    reg = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    f_h, f_t, f_s = ImageFont.truetype(bold, 30), ImageFont.truetype(bold, 22), ImageFont.truetype(reg, 16)
    dr.text((24, 22), "RIKISHI SET - pick your favourites by number", font=f_h, fill=(255, 226, 120, 255))
    for i, (d, a) in enumerate(items):
        cx, cy = 20 + (i % cols) * cell_w, 80 + (i // cols) * cell_h
        dr.rounded_rectangle([cx, cy, cx + cell_w - 10, cy + cell_h - 10], 14, fill=(36, 40, 66, 255))
        dr.ellipse([cx + 30, cy + sh - 6, cx + cell_w - 40, cy + sh + 22], fill=(20, 22, 36, 255))
        img.alpha_composite(up(a, k), (cx + 15, cy + 8))
        dr.text((cx + 14, cy + sh + 26), f"{i + 1}. {d['name']}", font=f_t, fill=(255, 255, 255, 255))
        words, line, lines = d["blurb"].split(), "", []
        for w_ in words:
            if dr.textlength((line + " " + w_).strip(), font=f_s) > cell_w - 34:
                lines.append(line); line = w_
            else:
                line = (line + " " + w_).strip()
        lines.append(line)
        for j, ln in enumerate(lines[:2]):
            dr.text((cx + 14, cy + sh + 54 + j * 20), ln, font=f_s, fill=(190, 196, 220, 255))
    img.save(OUT / "rikishi_lineup.png")


if __name__ == "__main__":
    main()
