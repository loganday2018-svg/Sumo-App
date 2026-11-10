"""Generate GBA-style (Legacy of Goku 2 inspired) sumo sprite concepts.

Each sprite is built from shaded shapes on a small pixel grid, then given
dark outlines, a hand-placed face and per-design accessories.
Run:  python sprites/make_sumo_sprites.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 48, 60
CX = 23.5
OUT = Path(__file__).parent / "concepts"
LIGHT = np.array([-0.45, -0.7, 0.55])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def hexc(h: str) -> tuple[int, int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)


class Sprite:
    def __init__(self, outline: str):
        self.part = np.full((H, W), -1, dtype=int)  # part id per pixel
        self.z = np.full((H, W), -1, dtype=int)
        self.col = np.zeros((H, W, 4), dtype=np.uint8)
        self.ramps: dict[int, list] = {}
        self.hard: dict[int, bool] = {}
        self.group: dict[int, str] = {}
        self.outline = hexc(outline)
        self.next_id = 0

    def add(self, mask, normals, ramp, z, hard=True, group=None):
        pid = self.next_id
        self.next_id += 1
        self.ramps[pid] = [hexc(c) for c in ramp]
        self.hard[pid] = hard
        self.group[pid] = group or str(pid)
        inten = np.clip(normals @ LIGHT, 0, 1)
        levels = np.digitize(inten, [0.18, 0.45, 0.78])
        for y, x in zip(*np.nonzero(mask)):
            if z >= self.z[y, x]:
                self.z[y, x] = z
                self.part[y, x] = pid
                self.col[y, x] = self.ramps[pid][levels[y, x]]
        return pid

    def finish(self) -> np.ndarray:
        img = self.col.copy()
        # internal lines where a front part meets a part behind it
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
        # outer silhouette
        filled = self.part >= 0
        out = img.copy()
        for y in range(H):
            for x in range(W):
                if filled[y, x]:
                    continue
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < H and 0 <= xx < W and filled[yy, xx]:
                        out[y, x] = self.outline
                        break
        return out


YY, XX = np.mgrid[0:H, 0:W].astype(float)


def ellipse(cx, cy, rx, ry, rot=0.0):
    c, s = math.cos(rot), math.sin(rot)
    dx, dy = XX + 0.5 - cx, YY + 0.5 - cy
    u = (dx * c + dy * s) / rx
    v = (-dx * s + dy * c) / ry
    d = u * u + v * v
    mask = d <= 1.0
    nz = np.sqrt(np.clip(1 - d, 0, 1))
    nx = u * c - v * s
    ny = u * s + v * c
    n = np.stack([nx, ny, nz * 1.1], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return mask, n


def cylinder(mask, cx, rx):
    u = np.clip((XX + 0.5 - cx) / rx, -1, 1)
    n = np.stack([u, np.full_like(u, -0.15), np.sqrt(1 - u * u) + 0.2], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return mask, n


def px(img, pts, color):
    c = hexc(color) if isinstance(color, str) else color
    for x, y in pts:
        if 0 <= x < W and 0 <= y < H:
            img[y, x] = c


def mirror(pts):
    return pts + [(47 - x, y) for x, y in pts]


# ---------------------------------------------------------------- base body
def build_body(d, bob=0):
    s = Sprite(d["outline"])
    skin, mw = d["skin"], d["mawashi"]
    # legs
    for side in (-1, 1):
        m, n = ellipse(CX + side * 8, 51, 6.2, 7.5)
        s.add(m, n, skin, z=1, group="leg%d" % side)
    # torso / belly
    m, n = ellipse(CX, 33 + bob, 16.5, 15.5)
    torso = m
    s.add(m, n, skin, z=2, group="torso")
    # chest line / pec split is drawn later as detail
    # mawashi band
    band = ((YY >= 39 + bob) & (YY <= 44 + bob)) & (ellipse(CX, 33 + bob, 17.5, 16.5)[0])
    m, n = cylinder(band, CX, 17)
    s.add(m, n, mw, z=3, hard=False, group="mawashi")
    # front panel down between the legs
    front = (np.abs(XX + 0.5 - CX) <= 3.5) & (YY >= 44 + bob) & (YY <= 50)
    m, n = cylinder(front, CX, 5)
    s.add(m, n, mw, z=3, hard=False, group="mawashi")
    # arms
    for side in (-1, 1):
        m, n = ellipse(CX + side * 16, 31 + bob, 5.5, 10.5, rot=side * -0.22)
        s.add(m, n, skin, z=4, group="arm%d" % side)
        m, n = ellipse(CX + side * 18, 41 + bob, 4.3, 4.0)
        s.add(m, n, skin, z=5, group="arm%d" % side)
    # head
    m, n = ellipse(CX, 14 + bob, 9.0, 8.5)
    s.add(m, n, skin, z=6, group="head")
    return s


def finish(s, d, bob=0):
    img = s.finish()
    o = d["outline"]
    sk = d["skin"]
    b = bob
    # belly button + pec shading
    px(img, [(23, 37 + b), (24, 37 + b)], sk[0])
    px(img, mirror([(17, 25 + b), (18, 26 + b), (19, 26 + b), (20, 26 + b), (21, 25 + b)]), sk[1])
    # fist knuckle lines
    px(img, mirror([(5, 41 + b), (5, 42 + b)]), sk[0])
    # toes
    px(img, mirror([(13, 58), (15, 58)]), sk[0])
    return img


def face(img, d, bob=0, brow="angry"):
    o, b = d["outline"], bob
    eye_w = "#ffffff"
    pupil = d.get("eye", "#1a1a2e")
    # eyes (2px wide, lid on top)
    px(img, mirror([(19, 14 + b), (20, 14 + b), (21, 14 + b)]), o)        # upper lid
    px(img, mirror([(19, 15 + b), (20, 15 + b)]), eye_w)
    px(img, [(21, 15 + b), (26, 15 + b)], pupil)
    px(img, mirror([(21, 15 + b)]), pupil)
    if brow == "angry":
        px(img, mirror([(18, 11 + b), (19, 11 + b), (20, 12 + b), (21, 12 + b), (22, 13 + b)]), d.get("brow", o))
    elif brow == "bushy":
        px(img, mirror([(17, 12 + b), (18, 11 + b), (19, 11 + b), (20, 11 + b), (21, 12 + b),
                        (17, 13 + b), (18, 12 + b), (19, 12 + b), (20, 12 + b)]), d.get("brow", o))
    # nose
    px(img, [(24, 16 + b), (24, 17 + b)], d["skin"][0])
    px(img, [(23, 17 + b)], d["skin"][1])
    # mouth
    mouth = d.get("mouth", "grit")
    if mouth == "grit":
        px(img, [(21, 18 + b), (26, 18 + b)], o)
        px(img, [(22, 18 + b), (23, 18 + b), (24, 18 + b), (25, 18 + b)], o)
        px(img, [(22, 19 + b), (23, 19 + b), (24, 19 + b), (25, 19 + b)], "#f4f1e8")
        px(img, [(22, 20 + b), (23, 20 + b), (24, 20 + b), (25, 20 + b)], o)
    elif mouth == "smirk":
        px(img, [(21, 19 + b), (22, 19 + b), (23, 19 + b), (24, 19 + b), (25, 18 + b)], o)
    elif mouth == "flat":
        px(img, [(22, 19 + b), (23, 19 + b), (24, 19 + b), (25, 19 + b)], o)
    # cheek blush
    if d.get("blush"):
        px(img, mirror([(17, 17 + b), (18, 17 + b)]), d["blush"])


def hair_topknot(s, d, bob=0):
    hr = d["hair"]
    cap = ellipse(CX, 14 + bob, 8.6, 8.6)[0] & (YY <= 10 + bob)
    side = ellipse(CX, 14 + bob, 8.6, 8.6)[0] & (np.abs(XX + 0.5 - CX) >= 7.2) & (YY <= 15 + bob)
    m, n = ellipse(CX, 14 + bob, 8.5, 8.5)
    s.add(cap | side, n, hr, z=7, hard=False, group="head")
    m, n = ellipse(CX, 4 + bob, 3.6, 2.6)
    s.add(m, n, hr, z=8, hard=True, group="knot")
    m, n = ellipse(CX + 3.5, 3.5 + bob, 3.0, 1.8, rot=-0.3)
    s.add(m, n, hr, z=8, hard=True, group="knot")


def hair_spiky(s, d, bob=0):
    hr = d["hair"]
    m, n = ellipse(CX, 13 + bob, 9.2, 8.0)
    cap = m & (YY <= 11 + bob)
    s.add(cap, n, hr, z=7, hard=False, group="hair")
    spikes = [(-10, 4, 3.0, 6.5, -0.9), (-6, 2, 2.6, 6.5, -0.45), (0, 0.5, 2.6, 6.5, 0.0),
              (6, 2, 2.6, 6.5, 0.45), (10, 4, 3.0, 6.5, 0.9), (-11, 9, 2.4, 5, -1.3),
              (11, 9, 2.4, 5, 1.3)]
    for dx, cy, rx, ry, rot in spikes:
        m, n = ellipse(CX + dx, cy + bob + 2, rx, ry, rot=rot)
        s.add(m, n, hr, z=7, hard=False, group="hair")
    # bangs over forehead
    for dx, rot in ((-3, -0.35), (2.5, 0.4)):
        m, n = ellipse(CX + dx, 10 + bob, 1.6, 3.2, rot=rot)
        s.add(m, n, hr, z=8, hard=False, group="hair")


def hair_wild(s, d, bob=0):
    hr = d["hair"]
    m, n = ellipse(CX, 13 + bob, 9.5, 8.0)
    s.add(m & (YY <= 11 + bob), n, hr, z=7, hard=False, group="hair")
    for dx, cy in ((-9, 13), (9, 13), (-6, 6), (6, 6), (0, 4.5)):
        m, n = ellipse(CX + dx, cy + bob, 3.4, 3.2)
        s.add(m, n, hr, z=7, hard=False, group="hair")


# ---------------------------------------------------------------- designs
DESIGNS = [
    dict(
        key="kaiju_yokozuna", name="KAIJU YOKOZUNA",
        blurb="Classic grand champion. Tsuna rope, topknot, purple silk.",
        outline="#1f1420",
        skin=["#9c5a3c", "#d9895f", "#f2b48a", "#ffd9b8"],
        mawashi=["#2a1440", "#4b2475", "#6e3aa6", "#9c6bd1"],
        hair=["#0d0d18", "#1d1d30", "#2f2f4a", "#4b4b6e"],
        hair_fn=hair_topknot, mouth="flat", blush="#e8836a",
        aura=["#fff7c2", "#ffe066", "#ffb627"],
        extras="tsuna",
    ),
    dict(
        key="super_rikishi", name="SUPER RIKISHI",
        blurb="Powered-up warrior. Golden hair, orange mawashi, blue wristbands.",
        outline="#2a1608",
        skin=["#8d4f2c", "#c97b4b", "#eaa877", "#ffd2a6"],
        mawashi=["#7a2c00", "#c75200", "#ff7a1a", "#ffb15c"],
        hair=["#a8680a", "#e8a910", "#ffd83b", "#fff3a0"],
        hair_fn=hair_spiky, mouth="grit", eye="#0e6b5e", brow="#a8680a",
        aura=["#fffbd6", "#ffe75a", "#ffc21a"],
        extras="wristbands",
    ),
    dict(
        key="oni_crusher", name="ONI CRUSHER",
        blurb="Demon brawler. Red hide, horns, tiger-stripe mawashi.",
        outline="#240a0a",
        skin=["#6e1414", "#a82424", "#d9443a", "#ff7a63"],
        mawashi=["#7a5200", "#c98a00", "#ffc21a", "#ffe27a"],
        hair=["#0b0b10", "#1c1c26", "#2e2e3d", "#4a4a5e"],
        hair_fn=hair_wild, mouth="grit", eye="#ffcc00",
        aura=["#ffd1c2", "#ff6a3d", "#c41e1e"],
        extras="oni",
    ),
    dict(
        key="storm_warden", name="STORM WARDEN",
        blurb="Stoic thunder fighter. Headband, scar, navy mawashi.",
        outline="#120d14",
        skin=["#3e2416", "#6b4128", "#93603d", "#bd8a62"],
        mawashi=["#0b1a3d", "#173a7a", "#2a63c4", "#64a0ff"],
        hair=["#08080c", "#14141c", "#22222e", "#38384a"],
        hair_fn=hair_topknot, mouth="smirk",
        aura=["#e8f7ff", "#7fdcff", "#2d8cff"],
        extras="headband",
    ),
    dict(
        key="old_master", name="OLD MASTER",
        blurb="Ancient sensei. White beard, shades, jade mawashi.",
        outline="#1a1a14",
        skin=["#a0644a", "#dc9a78", "#f5c4a4", "#ffe3cf"],
        mawashi=["#0d3b22", "#16693a", "#22a35a", "#6ee09a"],
        hair=["#8a8a94", "#bcbcc6", "#e4e4ec", "#ffffff"],
        hair_fn=hair_topknot, mouth=None, brow="#e4e4ec",
        aura=["#f2fff4", "#9dffb8", "#2bd46a"],
        extras="master",
    ),
]


def extras(img, s_unused, d, bob=0):
    b = bob
    kind = d["extras"]
    o = d["outline"]
    if kind == "tsuna":
        rope = ["#9a9488", "#d9d4c7", "#f7f4ec"]
        for x in range(8, 40):
            # twisted rope band just above the mawashi
            y0 = 38 + b
            px(img, [(x, y0)], rope[(x // 2) % 2 + 1])
            px(img, [(x, y0 + 1)], rope[((x + 1) // 2) % 2])
        # shide (zig-zag paper) hanging from rope
        tab = ["WW.", ".WW", "WW.", ".WW", "WW."]
        for sx in (11, 19, 26, 34):
            for ry, row in enumerate(tab):
                for rx, ch in enumerate(row):
                    if ch == "W":
                        px(img, [(sx + rx, 40 + ry + b)], "#ffffff" if rx < 2 else "#c9c4b8")
            for ry, row in enumerate(tab):
                for rx in (-1, 3):
                    xx = sx + rx
                    if 0 <= xx < W and img[40 + ry + b, xx, :3].tolist() != [255, 255, 255]:
                        px(img, [(xx, 40 + ry + b)], o)
        # rope bow knot at back shows as loops at sides
        px(img, [(7, 38 + b), (40, 38 + b), (7, 39 + b), (40, 39 + b)], o)
    elif kind == "wristbands":
        band = ["#0f2a6b", "#1f4fc2", "#4a83ff"]
        for side in (-1, 1):
            cx = 23.5 + side * 17.5
            for x in range(int(cx - 3.5), int(cx + 4.5)):
                px(img, [(x, 36 + b)], band[1])
                px(img, [(x, 37 + b)], band[0])
                px(img, [(x, 35 + b)], band[2])
        # sagari (stiff strings) on the mawashi
        for x in range(16, 33, 3):
            if abs(x - 23.5) < 4:
                continue
            for y in range(45, 49):
                px(img, [(x, y + b)], d["mawashi"][1])
    elif kind == "oni":
        horn = ["#8a7a5c", "#d9cba3", "#fff4d6"]
        for side in (0, 1):
            pts = [(15, 7), (14, 6), (14, 5), (13, 4), (13, 3), (12, 2)]
            pts = pts if side == 0 else [(47 - x, y) for x, y in pts]
            for i, (x, y) in enumerate(pts):
                px(img, [(x, y + b)], horn[2] if i % 2 else horn[1])
                px(img, [(x - 1 if side == 0 else x + 1, y + b)], o)
        # tiger stripes on mawashi
        for x in range(9, 39, 4):
            px(img, [(x, 39 + b), (x + 1, 40 + b), (x + 1, 41 + b), (x, 42 + b)], "#2a1a08")
        # fangs
        px(img, [(22, 20 + b), (25, 20 + b)], "#ffffff")
        # war paint stripes on cheeks
        px(img, mirror([(16, 16 + b), (17, 17 + b), (16, 18 + b)]), "#2a0a0a")
    elif kind == "headband":
        hb = ["#a8a8b8", "#e6e6f0", "#ffffff"]
        for x in range(15, 33):
            px(img, [(x, 10 + b)], hb[2])
            px(img, [(x, 11 + b)], hb[1])
        px(img, [(15, 10 + b), (15, 11 + b), (32, 10 + b), (32, 11 + b)], o)
        # trailing tails of headband
        px(img, [(33, 11 + b), (34, 12 + b), (35, 12 + b), (36, 13 + b), (34, 13 + b), (35, 14 + b)], hb[1])
        px(img, [(37, 13 + b), (36, 14 + b), (36, 15 + b)], o)
        # red rising-sun dot
        px(img, [(23, 10 + b), (24, 10 + b), (23, 11 + b), (24, 11 + b)], "#d62828")
        # scar over left eye
        px(img, [(19, 13 + b), (20, 14 + b), (20, 16 + b), (21, 17 + b)], "#c26a5a")
        # lightning tattoo on right arm
        px(img, [(39, 27 + b), (40, 28 + b), (39, 29 + b), (40, 30 + b), (41, 31 + b)], "#64a0ff")
    elif kind == "master":
        sh = "#101018"
        # sunglasses
        px(img, mirror([(18, 14 + b), (19, 14 + b), (20, 14 + b), (21, 14 + b),
                        (18, 15 + b), (19, 15 + b), (20, 15 + b), (21, 15 + b),
                        (19, 16 + b), (20, 16 + b)]), sh)
        px(img, [(22, 14 + b), (23, 14 + b), (24, 14 + b), (25, 14 + b)], sh)
        px(img, mirror([(19, 14 + b)]), "#7ad7ff")
        # moustache + long beard
        beard = d["hair"]
        px(img, mirror([(19, 18 + b), (20, 18 + b), (21, 18 + b), (22, 18 + b), (23, 18 + b), (18, 19 + b)]), beard[3])
        for y in range(19, 31):
            half = max(1, 5 - (y - 19) // 3)
            for x in range(int(24 - half), int(24 + half)):
                c = beard[3] if (x + y) % 3 == 0 else beard[2]
                px(img, [(x, y + b)], c)
            px(img, [(int(24 - half) - 1, y + b), (int(24 + half), y + b)], o)
        px(img, [(23, 31 + b), (24, 31 + b)], o)


def aura(img, d, frame):
    """Paint a flame-like ki aura behind the sprite."""
    alpha = img[:, :, 3] > 0
    from scipy import ndimage  # noqa
    return img


def make_aura(img, d, frame):
    filled = img[:, :, 3] > 0
    # distance field via repeated dilation
    rng = np.random.default_rng(7 + frame)
    layers = []
    cur = filled.copy()
    for i in range(6):
        nxt = cur.copy()
        nxt[1:, :] |= cur[:-1, :]
        nxt[:-1, :] |= cur[1:, :]
        nxt[:, 1:] |= cur[:, :-1]
        nxt[:, :-1] |= cur[:, 1:]
        # flames lick upward: extra upward growth
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
                # vertical flame streaks that scroll upward each frame
                if ((y + frame * 3) // 2 + x) % 4 == 0:
                    continue
                c, a = cols[2], 170
            out[y, x] = (*c[:3], a)
    # tall flame tongues
    for x in range(4, W - 4, 4):
        top = np.nonzero(cur[:, x])[0]
        if len(top) == 0:
            continue
        t = top[0]
        h = 2 + (x * 3 + frame * 2) % 5
        for k in range(h):
            yy = t - k
            for xx in (x, x + 1):
                if 0 <= yy < H and out[yy, xx, 3] == 0 and not (k == h - 1 and xx == x + 1):
                    out[yy, xx] = (*cols[2][:3], 170)
    return out


def render(d, bob=0):
    s = build_body(d, bob)
    d["hair_fn"](s, d, bob)
    img = finish(s, d, bob)
    face(img, d, bob, brow="bushy" if d["key"] == "old_master" else "angry")
    extras(img, s, d, bob)
    return img


def upscale(arr, k):
    im = Image.fromarray(arr, "RGBA")
    return im.resize((im.width * k, im.height * k), Image.NEAREST)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pad = 8
    sheet_frames = []
    for d in DESIGNS:
        idle = render(d, 0)
        idle2 = render(d, 1)
        # expand canvas so the aura fits
        def padded(a):
            p = np.zeros((H + pad * 2, W + pad * 2, 4), dtype=np.uint8)
            p[pad:pad + H, pad:pad + W] = a
            return p
        idle_p, idle2_p = padded(idle), padded(idle2)
        global_H, global_W = H, W
        powered = [make_aura_padded(idle_p, d, f) for f in range(3)]
        Image.fromarray(idle, "RGBA").save(OUT / f"{d['key']}_idle.png")
        upscale(idle, 6).save(OUT / f"{d['key']}_idle_x6.png")
        frames = [upscale(idle_p, 6), upscale(idle2_p, 6)]
        frames[0].save(OUT / f"{d['key']}_idle_x6.gif", save_all=True,
                       append_images=frames[1:], duration=420, loop=0, disposal=2)
        pf = [upscale(p, 6) for p in powered]
        pf[0].save(OUT / f"{d['key']}_powerup_x6.gif", save_all=True,
                   append_images=pf[1:], duration=110, loop=0, disposal=2)
        upscale(powered[0], 6).save(OUT / f"{d['key']}_powerup_x6.png")
        sheet_frames.append((d, idle_p, powered[0]))
    build_sheet(sheet_frames)


def make_aura_padded(img, d, frame):
    global H, W
    oh, ow = H, W
    H, W = img.shape[0], img.shape[1]
    try:
        return make_aura(img, d, frame)
    finally:
        H, W = oh, ow


def build_sheet(items):
    k = 5
    ch, cw = items[0][1].shape[0] * k, items[0][1].shape[1] * k
    col_w = cw * 2 + 30
    sheet = Image.new("RGBA", (col_w * len(items) + 40, ch + 190), (24, 26, 44, 255))
    dr = ImageDraw.Draw(sheet)
    # simple dohyo floor
    dr.rectangle([0, 120 + ch - 40, sheet.width, sheet.height], fill=(196, 160, 110, 255))
    dr.rectangle([0, 120 + ch - 40, sheet.width, 120 + ch - 34], fill=(150, 112, 70, 255))
    try:
        f_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        f_t = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        f_s = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    except OSError:
        f_big = f_t = f_s = ImageFont.load_default()
    dr.text((24, 18), "SUMO SPRITE CONCEPTS  -  idle / power-up", font=f_big, fill=(255, 226, 120, 255))
    for i, (d, idle, pw) in enumerate(items):
        x0 = 20 + i * col_w
        dr.text((x0 + 10, 64), f"{i + 1}. {d['name']}", font=f_t, fill=(255, 255, 255, 255))
        sheet.alpha_composite(upscale(idle, k), (x0, 100))
        sheet.alpha_composite(upscale(pw, k), (x0 + cw + 10, 100))
        words, line, lines = d["blurb"].split(), "", []
        for w_ in words:
            if dr.textlength(line + " " + w_, font=f_s) > col_w - 20:
                lines.append(line); line = w_
            else:
                line = (line + " " + w_).strip()
        lines.append(line)
        for j, ln in enumerate(lines):
            dr.text((x0 + 10, 100 + ch + 10 + j * 24), ln, font=f_s, fill=(40, 24, 10, 255))
    sheet.save(OUT / "sumo_lineup.png")


if __name__ == "__main__":
    main()
