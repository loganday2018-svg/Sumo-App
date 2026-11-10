"""4-direction walk cycle for the Kaiju Yokozuna, ready for the GBA.

Outputs (sprites/walk/):
  kaiju_yokozuna_walk_sheet.png      native 64x64 frames, rows = down/up/left/right
  kaiju_yokozuna_walk_sheet_x4.png   enlarged preview of the same sheet
  kaiju_yokozuna_walk_x4.gif         all four directions looping side by side
  kaiju_yokozuna_demo.gif            walking a square on a 240x160 GBA-sized screen
  kaiju_yokozuna_walk.c / .h         4bpp tiles, BGR555 palette and anim tables

Run:  python sprites/make_walk_cycle.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
import make_rikishi_set as rs  # noqa: E402

OUT = Path(__file__).parent / "walk"
D = next(d for d in rs.DESIGNS if d["key"] == "kaiju_yokozuna")
W, H, CX = rs.W, rs.H, rs.CX
XX, YY = rs.XX, rs.YY
ell, cyl, Sprite = rs.ellipse, rs.cyl, rs.Sprite
SK, MW, HR = D["skin"], D["mawashi"], D["hair"]
FAR = [SK[0], SK[0], SK[1], SK[2]]          # darker ramp for far-side limbs
O = D["outline"]

# ---------------------------------------------------------------- poses
# Cycle is stand -> step A -> stand -> step B. Each frame: body bob (+ = down),
# sideways sway, per-leg offsets and arm swing. Head trails the body by a frame.
FRONT_POSES = {
    "stand": dict(bob=0, sway=0, l=(0, 0), r=(0, 0), arm=0, head=1, squash=0),
    "stepA": dict(bob=1, sway=-1, l=(0, 1), r=(0, -2), arm=1, head=0, squash=1),
    "stepB": dict(bob=1, sway=1, l=(0, -2), r=(0, 1), arm=-1, head=0, squash=1),
}
SIDE_POSES = {
    "stand": dict(bob=0, near=-1, far=2, nlift=0, flift=0, arm=0, head=1, squash=0),
    "stepA": dict(bob=1, near=-6, far=5, nlift=0, flift=-1, arm=2, head=0, squash=1),
    "stepB": dict(bob=1, near=5, far=-5, nlift=-1, flift=0, arm=-2, head=0, squash=1),
}
CYCLE = ["stand", "stepA", "stand", "stepB"]


def shade(text_img, pts, color):
    for x, y in pts:
        if 0 <= x < W and 0 <= y < H:
            text_img[y, x] = rs.hexc(color)


# ---------------------------------------------------------------- front / back
def front_back(pose, back=False):
    p = pose
    bob, sw = p["bob"], p["sway"]
    s = Sprite(O)
    cx = CX + sw
    for side, (dx, dy) in ((-1, p["l"]), (1, p["r"])):
        lift = dy < 0
        m, n = ell(CX + side * 8 + dx, 52 + dy, 6.2, 7.5 - (0.8 if lift else 0))
        s.add(m, n, SK, 1, group=f"leg{side}")
    trx, tr_y = 16.5 + 0.5 * p["squash"], 15.5 - 0.5 * p["squash"]
    tcy = 33 + bob
    m, n = ell(cx, tcy, trx, tr_y)
    s.add(m, n, SK, 2, group="torso")
    top = tcy + 6
    band = (YY >= top) & (YY <= top + 5) & ell(cx, tcy, trx + 1, tr_y + 1)[0]
    s.add(*cyl(band, trx + 1), MW, 3, hard=False, group="mawashi")
    strip = (np.abs(XX + 0.5 - cx) <= (2.5 if back else 3.5)) & (YY >= top + 5) & (YY <= 51)
    s.add(*cyl(strip, 5), MW, 3, hard=False, group="mawashi")
    if back:
        # musubi: the folded knot at the back of the mawashi
        knot = (np.abs(XX + 0.5 - cx) <= 4.5) & (YY >= top - 1) & (YY <= top + 6)
        s.add(*cyl(knot, 6), MW, 3.2, hard=True, group="knot_m")
    for side in (-1, 1):
        ay = tcy - 2 + side * p["arm"] * (1 if not back else -1)
        m, n = ell(cx + side * 16, ay, 5.5, 10.5, rot=side * -0.22)
        s.add(m, n, SK, 4, group=f"arm{side}")
        m, n = ell(cx + side * 18, ay + 10, 4.3, 4.0)
        s.add(m, n, SK, 5, group=f"arm{side}")
    hy = head_y(p)
    m, n = ell(cx, hy, 9.0, 8.5)
    s.add(m, n, SK, 6, group="head")
    if back:
        hm, hn = ell(cx, hy, 9.1, 8.6)
        s.add(hm & (YY <= hy + 4), hn, HR, 7, hard=False, group="head")
    else:
        hm, hn = ell(cx, hy, 9.1, 8.6)
        side_hair = hm & (np.abs(XX + 0.5 - cx) >= 7.6) & (YY <= hy + 1)
        s.add((hm & (YY <= hy - 4)) | side_hair, hn, HR, 7, hard=False, group="head")
    m, n = ell(cx, hy - 10, 3.6, 2.6)
    s.add(m, n, HR, 8, group="topknot")
    m, n = ell(cx + 3.5, hy - 10.5, 3.0, 1.8, rot=-0.3)
    s.add(m, n, HR, 8, group="topknot")
    img = s.finish()

    sh = int(sw)
    rs.OX, rs.OY = 4 + sh, 0
    if not back:
        # belly button, pecs, knuckles, face, tsuna
        rs.px(img, [(23, tcy + 4), (24, tcy + 4)], SK[0])
        rs.px(img, rs.mirror([(17, tcy - 8), (18, tcy - 7), (19, tcy - 7), (20, tcy - 7), (21, tcy - 8)]), SK[1])
        rs.OY = int(hy - 14)
        rs.face(img, D)
        rs.OY = 0
        rs.gear(img, s, dict(D, gear=["tsuna"]), dict(rs.BODIES["classic"], tcy=tcy, trx=trx))
    else:
        # spine and shoulder blades
        rs.px(img, [(23, y) for y in range(tcy - 10, tcy + 3, 2)] + [(24, y) for y in range(tcy - 9, tcy + 4, 2)], SK[1])
        rs.px(img, rs.mirror([(15, tcy - 8), (16, tcy - 9), (17, tcy - 9), (18, tcy - 8)]), SK[1])
        back_tsuna(img, s, tcy, trx)
        # buttock crease either side of the strip
        rs.px(img, rs.mirror([(19, top + 6), (20, top + 7), (20, top + 8)]), SK[0])
    for side, (dx, dy) in ((-1, p["l"]), (1, p["r"])):
        lx = 23.5 + side * 8 + dx
        if not back:
            rs.px(img, [(int(lx) - 2 - sh, 59 + dy), (int(lx) - sh, 59 + dy)], SK[0])
    rs.OX, rs.OY = 4, 0
    return img


def head_y(p):
    """Head trails the body: on step frames it hasn't dropped yet, on the
    following stand frame it is still settling one pixel low."""
    return 14 + (1 if p["head"] == 1 else 0)


def back_tsuna(img, s, tcy, trx):
    rope = ["#8a8478", "#cfc9bb", "#ffffff"]
    y0 = tcy + 5
    for x in range(int(CX - trx - 2), int(CX + trx + 3)):
        if 0 <= x < W and s.group_at(x, y0) in ("torso", "mawashi", "knot_m"):
            img[y0 - 2, x] = rs.hexc(O)
            img[y0 - 1, x] = rs.hexc(rope[2 if (x // 2) % 2 else 1])
            img[y0, x] = rs.hexc(rope[1 if (x // 2) % 2 else 0])
            img[y0 + 1, x] = rs.hexc(O)
    # the bow of the yokozuna rope: two solid twisted loops and a centre knot
    hc = rs.hexc
    mask = np.zeros((H, W), bool)
    for sd in (-1, 1):
        m, _ = ell(CX + sd * 5.5, y0 - 4, 5.0, 2.9, rot=sd * -0.35)
        mask |= m
    m, _ = ell(CX, y0 - 3, 2.2, 2.6)
    mask |= m
    for y, x in zip(*np.nonzero(mask)):
        img[y, x] = hc(rope[2] if ((x + y) // 2) % 2 else rope[1])
        if abs(x + 0.5 - CX) <= 1.5:
            img[y, x] = hc(rope[0] if y % 2 else rope[1])
    for y in range(H):
        for x in range(W):
            if not mask[y, x] and any(0 <= y + dy < H and 0 <= x + dx < W and mask[y + dy, x + dx]
                                      for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                img[y, x] = hc(O)


# ---------------------------------------------------------------- side (facing left)
def side(pose):
    p = pose
    bob = p["bob"]
    s = Sprite(O)
    cx = CX + 1
    tcy = 33 + bob
    hy = head_y(dict(head=p["head"], bob=bob))
    # far leg + far arm (behind)
    m, n = ell(cx + p["far"], 52 + p["flift"], 5.8, 7.5 + p["flift"] * 0.6)
    s.add(m, n, FAR, 0.5, group="farleg")
    m, n = ell(cx + p["far"] - 2, 58.5 + p["flift"], 4.2, 1.8)
    s.add(m, n, FAR, 0.6, group="farleg")
    m, n = ell(cx + 3 - p["arm"], tcy - 2, 5.2, 10.0, rot=0.15 + p["arm"] * 0.06)
    s.add(m, n, FAR, 0.7, group="fararm")
    # torso with forward belly
    m1, n1 = ell(cx + 2, tcy, 11.5, 15.5)
    s.add(m1, n1, SK, 2, group="torso")
    m2, n2 = ell(cx - 5.5, tcy + 3 + p["squash"], 11.5, 11.5 - 0.5 * p["squash"])
    s.add(m2 & ~m1, n2, SK, 2, group="torso")
    body = m1 | m2
    top = tcy + 6
    band = (YY >= top) & (YY <= top + 5) & (ell(cx + 2, tcy, 12.5, 16.5)[0] | ell(cx - 5.5, tcy + 3, 12.5, 12.5)[0])
    s.add(*cyl(band, 14), MW, 3, hard=False, group="mawashi")
    front = (XX + 0.5 >= cx - 8) & (XX + 0.5 <= cx - 3) & (YY >= top + 5) & (YY <= 50)
    s.add(*cyl(front, 6), MW, 3, hard=False, group="mawashi")
    knot = (XX + 0.5 >= cx + 11) & (XX + 0.5 <= cx + 15) & (YY >= top - 1) & (YY <= top + 6)
    s.add(*cyl(knot, 4), MW, 3.2, hard=True, group="knot_m")
    # near leg
    m, n = ell(cx + p["near"], 52 + p["nlift"], 6.2, 7.5 + p["nlift"] * 0.6)
    s.add(m, n, SK, 3.5, group="nearleg")
    m, n = ell(cx + p["near"] - 2.5, 58.5 + p["nlift"], 4.6, 2.0)
    s.add(m, n, SK, 3.6, group="nearleg")
    # near arm
    m, n = ell(cx + 1 + p["arm"], tcy - 2, 5.6, 10.5, rot=0.15 - p["arm"] * 0.07)
    s.add(m, n, SK, 4, group="arm")
    m, n = ell(cx + 1 + p["arm"] * 1.8 - 0.5, tcy + 8, 4.3, 4.0)
    s.add(m, n, SK, 5, group="arm")
    # head in profile
    hx = cx - 2
    m, n = ell(hx, hy, 8.8, 8.5)
    s.add(m, n, SK, 6, group="head")
    m, n = ell(hx - 8.6, hy + 2.5, 1.6, 1.4)          # nose
    s.add(m, n, SK, 6, group="head")
    hm, hn = ell(hx, hy, 8.9, 8.6)
    hair = hm & ((YY <= hy - 4) | (XX + 0.5 >= hx + 1.5)) & (YY <= hy + 5)
    s.add(hair, hn, HR, 7, hard=False, group="head")
    m, n = ell(hx + 1.5, hy + 1, 1.8, 2.3)            # ear
    s.add(m, n, SK, 8, group="ear")
    m, n = ell(hx - 1, hy - 10, 4.8, 2.2, rot=0.15)   # topknot lying forward
    s.add(m, n, HR, 8, group="topknot")
    m, n = ell(hx + 3, hy - 9, 2.0, 2.0)
    s.add(m, n, HR, 8, group="topknot")
    img = s.finish()

    hc = rs.hexc
    X, Y = int(hx), int(hy)
    # eye, brow, mouth
    for x, y, c in ((X - 6, Y - 3, O), (X - 5, Y - 3, O), (X - 4, Y - 2, O),
                    (X - 6, Y - 1, O), (X - 5, Y - 1, O), (X - 6, Y, "#ffffff"), (X - 5, Y, D.get("eye", "#1a1a2e")),
                    (X - 8, Y + 5, O), (X - 7, Y + 5, O), (X - 6, Y + 5, O),
                    (X - 6, Y + 2, D["blush"])):
        img[y, x] = hc(c)
    # tsuna across the side, shide at the front
    rope = ["#8a8478", "#cfc9bb", "#ffffff"]
    y0 = top - 1
    for x in range(W):
        if s.group_at(x, y0) in ("torso", "mawashi", "knot_m"):
            img[y0 - 2, x] = hc(O)
            img[y0 - 1, x] = hc(rope[2 if (x // 2) % 2 else 1])
            img[y0, x] = hc(rope[1 if (x // 2) % 2 else 0])
            img[y0 + 1, x] = hc(rope[0])
            img[y0 + 2, x] = hc(O)
    tab = ["WW.", ".WW", "WW.", ".WW", "WW."]
    for sx in (int(cx) - 15, int(cx) - 7):
        for ry, row in enumerate(tab):
            for rx_, ch in enumerate(row):
                if ch == "W":
                    img[y0 + 3 + ry, sx + rx_] = hc("#ffffff" if rx_ < 2 else "#c9c4b8")
            f, l = row.index("W"), row.rindex("W")
            img[y0 + 3 + ry, sx + f - 1] = hc(O)
            img[y0 + 3 + ry, sx + l + 1] = hc(O)
    # belly button, toe line
    img[tcy + 5, int(cx) - 16] = hc(SK[0])
    img[59 + p["nlift"], int(cx + p["near"]) - 5] = hc(SK[0])
    return img


# ---------------------------------------------------------------- palette (GBA: 15 colours + transparent)
PALETTE = [
    O, *SK,                                   # outline + 4 skin
    MW[1], MW[2], MW[3],                      # 3 mawashi
    HR[1], HR[2], HR[3],                      # 3 hair
    "#ffffff", "#cfc9bb", "#8a8478",          # white, rope mid, rope dark
    D["blush"],
]
PAL = np.array([rs.hexc(c)[:3] for c in PALETTE], dtype=int)


def quantize(img):
    """Map every opaque pixel to the nearest of the 15 palette colours.
    Returns (rgba image, index image) where index 0 is transparent."""
    out = np.zeros_like(img)
    idx = np.zeros(img.shape[:2], dtype=np.uint8)
    op = img[:, :, 3] > 0
    rgb = img[:, :, :3].astype(int)
    d = ((rgb[:, :, None, :] - PAL[None, None, :, :]) ** 2).sum(-1)
    near = d.argmin(-1)
    idx[op] = near[op] + 1
    out[op, :3] = PAL[near[op]]
    out[op, 3] = 255
    return out, idx


def to64(a):
    """Place a 56x62 sprite on a 64x64 GBA OBJ canvas, feet on the bottom row."""
    o = np.zeros((64, 64) + a.shape[2:], dtype=a.dtype)
    o[1:1 + H, 4:4 + W] = a
    return o


# ---------------------------------------------------------------- build everything
def build_frames():
    uniq = {}
    for name, pose in FRONT_POSES.items():
        uniq[f"down_{name}"] = front_back(pose)
        uniq[f"up_{name}"] = front_back(pose, back=True)
    for name, pose in SIDE_POSES.items():
        uniq[f"left_{name}"] = side(pose)
    frames, indexed = {}, {}
    for k, v in uniq.items():
        rgba, ix = quantize(v)
        frames[k], indexed[k] = to64(rgba), to64(ix)
    return frames, indexed


def anim(frames, direction):
    src = "left" if direction == "right" else direction
    seq = [frames[f"{src}_{n}"] for n in CYCLE]
    if direction == "right":
        seq = [f[:, ::-1] for f in seq]
    return seq


def up(a, k):
    im = Image.fromarray(np.ascontiguousarray(a), "RGBA")
    return im.resize((im.width * k, im.height * k), Image.NEAREST)


def write_sheet(frames):
    dirs = ["down", "up", "left", "right"]
    sheet = np.zeros((64 * 4, 64 * 4, 4), dtype=np.uint8)
    for r, dname in enumerate(dirs):
        for c, f in enumerate(anim(frames, dname)):
            sheet[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64] = f
    Image.fromarray(sheet, "RGBA").save(OUT / "kaiju_yokozuna_walk_sheet.png")
    k = 4
    big = Image.new("RGBA", (64 * 4 * k + 150, 64 * 4 * k + 60), (24, 26, 44, 255))
    dr = ImageDraw.Draw(big)
    from PIL import ImageFont
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    for c, n in enumerate(["stand", "step A", "stand", "step B"]):
        dr.text((150 + c * 64 * k + 90, 18), n, font=fs, fill=(190, 196, 220, 255))
    for r, dname in enumerate(dirs):
        dr.text((20, 60 + r * 64 * k + 110), dname.upper(), font=f, fill=(255, 226, 120, 255))
        for c in range(4):
            x0, y0 = 150 + c * 64 * k, 50 + r * 64 * k
            dr.rectangle([x0 + 2, y0 + 2, x0 + 64 * k - 3, y0 + 64 * k - 3], fill=(36, 40, 66, 255))
    big.alpha_composite(up(sheet, k), (150, 50))
    big.save(OUT / "kaiju_yokozuna_walk_sheet_x4.png")


def write_loop_gif(frames):
    k = 4
    dirs = ["down", "left", "up", "right"]
    seqs = [anim(frames, d) for d in dirs]
    out = []
    for i in range(4):
        canvas = Image.new("RGBA", (64 * k * 4 + 50, 64 * k + 20), (24, 26, 44, 255))
        for j, sq in enumerate(seqs):
            canvas.alpha_composite(up(sq[i], k), (10 + j * (64 * k + 10), 10))
        out.append(canvas.convert("RGB"))
    # 8 ticks per frame at 60 fps ~= 133 ms
    out[0].save(OUT / "kaiju_yokozuna_walk_x4.gif", save_all=True, append_images=out[1:],
                duration=133, loop=0)


def dohyo_bg():
    bg = Image.new("RGB", (240, 160), (196, 160, 110))
    d = ImageDraw.Draw(bg)
    for y in range(0, 160, 4):
        for x in range((y * 7) % 9, 240, 9):
            d.point((x, y), fill=(184, 148, 100))
    d.ellipse([20, 6, 220, 154], outline=(232, 214, 160), width=4)
    d.ellipse([22, 8, 218, 152], outline=(150, 120, 70), width=1)
    d.line([(102, 74), (114, 74)], fill=(250, 250, 250), width=2)
    d.line([(126, 86), (138, 86)], fill=(250, 250, 250), width=2)
    return bg


def write_demo(frames):
    """Walk a square on a GBA-sized screen: 1 px per tick, anim frame every 8 ticks."""
    bg = dohyo_bg()
    path = [("down", 0, 1, 48), ("right", 1, 0, 64), ("up", 0, -1, 48), ("left", -1, 0, 64)]
    x, y = 88, 28
    tick, gif = 0, []
    seqs = {d: anim(frames, d) for d in ("down", "up", "left", "right")}
    for dname, dx, dy, steps in path:
        # pause a few ticks facing the new direction (turn-in-place)
        for t in range(6):
            if t % 2 == 0:
                gif.append(compose(bg, seqs[dname][0], x, y))
        for s_ in range(steps):
            x += dx; y += dy; tick += 1
            if s_ % 2 == 0:
                f = seqs[dname][(s_ // 8 + 1) % 4]
                gif.append(compose(bg, f, x, y))
    gif = [g.resize((720, 480), Image.NEAREST) for g in gif]
    gif[0].save(OUT / "kaiju_yokozuna_demo.gif", save_all=True, append_images=gif[1:],
                duration=33, loop=0)


def compose(bg, frame, x, y):
    im = bg.copy().convert("RGBA")
    sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse([x + 12, y + 56, x + 52, y + 66], fill=(90, 60, 30, 110))
    im.alpha_composite(sh)
    im.alpha_composite(Image.fromarray(np.ascontiguousarray(frame), "RGBA"), (x, y))
    return im.convert("RGB")


# ---------------------------------------------------------------- GBA export
def bgr555(c):
    r, g, b = (int(v) >> 3 for v in rs.hexc(c)[:3])
    return r | (g << 5) | (b << 10)


def tiles_4bpp(ix):
    """64x64 indexed frame -> 64 tiles in 1D mapping order, 32 bytes each."""
    data = bytearray()
    for ty in range(8):
        for tx in range(8):
            t = ix[ty * 8:(ty + 1) * 8, tx * 8:(tx + 1) * 8]
            for row in t:
                for i in range(0, 8, 2):
                    data.append(int(row[i]) | (int(row[i + 1]) << 4))
    return bytes(data)


def write_gba(indexed):
    order = [f"{d}_{n}" for d in ("down", "up", "left") for n in ("stand", "stepA", "stepB")]
    pal = [0x0000] + [bgr555(c) for c in PALETTE]
    c_lines = ['#include "kaiju_yokozuna_walk.h"', "",
               "// 16-colour palette, BGR555. Index 0 is transparent.",
               "const unsigned short kaijuYokozunaPal[16] __attribute__((aligned(4))) = {",
               "    " + ", ".join(f"0x{v:04X}" for v in pal), "};", "",
               "// 9 frames x 64 tiles x 32 bytes, 4bpp, 1D OBJ mapping. Frame order:",
               "// " + ", ".join(order),
               "const unsigned int kaijuYokozunaTiles[KAIJU_FRAME_COUNT * KAIJU_FRAME_WORDS] __attribute__((aligned(4))) = {"]
    for k in order:
        raw = tiles_4bpp(indexed[k])
        words = [int.from_bytes(raw[i:i + 4], "little") for i in range(0, len(raw), 4)]
        c_lines.append(f"    // {k}")
        for i in range(0, len(words), 8):
            c_lines.append("    " + ", ".join(f"0x{w:08X}" for w in words[i:i + 8]) + ",")
    c_lines += ["};", "",
                "// Walk cycle per direction: stand, step A, stand, step B.",
                "// Right reuses the left frames with the OBJ horizontal-flip bit.",
                "const unsigned char kaijuWalkAnim[4][4] = {",
                "    {0, 1, 0, 2},   // DIR_DOWN",
                "    {3, 4, 3, 5},   // DIR_UP",
                "    {6, 7, 6, 8},   // DIR_LEFT",
                "    {6, 7, 6, 8},   // DIR_RIGHT (draw with hflip)",
                "};", ""]
    (OUT / "kaiju_yokozuna_walk.c").write_text("\n".join(c_lines))
    h = """#ifndef KAIJU_YOKOZUNA_WALK_H
#define KAIJU_YOKOZUNA_WALK_H

// Kaiju Yokozuna walk cycle. 64x64 4bpp sprite (OBJ shape square, size 3).
// Each frame is 2 KB: stream the current frame into OBJ VRAM with DMA in
// VBlank instead of keeping all frames resident.

#define KAIJU_FRAME_COUNT   9
#define KAIJU_FRAME_TILES   64
#define KAIJU_FRAME_WORDS   (KAIJU_FRAME_TILES * 8)   // 512 u32 = 2048 bytes
#define KAIJU_TICKS_PER_ANIM_FRAME 8                  // 60 fps / 8 = 7.5 anim fps
#define KAIJU_WALK_SPEED_FX 0x100                     // 1.0 px per tick in 24.8 fixed point

enum { DIR_DOWN, DIR_UP, DIR_LEFT, DIR_RIGHT };

extern const unsigned short kaijuYokozunaPal[16];
extern const unsigned int   kaijuYokozunaTiles[KAIJU_FRAME_COUNT * KAIJU_FRAME_WORDS];
extern const unsigned char  kaijuWalkAnim[4][4];

#endif
"""
    (OUT / "kaiju_yokozuna_walk.h").write_text(h)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames, indexed = build_frames()
    used = sorted({int(v) for ix in indexed.values() for v in np.unique(ix) if v})
    print("palette entries used:", len(used), "of 15")
    write_sheet(frames)
    write_loop_gif(frames)
    write_demo(frames)
    write_gba(indexed)


if __name__ == "__main__":
    main()
