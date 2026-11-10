"""Shiko (the sumo leg-raise and stomp) for the Kaiju Yokozuna, front view.

Frames are 96x64 because the raised leg reaches far out to the side.
On the GBA each frame is drawn as two OBJs: a 64x64 on the left and a
32x64 on the right, sharing one 16-colour palette.

Outputs (sprites/shiko/):
  kaiju_yokozuna_shiko_sheet.png     native frames, one row
  kaiju_yokozuna_shiko_sheet_x4.png  labelled enlarged sheet
  kaiju_yokozuna_shiko_dust.png      4-frame dust cloud, 64x32 each
  kaiju_yokozuna_shiko_x4.gif        full shiko on the dohyo with dust and shake
  kaiju_yokozuna_shiko.c / .h        tiles, timing table

Run:  python sprites/make_shiko.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import make_rikishi_set as rs  # noqa: E402
import make_walk_cycle as wc   # noqa: E402  (palette + quantizer + GBA helpers)

FW, FH = 96, 64
rs.W, rs.H = FW, FH
rs.YY, rs.XX = np.mgrid[0:FH, 0:FW].astype(float)
CX = FW / 2 - 0.5
rs.CX = CX

OUT = Path(__file__).parent / "shiko"
D = wc.D
SK, MW, HR, O = D["skin"], D["mawashi"], D["hair"], D["outline"]
ell, cyl, Sprite = rs.ellipse, rs.cyl, rs.Sprite


def limb(s, a, b, thick, ramp, z, group):
    (x1, y1), (x2, y2) = a, b
    L = math.hypot(x2 - x1, y2 - y1)
    m, n = ell(CX + (x1 + x2) / 2, (y1 + y2) / 2, L / 2 + thick * 0.55, thick, rot=math.atan2(y2 - y1, x2 - x1))
    s.add(m, n, ramp, z, group=group)


# ---------------------------------------------------------------- poses
# x values are offsets from the frame centre; y values are absolute.
# Each leg: hip, knee, foot. Each arm: shoulder, hand.
def squat(drop=0, wide=0):
    return dict(
        torso=(0, 36 + drop), trx=17 + 0.5 * drop, try_=14.5 - 0.5 * drop, head=(0, 18 + drop),
        legL=((-9, 46 + drop), (-23 - wide, 49 + drop // 2), (-22 - wide, 59)),
        legR=((9, 46 + drop), (23 + wide, 49 + drop // 2), (22 + wide, 59)),
        armL=((-14, 28 + drop), (-23 - wide, 45 + drop // 2)),
        armR=((14, 28 + drop), (23 + wide, 45 + drop // 2)),
        mouth="flat", eyes="normal",
    )


POSES = {
    "ready": squat(0),
    "settle": squat(1, 1),
    "stomp": dict(squat(3, 2), mouth="shout", eyes="glare"),
    "shift": dict(
        torso=(5, 35), trx=17, try_=14.5, head=(6, 17),
        legL=((-4, 45), (-15, 46), (-21, 54)),
        legR=((14, 46), (25, 49), (23, 59)),
        armL=((-9, 27), (-15, 43)),
        armR=((19, 27), (25, 45)),
        mouth="flat", eyes="glare",
    ),
    "raise": dict(
        torso=(8, 35), trx=17, try_=14.5, head=(10, 18),
        legL=((-1, 43), (-13, 37), (-25, 30)),
        legR=((17, 46), (27, 49), (24, 59)),
        armL=((-6, 27), (-12, 38)),
        armR=((22, 28), (27, 45)),
        mouth="grit", eyes="glare",
    ),
    "peak": dict(
        torso=(9, 35), trx=17, try_=14.5, head=(12, 19),
        legL=((0, 42), (-14, 32), (-29, 22)),
        legR=((18, 46), (28, 49), (25, 59)),
        armL=((-5, 27), (-13, 34)),
        armR=((23, 28), (28, 45)),
        mouth="grit", eyes="glare",
    ),
}


def flip_pose(p):
    """Same pose with the other leg raised (redrawn, not mirrored, so the
    topknot and light direction stay consistent)."""
    f = lambda pts: tuple((-x, y) for x, y in pts)
    q = dict(p)
    q["torso"] = (-p["torso"][0], p["torso"][1])
    q["head"] = (-p["head"][0], p["head"][1])
    q["legL"], q["legR"] = f(p["legR"]), f(p["legL"])
    q["armL"], q["armR"] = f(p["armR"]), f(p["armL"])
    return q


def render(p):
    s = Sprite(O)
    tx, ty = p["torso"]
    for name in ("legL", "legR"):
        hip, knee, foot = p[name]
        limb(s, hip, knee, 6.4, SK, 1, name)
        limb(s, knee, foot, 5.6, SK, 1.1, name)
        out = -1 if foot[0] < 0 else 1
        raised = foot[1] < 56
        if raised:
            m, n = ell(CX + foot[0] + out * 1.5, foot[1] - 0.5, 2.4, 4.0, rot=0.5 * out)
        else:
            m, n = ell(CX + foot[0] + out * 1.5, foot[1] + 0.2, 5.0, 2.2)
        s.add(m, n, SK, 1.2, group=name)
    m, n = ell(CX + tx, ty, p["trx"], p["try_"])
    s.add(m, n, SK, 2, group="torso")
    top = ty + 6
    band = (rs.YY >= top) & (rs.YY <= top + 5) & ell(CX + tx, ty, p["trx"] + 1, p["try_"] + 1)[0]
    s.add(*cyl(band, p["trx"] + 1), MW, 3, hard=False, group="mawashi")
    rs_cx = rs.CX
    rs.CX = CX + tx
    front = (np.abs(rs.XX + 0.5 - (CX + tx)) <= 3.5) & (rs.YY >= top + 5) & (rs.YY <= top + 10)
    s.add(*cyl(front, 5), MW, 3, hard=False, group="mawashi")
    rs.CX = rs_cx
    for name in ("armL", "armR"):
        sh, hand = p[name]
        limb(s, sh, hand, 5.4, SK, 4, name)
        m, n = ell(CX + hand[0], hand[1], 4.2, 3.9)
        s.add(m, n, SK, 5, group=name)
    hx, hy = p["head"]
    m, n = ell(CX + hx, hy, 9.0, 8.5)
    s.add(m, n, SK, 6, group="head")
    hm, hn = ell(CX + hx, hy, 9.1, 8.6)
    side = hm & (np.abs(rs.XX + 0.5 - (CX + hx)) >= 7.6) & (rs.YY <= hy + 1)
    s.add((hm & (rs.YY <= hy - 4)) | side, hn, HR, 7, hard=False, group="head")
    m, n = ell(CX + hx, hy - 10, 3.6, 2.6)
    s.add(m, n, HR, 8, group="topknot")
    m, n = ell(CX + hx + 3.5, hy - 10.5, 3.0, 1.8, rot=-0.3)
    s.add(m, n, HR, 8, group="topknot")
    img = s.finish()

    # body details + tsuna follow the torso
    rs.OX, rs.OY = int(round(CX + tx - 23.5)), 0
    rs.px(img, [(23, ty + 4), (24, ty + 4)], SK[0])
    rs.px(img, rs.mirror([(17, ty - 8), (18, ty - 7), (19, ty - 7), (20, ty - 7), (21, ty - 8)]), SK[1])
    rs.gear(img, s, dict(D, gear=["tsuna"]), dict(rs.BODIES["classic"], tcy=ty, trx=p["trx"]))
    # face follows the head
    rs.OX, rs.OY = int(round(CX + hx - 23.5)), int(hy - 14)
    rs.face(img, dict(D, mouth=p["mouth"], eyes=p["eyes"]))
    rs.OX, rs.OY = 4, 0
    return img


# ---------------------------------------------------------------- dust
def dust_frames():
    """4 frames of a 64x32 dust cloud kicked out sideways from a stomp."""
    pal = ["#ffffff", "#cfc9bb", "#8a8478"]
    frames = []
    for f in range(4):
        a = np.zeros((32, 64, 4), dtype=np.uint8)
        spread, rise, r = 6 + f * 6, f * 2.2, 3.2 + f * 1.3
        puffs = [(-spread, 0, 1.0), (spread, 0, 1.0), (-spread * 0.45, -2 - rise, 1.1),
                 (spread * 0.45, -2 - rise, 1.1), (0, -4 - rise * 1.3, 0.9),
                 (-spread * 1.5, 2, 0.7), (spread * 1.5, 2, 0.7)]
        for dx, dy, sc in puffs:
            cx, cy, rr = 32 + dx, 26 + dy, r * sc
            for y in range(32):
                for x in range(64):
                    d = ((x + .5 - cx) / rr) ** 2 + ((y + .5 - cy) / (rr * .75)) ** 2
                    if d > 1:
                        continue
                    if f == 3 and (x + y) % 2:
                        continue
                    lit = (x + .5 - cx) < 0 and (y + .5 - cy) < 0
                    c = pal[0] if (lit and f < 3) else (pal[1] if d < .7 else pal[2])
                    if a[y, x, 3] == 0 or c == pal[0]:
                        a[y, x] = rs.hexc(c)
        frames.append(a)
    return frames


# ---------------------------------------------------------------- timeline (ticks at 60 fps)
TIMELINE = [
    ("ready", 24, None), ("shift", 8, None), ("raise", 8, None), ("peak", 36, None),
    ("stomp", 6, "L"), ("settle", 10, None), ("ready", 20, None),
    ("shift_r", 8, None), ("raise_r", 8, None), ("peak_r", 36, None),
    ("stomp", 6, "R"), ("settle", 10, None),
]
ORDER = ["ready", "settle", "stomp", "shift", "raise", "peak", "shift_r", "raise_r", "peak_r"]


def build():
    frames = {}
    for k in ("ready", "settle", "stomp", "shift", "raise", "peak"):
        frames[k] = POSES[k]
    for k in ("shift", "raise", "peak"):
        frames[k + "_r"] = flip_pose(POSES[k])
    import make_shiko_mockups as mk  # chosen leg-raise pose (R1 Textbook)
    textbook = mk.textbook_frames()
    rgba, idx = {}, {}
    for k in ORDER:
        src = textbook[k] if k in textbook else render(frames[k])
        r, i = wc.quantize(src)
        rgba[k], idx[k] = r, i
    return rgba, idx


def up(a, k):
    im = Image.fromarray(np.ascontiguousarray(a), "RGBA")
    return im.resize((im.width * k, im.height * k), Image.NEAREST)


def write_sheets(rgba, dust):
    sheet = np.zeros((FH, FW * len(ORDER), 4), dtype=np.uint8)
    for i, k in enumerate(ORDER):
        sheet[:, i * FW:(i + 1) * FW] = rgba[k]
    Image.fromarray(sheet, "RGBA").save(OUT / "kaiju_yokozuna_shiko_sheet.png")
    dsheet = np.zeros((32, 64 * 4, 4), dtype=np.uint8)
    for i, d in enumerate(dust):
        dsheet[:, i * 64:(i + 1) * 64] = d
    Image.fromarray(dsheet, "RGBA").save(OUT / "kaiju_yokozuna_shiko_dust.png")

    k = 3
    cols = 3
    labels = {"ready": "ready", "settle": "settle", "stomp": "stomp", "shift": "shift weight",
              "raise": "raise", "peak": "peak (hold)", "shift_r": "shift (other side)",
              "raise_r": "raise (other side)", "peak_r": "peak (other side)"}
    cw, ch = FW * k + 20, FH * k + 50
    big = Image.new("RGBA", (cols * cw + 20, 3 * ch + 20), (24, 26, 44, 255))
    dr = ImageDraw.Draw(big)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    seq = ["ready", "shift", "raise", "peak", "stomp", "settle", "shift_r", "raise_r", "peak_r"]
    for i, key in enumerate(seq):
        x0, y0 = 10 + (i % cols) * cw, 10 + (i // cols) * ch
        dr.rounded_rectangle([x0, y0, x0 + cw - 10, y0 + ch - 10], 10, fill=(36, 40, 66, 255))
        big.alpha_composite(up(rgba[key], k), (x0 + 5, y0 + 5))
        dr.text((x0 + 12, y0 + FH * k + 10), f"{i + 1}. {labels[key]}", font=f, fill=(255, 226, 120, 255))
    big.save(OUT / "kaiju_yokozuna_shiko_sheet_x4.png")


def write_gif(rgba, dust):
    bg = wc.dohyo_bg()
    bx, by = 72, 40
    gif = []
    dust_t, dust_x, shake_t = 99, 0, 99
    for key, ticks, stomp in TIMELINE * 2:
        for t in range(ticks):
            if stomp and t == 0:
                dust_t, shake_t = 0, 0
                dust_x = bx + FW // 2 + (-24 if stomp == "L" else 24) - 32
            if t % 2 == 0:
                im = bg.copy().convert("RGBA")
                shadow = Image.new("RGBA", im.size, (0, 0, 0, 0))
                ImageDraw.Draw(shadow).ellipse([bx + 18, by + 56, bx + 78, by + 66], fill=(90, 60, 30, 110))
                im.alpha_composite(shadow)
                im.alpha_composite(Image.fromarray(rgba[key], "RGBA"), (bx, by))
                if dust_t < 12:
                    im.alpha_composite(Image.fromarray(dust[dust_t // 3], "RGBA"), (dust_x, by + 36))
                if shake_t < 6:
                    dy = [2, -2, 1][shake_t // 2]
                    shifted = Image.new("RGBA", im.size, (0, 0, 0, 255))
                    shifted.paste(im, (0, dy))
                    im = shifted
                gif.append(im.convert("RGB").resize((720, 480), Image.NEAREST))
            dust_t += 1
            shake_t += 1
    gif[0].save(OUT / "kaiju_yokozuna_shiko_x4.gif", save_all=True, append_images=gif[1:], duration=33, loop=0)


def write_gba(idx, dust):
    def tiles(ix, x0, w):
        part = ix[:, x0:x0 + w]
        data = bytearray()
        for ty in range(FH // 8):
            for tx in range(w // 8):
                for row in part[ty * 8:(ty + 1) * 8, tx * 8:(tx + 1) * 8]:
                    for i in range(0, 8, 2):
                        data.append(int(row[i]) | (int(row[i + 1]) << 4))
        return data

    def words(raw):
        return [int.from_bytes(raw[i:i + 4], "little") for i in range(0, len(raw), 4)]

    lines = ['#include "kaiju_yokozuna_shiko.h"', "",
             "// Uses kaijuYokozunaPal from kaiju_yokozuna_walk.c (same 16 colours).",
             "// Each frame: 64 tiles for the left 64x64 OBJ, then 32 tiles for the right 32x64 OBJ.",
             "// Frame order: " + ", ".join(ORDER),
             "const unsigned int kaijuShikoTiles[SHIKO_FRAME_COUNT * SHIKO_FRAME_WORDS] __attribute__((aligned(4))) = {"]
    for k in ORDER:
        raw = tiles(idx[k], 0, 64) + tiles(idx[k], 64, 32)
        w = words(raw)
        lines.append(f"    // {k}")
        for i in range(0, len(w), 8):
            lines.append("    " + ", ".join(f"0x{v:08X}" for v in w[i:i + 8]) + ",")
    lines.append("};")
    lines += ["", "// Dust cloud: 4 frames of a 64x32 OBJ (32 tiles each), same palette.",
              "const unsigned int kaijuShikoDustTiles[4 * 256] __attribute__((aligned(4))) = {"]
    for d in dust:
        _, di = wc.quantize(d)
        raw = bytearray()
        for ty in range(4):
            for tx in range(8):
                for row in di[ty * 8:(ty + 1) * 8, tx * 8:(tx + 1) * 8]:
                    for i in range(0, 8, 2):
                        raw.append(int(row[i]) | (int(row[i + 1]) << 4))
        w = words(raw)
        for i in range(0, len(w), 8):
            lines.append("    " + ", ".join(f"0x{v:08X}" for v in w[i:i + 8]) + ",")
    lines.append("};")
    names = {k: i for i, k in enumerate(ORDER)}
    lines += ["", "// Timeline: {frame index, ticks to hold, stomp side (0 none, 1 left foot, 2 right foot)}",
              "const ShikoStep kaijuShikoTimeline[SHIKO_STEPS] = {"]
    for key, ticks, st in TIMELINE:
        lines.append(f"    {{{names[key]}, {ticks}, {0 if st is None else (1 if st == 'L' else 2)}}},  // {key}")
    lines += ["};", ""]
    (OUT / "kaiju_yokozuna_shiko.c").write_text("\n".join(lines))
    (OUT / "kaiju_yokozuna_shiko.h").write_text(f"""#ifndef KAIJU_YOKOZUNA_SHIKO_H
#define KAIJU_YOKOZUNA_SHIKO_H

// Shiko (leg raise + stomp), front view. 96x64 frames drawn as two OBJs:
//   left  64x64 (square, size 3) at x
//   right 32x64 (tall,   size 3) at x + 64
// On a stomp step: shake the screen 2px for 6 ticks (BG scroll + OBJ y) and
// spawn the 64x32 dust OBJ centred on the planted foot (at frame x - 8 for the
// left foot, x + 40 for the right, y + 36), advancing its 4 frames every 3 ticks.

#define SHIKO_FRAME_COUNT {len(ORDER)}
#define SHIKO_FRAME_WORDS (96 * 8)   // 64 + 32 tiles, 8 words each
#define SHIKO_STEPS {len(TIMELINE)}

typedef struct {{ unsigned char frame, ticks, stomp; }} ShikoStep;

extern const unsigned int kaijuShikoTiles[SHIKO_FRAME_COUNT * SHIKO_FRAME_WORDS];
extern const unsigned int kaijuShikoDustTiles[4 * 256];
extern const ShikoStep    kaijuShikoTimeline[SHIKO_STEPS];

#endif
""")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rgba, idx = build()
    dust = [wc.quantize(d)[0] for d in dust_frames()]
    write_sheets(rgba, dust)
    write_gif(rgba, dust)
    write_gba(idx, dust)
    print("frames:", len(ORDER), "timeline ticks:", sum(t for _, t, _ in TIMELINE))


if __name__ == "__main__":
    main()
