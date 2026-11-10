"""Leg-raise mock-ups for the shiko: five ways to lean away from the kick.

Each variant tilts the torso toward the planted leg, shifts the hips over
the planted foot, and raises the other leg along a thigh/shin angle.
Output: sprites/shiko/mockups/
Run:  python sprites/make_shiko_mockups.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import make_shiko as ms              # noqa: E402  (sets the 96x64 canvas on rs)
import make_rikishi_set as rs        # noqa: E402
import make_walk_cycle as wc         # noqa: E402

FW, FH, CX = ms.FW, ms.FH, ms.CX
D, SK, MW, HR, O = ms.D, ms.SK, ms.MW, ms.HR, ms.O
ell, Sprite = rs.ellipse, rs.Sprite
OUT = Path(__file__).parent / "shiko" / "mockups"
THIGH, SHIN = 16.5, 14.5


def T(p, lx, ly):
    """Point in the torso's tilted frame -> frame offset coords."""
    t = math.radians(p["tilt"])
    tx, ty = p["torso"]
    return (tx + lx * math.cos(t) - ly * math.sin(t), ty + lx * math.sin(t) + ly * math.cos(t))


def along(pt, ang, length):
    a = math.radians(ang)
    return (pt[0] - math.cos(a) * length, pt[1] - math.sin(a) * length)


def joints(p):
    """Resolve a pose into absolute joint positions (offsets from centre)."""
    if p.get("squat"):
        hipL, kneeL, footL = (-9, 46), (-23, 49), (-22, 59)
        hipR = (9, 46)
    else:
        hipL, hipR = T(p, -11, 10), T(p, 9, 11)
        kneeL = along(hipL, p["a1"], THIGH)
        footL = along(kneeL, p["a2"], SHIN)
    kneeR, footR = p["kneeR"], p["footR"]
    if p.get("straight"):
        # standing leg locks: knee sits on the hip-foot line, a hair outward
        kneeR = ((hipR[0] + footR[0]) / 2 + 1.5, (hipR[1] + footR[1]) / 2)
    shL, shR = T(p, -13, -9), T(p, 13, -9)
    hands = {}
    for side, mode in (("L", p["handL"]), ("R", p["handR"])):
        sh = shL if side == "L" else shR
        if mode == "knee":
            k = kneeL if side == "L" else kneeR
            hands[side] = (k[0] + (2 if side == "L" else -1), k[1] - 3)
        elif mode == "thigh":
            h, k = (hipL, kneeL) if side == "L" else (hipR, kneeR)
            hands[side] = (h[0] + (k[0] - h[0]) * 0.6 + 1, h[1] + (k[1] - h[1]) * 0.6 - 3)
        elif mode == "up":
            hands[side] = (sh[0] - 3, sh[1] - 15)
        elif mode == "hip":
            hands[side] = T(p, 15, 6)
    head = T(p, 0, -18)
    head = (head[0] + p.get("head_dx", 0), head[1] + p.get("head_dy", 0))
    return dict(hipL=hipL, hipR=hipR, kneeL=kneeL, footL=footL, kneeR=kneeR, footR=footR,
                shL=shL, shR=shR, handL=hands["L"], handR=hands["R"], head=head)


READY = dict(squat=True, tilt=0, torso=(0, 36), trx=17, try_=14.5, a1=-13, a2=-92, kneeR=(23, 49), footR=(22, 59),
             handL="knee", handR="knee", mouth="flat", eyes="normal")


def lerp_joints(a, b, k):
    return {key: (a[key][0] + (b[key][0] - a[key][0]) * k, a[key][1] + (b[key][1] - a[key][1]) * k) for key in a}


def lerp_pose(a, b, k):
    p = dict(b)
    for key in ("tilt", "trx", "try_", "a1", "a2", "head_dx", "head_dy"):
        p[key] = a.get(key, 0) + (b.get(key, 0) - a.get(key, 0)) * k
    for key in ("torso", "kneeR", "footR"):
        p[key] = (a[key][0] + (b[key][0] - a[key][0]) * k, a[key][1] + (b[key][1] - a[key][1]) * k)
    return p


def render(p, j=None):
    j = j or joints(p)
    s = Sprite(O)
    tx, ty = p["torso"]
    t = math.radians(p["tilt"])
    # legs (behind the body)
    for leg, (hip, knee, foot) in (("legL", (j["hipL"], j["kneeL"], j["footL"])),
                                   ("legR", (j["hipR"], j["kneeR"], j["footR"]))):
        raised = leg == "legL" and foot[1] < 55
        z = 3.5 if raised else 1
        ms.limb(s, hip, knee, 7.6 if raised else 6.6, SK, z, leg)
        ms.limb(s, knee, foot, 6.3 if raised else 5.6, SK, z + 0.1, leg)
        out = -1 if foot[0] < tx else 1
        if foot[1] < 56:
            dx, dy = foot[0] - knee[0], foot[1] - knee[1]
            m, n = ell(CX + foot[0] + dx * 0.1, foot[1] + dy * 0.1, 5.6, 2.6, rot=math.atan2(dy, dx) + math.pi / 2)
            s.add(m, n, [SK[0], SK[0], SK[1], SK[1]], z + 0.2, group=leg)
            continue
        else:
            m, n = ell(CX + foot[0] + out * 1.5, foot[1] + 0.2, 5.0, 2.2)
        s.add(m, n, SK, z + 0.2, group=leg)
    # torso, tilted
    m, n = ell(CX + tx, ty, p["trx"], p["try_"], rot=t)
    s.add(m, n, SK, 2, group="torso")
    dx, dy = rs.XX + 0.5 - (CX + tx), rs.YY + 0.5 - ty
    u = dx * math.cos(t) + dy * math.sin(t)
    v = -dx * math.sin(t) + dy * math.cos(t)
    big, bn = ell(CX + tx, ty, p["trx"] + 1, p["try_"] + 1, rot=t)
    band = big & (v >= 6) & (v <= 11)
    s.add(band, bn, MW, 3, hard=False, group="mawashi")
    front = (np.abs(u) <= 3.5) & (v >= 11) & (v <= 16)
    s.add(front, bn, MW, 3, hard=False, group="mawashi")
    # arms
    for side in ("L", "R"):
        ms.limb(s, j["sh" + side], j["hand" + side], 5.4, SK, 4, "arm" + side)
        hx, hy = j["hand" + side]
        m, n = ell(CX + hx, hy, 4.2, 3.9)
        s.add(m, n, SK, 5, group="arm" + side)
    # head stays upright
    hx, hy = j["head"]
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

    hc = rs.hexc
    # tsuna rope following the tilt
    rope = ["#8a8478", "#cfc9bb", "#ffffff"]
    for y in range(FH):
        for x in range(FW):
            if s.group_at(x, y) not in ("torso", "mawashi"):
                continue
            vv, uu = v[y, x], u[y, x]
            tw = int(math.floor(uu / 2)) % 2
            if 3 <= vv < 4 or 7 <= vv < 8:
                img[y, x] = hc(O)
            elif 4 <= vv < 5:
                img[y, x] = hc(rope[2] if tw else rope[1])
            elif 5 <= vv < 6:
                img[y, x] = hc(rope[1] if tw else rope[0])
            elif 6 <= vv < 7:
                img[y, x] = hc(rope[1] if tw else rope[0])
    tab = ["WW.", ".WW", "WW.", ".WW", "WW."]
    for lu in (-12.5, -4.5, 3.5, 11.5):
        ax, ay = T(p, lu, 8)
        sx, sy = int(round(CX + ax)) - 1, int(round(ay))
        for ry, row in enumerate(tab):
            for rx_, ch in enumerate(row):
                if ch == "W" and 0 <= sy + ry < FH:
                    img[sy + ry, sx + rx_] = hc("#ffffff" if rx_ < 2 else "#c9c4b8")
            f, l = row.index("W"), row.rindex("W")
            if 0 <= sy + ry < FH:
                img[sy + ry, sx + f - 1] = hc(O)
                img[sy + ry, sx + l + 1] = hc(O)
    # belly button + pec lines in the tilted frame
    for lx, ly, c in ((0, 4, SK[0]), (1, 4, SK[0]), (-6, -7, SK[1]), (-5, -6, SK[1]), (-4, -6, SK[1]),
                      (-3, -6, SK[1]), (-2, -7, SK[1]), (2, -7, SK[1]), (3, -6, SK[1]), (4, -6, SK[1]),
                      (5, -6, SK[1]), (6, -7, SK[1])):
        x, y = T(p, lx, ly)
        X, Y = int(round(CX + x)), int(round(y))
        if s.group_at(X, Y) == "torso":
            img[Y, X] = hc(c)
    rs.OX, rs.OY = int(round(CX + hx - 23.5)), int(round(hy - 14))
    rs.face(img, dict(D, mouth=p["mouth"], eyes=p["eyes"]))
    rs.OX, rs.OY = 4, 0
    return wc.quantize(img)[0]


VARIANTS = [
    ("A", "Lean & Lift", "Torso tilts 22°, hips over the planted foot, hand rides the raised thigh.",
     dict(tilt=22, torso=(9, 36), trx=17, try_=14.5, a1=34, a2=50, kneeR=(27, 49), footR=(24, 59),
          handL="thigh", handR="knee", mouth="grit", eyes="glare")),
    ("B", "Sky Kick", "Textbook yokozuna height. Deep 32° lean, foot above the head.",
     dict(tilt=32, torso=(12, 37), trx=17, try_=14.5, a1=58, a2=76, kneeR=(28, 50), footR=(25, 59),
          handL="thigh", handR="knee", mouth="grit", eyes="glare", head_dx=-1)),
    ("C", "Low & Heavy", "Small 14° lean, leg kicks out wide and low. Feels massive.",
     dict(tilt=14, torso=(7, 38), trx=17.5, try_=14, a1=16, a2=4, kneeR=(26, 50), footR=(23, 59),
          handL="thigh", handR="knee", mouth="grit", eyes="glare")),
    ("D", "Power Fist", "24° lean with the free fist thrown up, charging-up style.",
     dict(tilt=24, torso=(10, 36), trx=17, try_=14.5, a1=44, a2=50, kneeR=(27, 49), footR=(24, 59),
          handL="up", handR="knee", mouth="grit", eyes="glare")),
    ("E", "Hip Hinge", "28° lean, head pushed over the planted leg, other hand on hip.",
     dict(tilt=28, torso=(11, 37), trx=17, try_=14.5, a1=46, a2=60, kneeR=(27, 50), footR=(24, 59),
          handL="thigh", handR="hip", mouth="grit", eyes="glare", head_dx=3, head_dy=1)),
]

STOMP = dict(READY, torso=(0, 39), trx=18.5, try_=13, kneeR=(25, 50), footR=(24, 59), a1=-10, a2=-92,
             mouth="shout", eyes="glare")


def frames_for(peak):
    shift = lerp_pose(READY, dict(peak, a1=-5, a2=-75), 0.45)
    shift.pop("straight", None)
    shift["mouth"], shift["eyes"] = "flat", "glare"
    raise_ = lerp_pose(READY, peak, 0.72)
    raise_["handL"], raise_["handR"] = peak["handL"], peak["handR"]
    out = {}
    jr, jp = joints(READY), joints(peak)
    out["ready"] = render(READY)
    shift.pop("squat", None); raise_.pop("squat", None)
    js = joints(shift)
    js["footL"] = (js["footL"][0], min(js["footL"][1], 57))
    out["shift"] = render(shift, lerp_joints(jr, js, 1.0))
    out["raise"] = render(raise_)
    out["peak"] = render(peak)
    out["stomp"] = render(STOMP)
    return out


def up(a, k):
    im = Image.fromarray(np.ascontiguousarray(a), "RGBA")
    return im.resize((im.width * k, im.height * k), Image.NEAREST)


VARIANTS_V2 = [
    ("R1", "Textbook", "Standing leg locks straight, raised leg straight at 50°, hand on the raised knee.",
     dict(straight=True, tilt=28, torso=(11, 30), trx=17, try_=14.5, a1=48, a2=50, kneeR=(24, 50), footR=(22, 59),
          handL="knee", handR="thigh", mouth="grit", eyes="glare")),
    ("R2", "Yokozuna High", "Deepest lean with the leg near vertical, like the top champions.",
     dict(straight=True, tilt=36, torso=(13, 31), trx=17, try_=14.5, a1=62, a2=66, kneeR=(24, 50), footR=(22, 59),
          handL="thigh", handR="thigh", mouth="grit", eyes="glare")),
    ("R3", "Steady", "Gentler 20° lean, leg straight out at shoulder height.",
     dict(straight=True, tilt=20, torso=(9, 31), trx=17, try_=14.5, a1=34, a2=36, kneeR=(24, 50), footR=(22, 59),
          handL="knee", handR="thigh", mouth="grit", eyes="glare")),
    ("R4", "Power Fist", "Textbook lean and leg, with the free fist punched up for a charge-up feel.",
     dict(straight=True, tilt=28, torso=(11, 30), trx=17, try_=14.5, a1=48, a2=50, kneeR=(24, 50), footR=(22, 59),
          handL="up", handR="thigh", mouth="grit", eyes="glare")),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    build_set(VARIANTS, "leg_raise_mockups", "LEG RAISE MOCK-UPS - pick a letter")
    build_set(VARIANTS_V2, "leg_raise_mockups_v2", "LEG RAISE v2 - straight standing leg, straight kick")


def build_set(variants, stem, heading):
    all_frames = []
    for code, name, blurb, peak in variants:
        fr = frames_for(peak)
        all_frames.append((code, name, blurb, fr))
        Image.fromarray(fr["peak"], "RGBA").save(OUT / f"peak_{code}.png")

    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    reg = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    f_h, f_t, f_s = ImageFont.truetype(bold, 28), ImageFont.truetype(bold, 22), ImageFont.truetype(reg, 16)

    # sheet: big peak pose + small shift/raise strip per variant
    k = 3
    cw, chh = FW * k + 30, FH * k + 200
    sheet = Image.new("RGBA", (cw * len(all_frames) + 30, chh + 80), (24, 26, 44, 255))
    dr = ImageDraw.Draw(sheet)
    dr.text((24, 20), heading, font=f_h, fill=(255, 226, 120, 255))
    for i, (code, name, blurb, fr) in enumerate(all_frames):
        x0, y0 = 15 + i * cw, 70
        dr.rounded_rectangle([x0, y0, x0 + cw - 12, y0 + chh - 10], 12, fill=(36, 40, 66, 255))
        sheet.alpha_composite(up(fr["peak"], k), (x0 + 3, y0 + 6))
        for j, key in enumerate(("shift", "raise", "peak")):
            sheet.alpha_composite(up(fr[key], 1), (x0 + 6 + j * 92, y0 + FH * k + 12))
        dr.text((x0 + 12, y0 + FH * k + 84), f"{code}. {name}", font=f_t, fill=(255, 255, 255, 255))
        words, line, lines = blurb.split(), "", []
        for w_ in words:
            if dr.textlength((line + " " + w_).strip(), font=f_s) > cw - 40:
                lines.append(line); line = w_
            else:
                line = (line + " " + w_).strip()
        lines.append(line)
        for li, ln in enumerate(lines[:3]):
            dr.text((x0 + 12, y0 + FH * k + 114 + li * 20), ln, font=f_s, fill=(190, 196, 220, 255))
    sheet.save(OUT / f"{stem}.png")
    # raw frame strip for the web preview: one row per variant, columns ready/shift/raise/peak/stomp
    strip = np.zeros((FH * len(all_frames), FW * 5, 4), dtype=np.uint8)
    for r, (_, _, _, fr) in enumerate(all_frames):
        for c, key in enumerate(("ready", "shift", "raise", "peak", "stomp")):
            strip[r * FH:(r + 1) * FH, c * FW:(c + 1) * FW] = fr[key]
    Image.fromarray(strip, "RGBA").save(OUT / f"{stem}_frames.png")

    # animated comparison, all five in sync
    seq = [("ready", 20), ("shift", 8), ("raise", 8), ("peak", 34), ("stomp", 8), ("ready", 14)]
    k = 3
    gif = []
    bg = (196, 160, 110)
    for key, ticks in seq:
        for t in range(0, ticks, 2):
            im = Image.new("RGBA", (len(all_frames) * (FW * k + 10) + 10, FH * k + 50), (*bg, 255))
            d2 = ImageDraw.Draw(im)
            for i, (code, name, _, fr) in enumerate(all_frames):
                x0 = 10 + i * (FW * k + 10)
                d2.ellipse([x0 + 26 * k, (FH - 6) * k + 10, x0 + 70 * k, (FH + 2) * k + 10], fill=(150, 112, 70, 255))
                shake = 0
                if key == "stomp" and t < 6:
                    shake = [2, -2, 1][t // 2] * k
                im.alpha_composite(up(fr[key], k), (x0, 6 + shake))
                d2.text((x0 + 10, FH * k + 14), f"{code}. {name}", font=f_t, fill=(40, 24, 10, 255))
            gif.append(im.convert("RGB"))
    gif[0].save(OUT / f"{stem}.gif", save_all=True, append_images=gif[1:], duration=33, loop=0)


if __name__ == "__main__":
    main()
