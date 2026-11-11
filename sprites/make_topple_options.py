"""Five topple (lose-balance) animation options for the Kaiju Yokozuna.

Each option is a strip of 128x96 frames plus a timeline of
{frame, ticks, dx, dy, fx}. Rotated frames use nearest-neighbour rotation,
which is what the GBA's affine sprites do in hardware.

Outputs (sprites/topple/): topple_frames.png, topple_timelines.json
Run:  python sprites/make_topple_options.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import make_shiko_mockups as mk  # noqa: E402  (pose renderer, 96x64)

HERE = Path(__file__).parent
OUT = HERE / "topple"
CW, CH = 176, 96          # option frame canvas (wide enough to land in)
PX, PY = 40, 32           # where a 96x64 shiko frame sits inside it (feet at y = 91)
WALK = Image.open(HERE / "walk" / "kaiju_yokozuna_walk_sheet.png").convert("RGBA")

R1 = dict(next(v for v in mk.VARIANTS_V2 if v[0] == "R1")[3])


def pose(p, j=None):
    return Image.fromarray(mk.render(p, j), "RGBA")


def place(img, x=PX, y=PY):
    c = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    c.alpha_composite(img, (x, y))
    return c


def rot(img, deg, pivot, ground=91):
    """Rotate on a roomy canvas (nothing clips), then set the lowest pixel back on the ground line."""
    pad = 64
    big = Image.new("RGBA", (CW + 2 * pad, CH + 2 * pad), (0, 0, 0, 0))
    big.alpha_composite(img, (pad, pad))
    big = big.rotate(deg, resample=Image.NEAREST, center=(pivot[0] + pad, pivot[1] + pad))
    bbox = big.getbbox()
    out = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
    out.alpha_composite(big, (-pad, ground + 1 - bbox[3]))
    return out


def walk(row, flip=False):
    f = WALK.crop((0, row * 64, 64, row * 64 + 64))
    return f.transpose(Image.FLIP_LEFT_RIGHT) if flip else f


# ---------------------------------------------------------------- base poses
peak = place(pose(R1))
flailA = place(pose(dict(R1, tilt=34, a1=40, a2=44, handL="up", handR="out", mouth="shout", eyes="wide")))
flailB = place(pose(dict(R1, tilt=21, a1=58, a2=62, handL="out", handR="up", mouth="shout", eyes="wide")))
settle = place(Image.fromarray(np.array(Image.open(HERE / "shiko" / "kaiju_yokozuna_shiko_sheet.png").convert("RGBA"))[:, 96:192], "RGBA"))
ready = place(Image.fromarray(np.array(Image.open(HERE / "shiko" / "kaiju_yokozuna_shiko_sheet.png").convert("RGBA"))[:, 0:96], "RGBA"))
ball = Image.fromarray(np.array(Image.open(HERE / "shiko" / "kaiju_yokozuna_shiko_sheet.png").convert("RGBA"))[:, 192:288], "RGBA")  # deep stomp squat


def flat_pose(torso_y, head_y, hip, knee, foot, sh_y, hand, mouth, eyes, trx=18, try_=13.5):
    p = dict(tilt=0, torso=(0, torso_y), trx=trx, try_=try_, mouth=mouth, eyes=eyes)
    m = lambda pt: (-pt[0], pt[1])
    j = dict(hipL=hip, kneeL=knee, footL=foot, hipR=m(hip), kneeR=m(knee), footR=m(foot),
             shL=(-13, sh_y), shR=(13, sh_y), handL=hand, handR=m(hand), head=(0, head_y))
    return place(pose(p, j))


sit = flat_pose(45, 27, (-10, 54), (-22, 51), (-25, 58), 36, (-25, 53), "frown", "closed")
half_split = flat_pose(40, 22, (-10, 50), (-22, 53), (-32, 58), 31, (-21, 49), "grit", "wide")
splits = flat_pose(46, 28, (-11, 54), (-26, 56), (-41, 57), 37, (-15, 57), "shout", "wide", try_=13)
squashed = settle.resize((CW, int(CH * 0.72)), Image.NEAREST)
squash = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
squash.alpha_composite(squashed, (0, CH - squashed.height))

FOOT = (PX + 48 + 22, 91)       # standing (right) foot in the 128x96 canvas
LEAD = [  # shared lead-in: holding the peak and starting to wobble
    ("peak", 10, 0, 0, None), ("peak", 4, -1, 0, None), ("peak", 4, 1, 0, None),
    ("peak", 4, -2, 0, None), ("peak", 4, 2, 0, None),
]


def build():
    opts = []

    # 1. Timber!  -- stiff tip over sideways, slam, stars, get up
    fr = {"peak": peak}
    for a in (12, 28, 48, 70, 88):          # negative = clockwise: falls over the standing leg
        fr[f"t{a}"] = rot(peak, -a, FOOT)
    fr["sit"], fr["settle"], fr["ready"] = sit, settle, ready
    tl = LEAD + [("t12", 6, 0, 0, None), ("t28", 5, 0, 0, None), ("t48", 4, 0, 0, None),
                 ("t70", 3, 0, 0, None), ("t88", 3, 0, -3, "dust,shake"), ("t88", 3, 0, 0, None),
                 ("t88", 3, 0, -1, None), ("t88", 46, 0, 0, "stars"), ("sit", 18, 0, 0, "stars"),
                 ("settle", 14, 0, 0, None), ("ready", 20, 0, 0, None)]
    opts.append(("timber", "Timber!", "Goes stiff and tips over sideways like a felled tree, slams down, sees stars, then gets up.", fr, tl))

    # 2. Hop Hop Plop -- flailing hops on the standing leg, then sits down hard
    fr = {"peak": peak, "fa": flailA, "fb": flailB, "sit": sit, "settle": settle, "ready": ready}
    tl = LEAD + [("fa", 5, 2, -4, None), ("fa", 4, 3, 0, None), ("fb", 5, 5, -4, None), ("fb", 4, 6, 0, None),
                 ("fa", 5, 8, -4, None), ("fa", 4, 9, 0, None), ("fb", 5, 11, -5, None),
                 ("sit", 3, 12, -4, None), ("sit", 4, 12, 0, "dust,shake"), ("sit", 3, 12, -1, None),
                 ("sit", 50, 12, 0, "stars"), ("settle", 14, 6, 0, None), ("ready", 20, 0, 0, None)]
    opts.append(("hop", "Hop Hop Plop", "Hops on his standing leg with arms windmilling, then drops onto his backside.", fr, tl))

    # 3. Spin Out -- spins on one foot faster and faster, collapses squashed
    fr = {"peak": peak, "down": place(walk(0), 56, 29), "right": place(walk(3), 56, 29), "up": place(walk(1), 56, 29),
          "left": place(walk(2), 56, 29), "squash": squash, "settle": settle, "ready": ready}
    spin = []
    for i, d in enumerate(["down", "right", "up", "left"] * 3):
        spin.append((d, max(2, 7 - i // 2), 0, 0, None))
    tl = LEAD + spin + [("squash", 3, 0, 0, "dust,shake"), ("squash", 50, 0, 0, "stars"),
                        ("settle", 14, 0, 0, None), ("ready", 20, 0, 0, None)]
    opts.append(("spin", "Spin Out", "Spins on one foot faster and faster, then collapses into a squashed, dizzy heap.", fr, tl))

    # 4. Rolling Boulder -- tips, curls into a ball and rolls across the ring
    fr = {"peak": peak, "t20": rot(peak, 20, FOOT), "t45": rot(peak, 45, FOOT), "sit": sit, "settle": settle, "ready": ready}
    bc = place(ball)
    for a in range(0, 720, 45):
        fr[f"b{a}"] = rot(bc, a, (88, 64))
    tl = LEAD + [("t20", 5, 0, 0, None), ("t45", 4, 0, 0, None)]
    x = 0
    for i, a in enumerate(range(0, 720, 45)):
        x -= 3
        tl.append((f"b{a}", 3, x, -1 if i % 2 else 0, "dust" if i == 0 else None))
    tl += [("sit", 4, x, -2, "shake"), ("sit", 50, x, 0, "stars"), ("settle", 16, x // 2, 0, None), ("ready", 20, 0, 0, None)]
    opts.append(("roll", "Rolling Boulder", "Tips over, curls into a ball and rolls across the ring before sitting up dazed.", fr, tl))

    # 5. Splits Fail -- standing foot slides out into a full matawari split
    fr = {"peak": peak, "half": half_split, "splits": splits, "settle": settle, "ready": ready}
    tl = LEAD + [("half", 5, 0, 0, None), ("splits", 3, 0, -2, "dust,shake"), ("splits", 3, 0, 0, None),
                 ("splits", 54, 0, 0, "sweat"), ("half", 16, 0, 0, "sweat"), ("settle", 14, 0, 0, None),
                 ("ready", 20, 0, 0, None)]
    opts.append(("splits", "Splits Fail", "His foot slides out into a full matawari split. Pained face, then he struggles back up.", fr, tl))
    return opts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    opts = build()
    ncols = max(len(fr) for *_, fr, _ in opts)
    sheet = Image.new("RGBA", (CW * ncols, CH * len(opts)), (0, 0, 0, 0))
    meta = []
    for r, (key, name, blurb, fr, tl) in enumerate(opts):
        names = list(fr)
        for c, k in enumerate(names):
            sheet.alpha_composite(fr[k], (c * CW, r * CH))
        heads = []
        for k in names:  # top-centre of each frame, used to place stars / sweat at runtime
            bb = fr[k].getbbox() or (0, 0, CW, CH)
            a = np.array(fr[k])[:, :, 3] > 0
            top = bb[1]
            xs = np.nonzero(a[top:top + 6].any(axis=0))[0]
            heads.append([int(xs.mean()) if len(xs) else CW // 2, int(top)])
        meta.append(dict(key=key, name=name, blurb=blurb, heads=heads,
                         steps=[[names.index(f), t, dx, dy, fx or ""] for f, t, dx, dy, fx in tl]))
    sheet.save(OUT / "topple_frames.png")
    (OUT / "topple_timelines.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    big = sheet.resize((sheet.width * 2, sheet.height * 2), Image.NEAREST)
    bg = Image.new("RGBA", big.size, (36, 40, 66, 255)); bg.alpha_composite(big); bg.save(OUT / "topple_frames_x2.png")
    for m in meta:
        print(m["name"], "ticks:", sum(s[1] for s in m["steps"]))


if __name__ == "__main__":
    main()
