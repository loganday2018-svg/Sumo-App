"""GBA-screen (240x160) mock-ups for five shiko mini-game ideas.

Uses the real Kaiju Yokozuna shiko frames plus simple new pieces
(spirit wisps, meters, notes). Output: sprites/minigames/
Run:  python sprites/make_minigame_mockups.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
OUT = HERE / "minigames"
SHEET = Image.open(HERE / "shiko" / "kaiju_yokozuna_shiko_sheet.png").convert("RGBA")
ORDER = ["ready", "settle", "stomp", "shift", "raise", "peak", "shift_r", "raise_r", "peak_r"]
DUST = Image.open(HERE / "shiko" / "kaiju_yokozuna_shiko_dust.png").convert("RGBA")
OYAKATA = Image.open(HERE / "concepts" / "rikishi_set" / "oyakata.png").convert("RGBA")

BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
F7, F9, F12, F16 = (ImageFont.truetype(BOLD, s) for s in (7, 9, 12, 16))

NAVY, NAVY2, EDGE = (22, 24, 48), (36, 40, 78), (232, 226, 250)
GOLD, WHITE, RED, GREEN, CYAN = (255, 214, 90), (250, 250, 250), (230, 70, 60), (90, 220, 120), (120, 220, 255)


def frame(name):
    i = ORDER.index(name)
    return SHEET.crop((i * 96, 0, i * 96 + 96, 64))


def swap(img, mapping):
    a = np.array(img)
    for src, dst in mapping.items():
        m = (a[:, :, 0] == src[0]) & (a[:, :, 1] == src[1]) & (a[:, :, 2] == src[2]) & (a[:, :, 3] > 0)
        a[m, :3] = dst
    return Image.fromarray(a, "RGBA")


RIVAL = {  # palette swap: red silk, tanned skin
    (75, 36, 117): (138, 22, 22), (110, 58, 166): (200, 40, 40), (156, 107, 209): (240, 90, 74),
    (156, 90, 60): (141, 79, 44), (217, 137, 95): (201, 123, 75), (242, 180, 138): (234, 168, 119),
    (255, 217, 184): (255, 210, 166),
}


def dohyo(dim=False):
    bg = Image.new("RGBA", (240, 160), (196, 160, 110, 255))
    d = ImageDraw.Draw(bg)
    for y in range(0, 160, 4):
        for x in range((y * 7) % 9, 240, 9):
            d.point((x, y), fill=(184, 148, 100))
    d.ellipse([20, 14, 220, 158], outline=(232, 214, 160), width=4)
    d.ellipse([22, 16, 218, 156], outline=(150, 120, 70), width=1)
    if dim:
        bg.alpha_composite(Image.new("RGBA", bg.size, (20, 10, 40, 90)))
    return bg


def panel(d, box, title=None):
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=NAVY, outline=EDGE)
    d.rectangle([x0 + 1, y0 + 1, x1 - 1, y0 + 1], fill=NAVY2)
    if title:
        d.text((x0 + 4, y0 + 2), title, font=F7, fill=GOLD)


def text(d, xy, s, font=F9, fill=WHITE, shadow=True, anchor=None):
    if shadow:
        d.text((xy[0] + 1, xy[1] + 1), s, font=font, fill=(20, 10, 20), anchor=anchor)
    d.text(xy, s, font=font, fill=fill, anchor=anchor)


def bar(d, box, frac, col, label=None):
    x0, y0, x1, y1 = box
    d.rectangle(box, fill=(10, 10, 20), outline=EDGE)
    w = int((x1 - x0 - 2) * frac)
    d.rectangle([x0 + 1, y0 + 1, x0 + 1 + w, y1 - 1], fill=col)
    d.line([x0 + 1, y0 + 1, x0 + 1 + w, y0 + 1], fill=tuple(min(255, c + 60) for c in col))
    if label:
        text(d, (x0, y0 - 9), label, F7, GOLD)


def shadow(img, cx, y, w=44):
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).ellipse([cx - w // 2, y - 4, cx + w // 2, y + 4], fill=(90, 60, 30, 110))
    img.alpha_composite(sh)


WISP = [
    "....oo....",
    "...oCCo...",
    "..oCWWCo..",
    ".oCWWWWCo.",
    "oCWWWWWWCo",
    "oCWkWWkWCo",
    "oCWWWWWWCo",
    ".oCWWWWCo.",
    "..oCCCCo..",
    ".oC.oC.oC.",
    ".o...o..o.",
]


def wisp(img, x, y, alpha=255):
    pal = {"o": (40, 60, 140), "C": (120, 200, 255), "W": (235, 250, 255), "k": (20, 30, 70)}
    for j, row in enumerate(WISP):
        for i, ch in enumerate(row):
            if ch in pal and 0 <= x + i < 240 and 0 <= y + j < 160:
                img.putpixel((x + i, y + j), (*pal[ch], alpha))


# ---------------------------------------------------------------- 1. Spirit Stomp
def spirit_stomp():
    im = dohyo(dim=True)
    d = ImageDraw.Draw(im)
    cx, cy = 120, 96
    # shockwave preview (how far this stomp would reach right now)
    for a in range(0, 360, 6):
        x = cx + 58 * math.cos(math.radians(a)); y = cy + 30 * math.sin(math.radians(a))
        d.rectangle([x, y, x + 1, y + 1], fill=CYAN)
    shadow(im, cx, cy + 22)
    im.alpha_composite(frame("peak"), (cx - 48, cy - 42))
    for (x, y) in ((28, 40), (186, 52), (40, 118), (196, 120), (150, 34), (64, 40), (206, 88)):
        wisp(im, x, y)
    # spirit glow trails
    for (x, y) in ((24, 48), (194, 60), (36, 126)):
        d.point((x, y), fill=CYAN); d.point((x - 3, y + 2), fill=(80, 140, 220))
    panel(d, (4, 4, 92, 30), "POWER")
    bar(d, (8, 16, 88, 24), 0.78, (255, 170, 40))
    panel(d, (176, 4, 236, 30), "SPIRITS")
    text(d, (182, 14), "12 / 20", F9)
    panel(d, (4, 140, 236, 156))
    text(d, (10, 142), "HOLD A to raise the leg · RELEASE to stomp", F7, WHITE, shadow=False)
    return im


# ---------------------------------------------------------------- 2. Balance Hold
def balance_hold():
    im = dohyo()
    d = ImageDraw.Draw(im)
    cx, cy = 120, 104
    shadow(im, cx, cy + 22)
    im.alpha_composite(frame("peak"), (cx - 48, cy - 42))
    # balance arc meter
    mx, my, r = 120, 52, 28
    panel(d, (mx - 44, 6, mx + 44, 58), "BALANCE")
    for a in range(200, 341, 2):
        x = mx + r * math.cos(math.radians(a)); y = my + r * math.sin(math.radians(a))
        col = GREEN if 255 <= a <= 285 else ((255, 200, 60) if 230 <= a <= 310 else RED)
        d.rectangle([x - 1, y - 1, x + 1, y + 1], fill=col)
    ang = math.radians(256)
    d.line([mx, my, mx + (r - 4) * math.cos(ang), my + (r - 4) * math.sin(ang)], fill=WHITE, width=2)
    d.ellipse([mx - 2, my - 2, mx + 2, my + 2], fill=GOLD)
    panel(d, (4, 64, 60, 96), "HOLD")
    text(d, (10, 76), "3.2 s", F12, GOLD)
    panel(d, (180, 64, 236, 96), "SCORE")
    text(d, (186, 76), "4,850", F9)
    text(d, (190, 112), "x3", F16, GOLD)
    panel(d, (4, 140, 236, 156))
    text(d, (10, 142), "D-PAD keeps him steady · higher hold = more points", F7, WHITE, shadow=False)
    return im


# ---------------------------------------------------------------- 3. Taiko Rhythm
def taiko():
    im = dohyo()
    d = ImageDraw.Draw(im)
    cx, cy = 120, 74
    shadow(im, cx, cy + 22)
    im.alpha_composite(frame("raise"), (cx - 48, cy - 42))
    text(d, (120, 22), "PERFECT!", F16, GOLD, anchor="mm")
    text(d, (120, 36), "COMBO 24", F9, WHITE, anchor="mm")
    # note lane
    d.rectangle([0, 112, 239, 138], fill=(40, 16, 16), outline=(120, 60, 40))
    d.line([0, 125, 239, 125], fill=(70, 30, 24))
    hx = 26
    d.ellipse([hx - 11, 114, hx + 11, 136], outline=GOLD, width=2)
    d.ellipse([hx - 7, 118, hx + 7, 132], outline=(160, 120, 40))
    notes = [(52, "L", (230, 70, 60)), (84, "R", (70, 140, 240)), (118, "L", (230, 70, 60)),
             (150, "R", (70, 140, 240)), (192, "A", (255, 200, 40))]
    for x, lab, col in notes:
        rr = 10 if lab == "A" else 8
        d.ellipse([x - rr, 125 - rr, x + rr, 125 + rr], fill=col, outline=WHITE)
        text(d, (x, 125), lab, F9, WHITE, anchor="mm")
    text(d, (190, 102), "STOMP", F7, GOLD)
    panel(d, (4, 4, 70, 28), "SCORE")
    text(d, (10, 14), "18,200", F9)
    panel(d, (186, 4, 236, 28), "BPM")
    text(d, (196, 14), "96", F9)
    panel(d, (4, 142, 236, 157))
    text(d, (10, 144), "L / R raise that leg on the beat · A stomps", F7, WHITE, shadow=False)
    return im


# ---------------------------------------------------------------- 4. 200 a Day
def two_hundred():
    im = dohyo()
    d = ImageDraw.Draw(im)
    cx, cy = 104, 100
    shadow(im, cx, cy + 22)
    im.alpha_composite(frame("stomp"), (cx - 48, cy - 42))
    im.alpha_composite(DUST.crop((64, 0, 128, 32)), (cx - 24 - 32, cy - 6))
    # stablemaster with speech bubble
    ok = OYAKATA.resize((40, 44), Image.NEAREST)
    shadow(im, 206, 132, 28)
    im.alpha_composite(ok, (186, 90))
    d.rounded_rectangle([150, 52, 236, 82], 4, fill=WHITE, outline=(20, 10, 20))
    d.polygon([(204, 82), (212, 82), (210, 90)], fill=WHITE, outline=(20, 10, 20))
    d.line([(205, 82), (211, 82)], fill=WHITE)
    d.text((156, 56), "Hips lower!", font=F9, fill=(20, 10, 20))
    d.text((156, 68), "Don't rush it.", font=F7, fill=(80, 60, 80))
    panel(d, (4, 4, 90, 38), "SHIKO")
    text(d, (12, 16), "037", F16, GOLD)
    text(d, (52, 22), "/ 200", F9)
    panel(d, (100, 4, 236, 38), "")
    bar(d, (106, 20, 170, 28), 0.62, (90, 200, 255), "STAMINA")
    text(d, (180, 10), "FORM", F7, GOLD)
    text(d, (186, 18), "A", F16, GREEN)
    text(d, (206, 22), "+2 STR", F7, WHITE)
    panel(d, (4, 142, 236, 157))
    text(d, (10, 144), "Alternate L and R · steady rhythm keeps form high", F7, WHITE, shadow=False)
    return im


# ---------------------------------------------------------------- 5. Shiko Duel
def duel():
    im = dohyo()
    d = ImageDraw.Draw(im)
    d.line([(108, 92), (118, 92)], fill=WHITE, width=2)
    d.line([(122, 92), (132, 92)], fill=WHITE, width=2)
    you, rival = (70, 104), (172, 104)
    shadow(im, you[0], you[1] + 22)
    shadow(im, rival[0], rival[1] + 22)
    im.alpha_composite(frame("ready"), (you[0] - 48, you[1] - 42))
    im.alpha_composite(swap(frame("peak_r"), RIVAL), (rival[0] - 48, rival[1] - 42))
    # pattern the rival is showing
    panel(d, (120, 6, 236, 36), "RIVAL'S PATTERN")
    seq = ["L", "R", "R", "L", "R"]
    for i, k in enumerate(seq):
        x = 128 + i * 21
        col = (230, 70, 60) if k == "L" else (70, 140, 240)
        if i == 4:
            d.rectangle([x - 2, 16, x + 15, 33], outline=GOLD)
        d.rectangle([x, 18, x + 13, 31], fill=col, outline=WHITE)
        text(d, (x + 7, 25), k, F9, WHITE, anchor="mm")
    panel(d, (4, 6, 112, 36), "YOUR TURN")
    for i, k in enumerate(seq[:3]):
        x = 10 + i * 21
        col = (230, 70, 60) if k == "L" else (70, 140, 240)
        d.rectangle([x, 18, x + 13, 31], fill=col, outline=WHITE)
        text(d, (x + 7, 25), k, F9, WHITE, anchor="mm")
    for i in range(3, 5):
        x = 10 + i * 21
        d.rectangle([x, 18, x + 13, 31], outline=(120, 120, 160))
        text(d, (x + 7, 25), "?", F9, (120, 120, 160), anchor="mm")
    text(d, (120, 52), "ROUND 5", F12, GOLD, anchor="mm")
    text(d, (70, 132), "YOU", F7, GOLD, anchor="mm")
    text(d, (172, 132), "AKAGANE", F7, (255, 140, 120), anchor="mm")
    panel(d, (4, 142, 236, 157))
    text(d, (10, 144), "Repeat the rival's stomps · each round adds one", F7, WHITE, shadow=False)
    return im


GAMES = [
    ("1", "Spirit Stomp", "Hold to raise, release to stomp. Bigger hold, bigger shockwave, but the wisps close in.", spirit_stomp),
    ("2", "Balance Hold", "Keep the needle in the green while the leg is up. Longer hold scores more.", balance_hold),
    ("3", "Taiko Rhythm", "Raise each leg and stomp on the drum beat. Combos shake the screen harder.", taiko),
    ("4", "200 a Day", "Endurance training. Stamina, form grade and stat gains, with the stablemaster watching.", two_hundred),
    ("5", "Shiko Duel", "Repeat the rival's stomp pattern. One more stomp each round.", duel),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shots = []
    for code, name, blurb, fn in GAMES:
        im = fn()
        key = name.lower().replace(" ", "_")
        im.save(OUT / f"{code}_{key}.png")
        big = im.resize((720, 480), Image.NEAREST)
        big.save(OUT / f"{code}_{key}_x3.png")
        shots.append((code, name, blurb, im))
    # combined sheet, two columns
    k = 3
    cw, ch = 240 * k + 30, 160 * k + 112
    cols = 2
    rows = math.ceil(len(shots) / cols)
    sheet = Image.new("RGB", (cols * cw + 30, rows * ch + 80), (24, 26, 44))
    d = ImageDraw.Draw(sheet)
    reg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    d.text((30, 22), "SHIKO MINI-GAME MOCK-UPS  (GBA screen, 240x160)", font=ImageFont.truetype(BOLD, 30), fill=GOLD)
    for i, (code, name, blurb, im) in enumerate(shots):
        x0, y0 = 15 + (i % cols) * cw, 75 + (i // cols) * ch
        d.rounded_rectangle([x0, y0, x0 + cw - 15, y0 + ch - 12], 14, fill=(36, 40, 66))
        sheet.paste(im.resize((240 * k, 160 * k), Image.NEAREST), (x0 + 8, y0 + 8))
        d.text((x0 + 14, y0 + 160 * k + 16), f"{code}. {name}", font=ImageFont.truetype(BOLD, 24), fill=WHITE)
        words, line, lines = blurb.split(), "", []
        for w_ in words:
            if d.textlength((line + " " + w_).strip(), font=reg) > cw - 50:
                lines.append(line); line = w_
            else:
                line = (line + " " + w_).strip()
        lines.append(line)
        for li, ln in enumerate(lines[:2]):
            d.text((x0 + 14, y0 + 160 * k + 46 + li * 24), ln, font=reg, fill=(190, 196, 220))
    sheet.save(OUT / "minigame_mockups.png")


if __name__ == "__main__":
    main()
