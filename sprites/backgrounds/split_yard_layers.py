"""Split each stable-yard background into a sky layer and a yard layer.

The day cycle draws its own blended sky and a moving sun and moon, then lays
the yard on top. Every yard script paints the sky first, in a section that
starts at "# ---- sky: bands with dither seams ----" and ends right before
"# distant hills", so this runs each script twice:

- sky:  only the sky section, with any sun or moon removed, plus the dark veil
        over the top rows that keeps the HUD text readable.
- yard: the whole script with the sky section replaced by magenta, which then
        becomes transparent.

Writes layers/sky_<key>.png and layers/fg_<key>.png (RGBA) next to this file.
Run:  python sprites/backgrounds/split_yard_layers.py
"""
import os
import re
import tempfile

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
LAYERS = os.path.join(HERE, 'layers')
SKY_START = '# ---- sky: bands with dither seams ----'
SKY_END = '# distant hills'
MAGENTA = (255, 0, 255)

# key, script, comment that starts the sun or moon block, comment that ends it
YARDS = [
    ('night', 'make_yard_night.py', '# moon:', '# thin moonlit'),
    ('morning', 'make_yard_morning.py', '# low sun', '# clouds'),
    ('noon', 'make_yard_noon.py', None, None),
    ('dusk', 'make_yard.py', None, None),
]


def cut_block(src, start, end):
    """Removes the lines from the one starting with `start` up to, not including, `end`."""
    if not start:
        return src
    a = src.index(start)
    b = src.index(end, a)
    return src[:a] + src[b:]


def drop_top_veil(src):
    """The veil recolours sky pixels by value, which would eat the magenta."""
    return re.sub(r'\n# top 30[^\n]*\n(?:[ \t]+[^\n]*\n|\n)*', '\n', src, count=1)


def run(src, workdir):
    src = re.sub(r'^OUT = .*$', f'OUT = {workdir!r}', src, flags=re.M)
    ns = {'__file__': os.path.join(workdir, 'x.py'), '__name__': '__split__'}
    exec(compile(src, 'yard', 'exec'), ns)
    return ns


def sky_layer(src, sun_start, sun_end, workdir):
    head = src[:src.index(SKY_END)]
    head = cut_block(head, sun_start, sun_end)
    ns = run(head, workdir)
    img = ns['img'].copy()
    bayer, sky = ns['BAYER'], ns['SKY']
    for y in range(14):
        for x in range(img.shape[1]):
            if bayer[y % 4, x % 4] / 16 < (14 - y) / 20.0:
                img[y, x] = sky[0] if y < 8 else sky[1]
    return img


def yard_layer(src, workdir):
    a = src.index(SKY_START)
    b = src.index(SKY_END, a)
    src = src[:a] + 'img[:, :] = (255, 0, 255)\n' + src[b:]
    ns = run(drop_top_veil(src), workdir)
    rgb = ns['img']
    see = np.all(rgb == MAGENTA, axis=-1)
    # A dark dither some scripts lay over the top rows belongs to the sky, not the yard.
    for c in ns['SKY'][:2]:
        see[:14] |= np.all(rgb[:14] == c, axis=-1)
    alpha = np.where(see, 0, 255).astype(np.uint8)
    return np.dstack([rgb, alpha])


def main():
    os.makedirs(LAYERS, exist_ok=True)
    with tempfile.TemporaryDirectory() as workdir:
        for key, script, sun_start, sun_end in YARDS:
            path = os.path.join(HERE, script)
            if not os.path.exists(path):
                print(f'skip {key}: {script} not found')
                continue
            src = open(path).read()
            Image.fromarray(sky_layer(src, sun_start, sun_end, workdir), 'RGB').save(
                os.path.join(LAYERS, f'sky_{key}.png'))
            fg = yard_layer(src, workdir)
            Image.fromarray(fg, 'RGBA').save(os.path.join(LAYERS, f'fg_{key}.png'))
            clear = int((fg[:, :, 3] == 0).sum())
            print(f'{key}: sky ok, yard ok, {clear} sky pixels see through')


if __name__ == '__main__':
    main()
