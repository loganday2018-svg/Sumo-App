# Kaiju Yokozuna walk cycle

Regenerate everything with `python sprites/make_walk_cycle.py`.

| File | What it is |
|---|---|
| `kaiju_yokozuna_walk_sheet.png` | Native 64x64 frames. Rows: down, up, left, right. Columns: stand, step A, stand, step B. |
| `kaiju_yokozuna_walk_x4.gif` | All four directions looping at game speed. |
| `kaiju_yokozuna_demo.gif` | Walking a square on a 240x160 GBA-sized screen. |
| `kaiju_yokozuna_walk.c/.h` | GBA data: 16-colour BGR555 palette, 9 frames of 4bpp tiles (1D mapping), animation table. |

## Using it on the GBA

- Sprite is a 64x64 square OBJ, 4bpp, using one 16-colour palette bank.
- Only 9 unique frames exist. Right reuses the left frames with the OBJ horizontal-flip bit.
- Each frame is 2 KB. DMA the current frame into one fixed OBJ VRAM slot during VBlank instead of loading all frames.
- Walk speed is 1 px per tick. Advance the animation every 8 ticks, so one full cycle covers 32 px, which is one footstep per 16 px tile.
- A short button tap turns the sprite to face the new direction without moving. Holding the button starts walking.

```c
// per tick, while walking
if (++animTimer >= KAIJU_TICKS_PER_ANIM_FRAME) { animTimer = 0; animStep = (animStep + 1) & 3; }
int frame = kaijuWalkAnim[dir][moving ? animStep : 0];
dma3_cpy(&tile_mem[4][0], &kaijuYokozunaTiles[frame * KAIJU_FRAME_WORDS], KAIJU_FRAME_WORDS * 4);
obj_set_attr(obj, ATTR0_SQUARE | ATTR0_4BPP, ATTR1_SIZE_64 | (dir == DIR_RIGHT ? ATTR1_HFLIP : 0), ATTR2_PALBANK(0) | 0);
```
