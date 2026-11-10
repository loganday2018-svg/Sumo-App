#ifndef KAIJU_YOKOZUNA_WALK_H
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
