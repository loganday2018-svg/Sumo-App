#ifndef KAIJU_YOKOZUNA_SHIKO_H
#define KAIJU_YOKOZUNA_SHIKO_H

// Shiko (leg raise + stomp), front view. 96x64 frames drawn as two OBJs:
//   left  64x64 (square, size 3) at x
//   right 32x64 (tall,   size 3) at x + 64
// On a stomp step: shake the screen 2px for 6 ticks (BG scroll + OBJ y) and
// spawn the 64x32 dust OBJ centred on the planted foot (at frame x - 8 for the
// left foot, x + 40 for the right, y + 36), advancing its 4 frames every 3 ticks.

#define SHIKO_FRAME_COUNT 9
#define SHIKO_FRAME_WORDS (96 * 8)   // 64 + 32 tiles, 8 words each
#define SHIKO_STEPS 12

typedef struct { unsigned char frame, ticks, stomp; } ShikoStep;

extern const unsigned int kaijuShikoTiles[SHIKO_FRAME_COUNT * SHIKO_FRAME_WORDS];
extern const unsigned int kaijuShikoDustTiles[4 * 256];
extern const ShikoStep    kaijuShikoTimeline[SHIKO_STEPS];

#endif
