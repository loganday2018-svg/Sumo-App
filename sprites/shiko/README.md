# Kaiju Yokozuna shiko

Regenerate with `python sprites/make_shiko.py`.

Shiko is the training stomp: squat, shift weight, raise one leg high to the side, hold, slam it down, then the other leg.

- 9 unique 96x64 frames, front view. Each frame is two GBA OBJs: a 64x64 on the left and a 32x64 on the right.
- Same 16-colour palette as the walk cycle, so both use one palette bank.
- Full set (both legs) is 180 ticks, 3 seconds at 60 fps. Timing lives in `kaijuShikoTimeline`.
- On each stomp: shake the screen 2 px for 6 ticks and play the 4-frame 64x32 dust cloud under the planted foot, 3 ticks per frame.
- The other-side frames are redrawn, not flipped, so the topknot and lighting stay consistent.
