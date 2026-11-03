# Exp 4, Kinetix car-on-ramp — one-shot weight edit by Claude Fable 5.1 (no tools; written from the prompt's scene,
# observation layout, episode record and edit language; NOT yet scored here because Kinetix could not be installed in
# this session)

## Why the trained network fails (from the prompt's episode record and scene)

The recorded episode shows entry 0 = value 1 (drive +1) from the first step: the car body goes from x 2.80 to 2.38
(step 16) to 2.00 (step 32) and then sits at x 2.00, y 0.92, angle -0.27 for the remaining 224 steps with the policy
flickering between "off" and "+1". So +1 drives the car LEFT, straight at the ramp (shape 6, a bar tilted 0.95 rad
whose high end is at x 1.9, y 0.8), and from a standing start 0.3 m away it has no speed: it climbs a little, stalls
on the slope and alternates throttle and coast forever (probabilities 1.00/0.00 then 0.37/0.63 ...). The red strip
(shape 7) lies on the floor at x 1.18..2.07 right after the ramp; the blue box (shape 4) is at x 0.03..0.44. Clearing
the strip needs a run-up: back up to the right (drive -1, value 2) on the flat floor (x up to ~4.6, the wall is at
x 5), then drive +1 at full motor speed all the way through the ramp, fly over the strip and keep driving into the box.
The trained network never samples value 2 (its probability is 0.00 in every recorded step), so it can never build
speed; that is the single cause of the failure.

## The fix: a programmed controller on a few hijacked units, with the velocity sign as the "phase" memory

Rule for entry 0 at every step:
    drive +1 (value 1) if  body x >= 3.8   OR   body vx <= -0.1   OR   green-wheel x < 0.95
    drive -1 (value 2) otherwise;  value 0 (motor off) is never chosen.
From rest at x 2.8 (vx 0) this backs the car up; at x 3.8 it switches to +1 (the car still coasts right a little while
decelerating, then returns, and from then on vx < 0 keeps +1 through the ramp and the jump even once x < 3.8); after
landing near the box the green wheel is at x < 0.95 and +1 is held so the car pushes into the box. If a jump ever
fails and the car stalls or rolls back (vx >= 0 at x < 3.8 and wheel x >= 0.95) it automatically backs up and retries.

Observation entries used (polygon slot 5 = car body: entries 52..77, x at 52, vx at 54; circle slot 12 = green wheel:
entries 234..251, x at 234). Units taken over: embed 125, 126, 127 (ReLU threshold detectors), actor0..actor4 unit
127 (an OR, then saturated pass-throughs). Their original incoming and outgoing weights are zeroed so the rest of the
trained network sees them as dead units, and the entry-0 output columns are rewritten entirely (the trained network
keeps its other outputs, which have no effect in this scene). Values are in the scale shown (embed.W scale 1e-2 etc.
does not matter: edit statements set real values).

```edits
# --- three threshold detectors in the embedding layer (ReLU) ---
embed.W[:, 125:128] = 0
embed.b[125:128] = 0
embed.W[52, 125] = 10          # e125 = relu(10 * (body_x - 3.8))
embed.b[125] = -38
embed.W[54, 126] = -10         # e126 = relu(-10 * body_vx - 1)  > 0 iff vx < -0.1
embed.b[126] = -1
embed.W[234, 127] = -10        # e127 = relu(9.5 - 10 * green_x) > 0 iff green_x < 0.95
embed.b[127] = 9.5
# --- actor0 unit 127 = OR of the three detectors ---
actor0.W[:, 127] = 0
actor0.W[127, :] = 0
actor0.W[125:128, 127] = 10
actor0.b[127] = -2
# --- actor1..actor4 unit 127: saturated pass-through ---
actor1.W[:, 127] = 0
actor1.W[127, :] = 0
actor1.W[127, 127] = 4
actor1.b[127] = 0
actor2.W[:, 127] = 0
actor2.W[127, :] = 0
actor2.W[127, 127] = 4
actor2.b[127] = 0
actor3.W[:, 127] = 0
actor3.W[127, :] = 0
actor3.W[127, 127] = 4
actor3.b[127] = 0
actor4.W[:, 127] = 0
actor4.W[127, :] = 0
actor4.W[127, 127] = 4
actor4.b[127] = 0
# --- entry 0 logits come only from unit 127: +1 when it is on, -1 when it is off, never "motor off" ---
actor_head.W[:, 0:3] = 0
actor_head.b[0:3] = [-100, 0, 0]
actor_head.W[127, 1] = 60
actor_head.W[127, 2] = -60
```

What the edits do: three ReLU units read body x, body vx and green-wheel x and fire when the car is far enough right,
moving left, or already past the strip; one tanh unit ORs them and four more carry the +-1 flag to the output, where it
adds +60/-60 to the logits of "+1"/"-1" (the "off" logit is -100). The result is: back up, then full throttle left
through the ramp and into the blue box, retrying automatically if a jump falls short. About 3,100 of the 171,664
weights change (1.8%), most of them zeroed connections of the eight hijacked units.
