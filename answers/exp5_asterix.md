# Exp 5, Asterix — one-shot network written by Claude Fable 5.1 (no tools, no simulation before scoring)

## Design

Facts used: one entity per row (slot i lives in row i+1); entities only move horizontally; the player moves before the
entities and collision is checked after each; enemies get faster (down to one cell per step) and spawn faster
(down to one per step), so late in the game nearly every row holds something. Entities enter at the edges, so the
player never visits columns 0 or 9; it never moves horizontally at all (left/right/fire logits are -50), so it stays
in column 5 where it starts, and every "distance" below is measured from column 5.

Per-row threat score (enemy channel 1, trail channel 4): an enemy at horizontal distance 1..3 that is moving TOWARD
column 5 (its trail is on the far side) scores >= 1; an enemy moving away, or one 4+ columns off, scores <= 0.
Enemy weights by column: [0, 0, 1.5, 2, 3, 5, 3, 2, 1.5, 0]; trail weights: [0, -0.5, -0.5, -2, -2.5, -3.5, -2.5, -2,
-0.5, 0] (chosen so that "enemy at c with trail behind it" lands at >= 1 and "enemy at c with trail in front of it"
lands at -0.5 or 0; gold trails alone are <= 0).
Per-row gold score (gold channel 2): approaching gold anywhere in columns 1..9 scores >= 1, receding gold 0:
gold weights [0.5, 1, 2, 3, 4, 5, 4, 3, 2, 1], trail weights [0, -0.5, -1, -2, -3, -4, -3, -2, -1, 0].

Layer 1 (32 units): the "player in row r AND threat in row r" detectors are built in ONE unit each with the trick
tanh(6 * (score - 0.5 + 10 * (player_in_row - 1))): the 10*(...) term switches the unit off unless the player is in
that row. Units 0-7: threat in my row (rows 1-8). Units 8-15: threat in the row above me (unit 8 = "I am in row 1",
where up is impossible). Units 16-23: threat in the row below me (unit 23 = "I am in row 8"). Units 24-27: player in
row band {1,2}, {3,4}, {5,6}, {7,8}. Units 28-31: approaching gold in those bands.
Layer 2: v0/v1/v2 = OR over the 8 units of each group (threat in my row / above / below); v3-v5 = "gold in a band
above mine"; v6-v8 = "gold in a band below mine".
Layer 3 (logit = sum of w*(h+1)/2): up = 30*[threat here] - 40*[threat above] + 15*[gold above]; down likewise;
no-op = 8. So: a threat in my row makes me step into a neighbouring row that has no approaching enemy within three
columns (30 vs -10); gold above/below pulls me one band at a time when the way is clear (15 vs 8); otherwise I sit.

```edits
# ===== layer 1 =====
# --- units 0-7: threat in MY row r (r = 1..8): enemy row r, trail row r, player row r, bias -63 ---
W1[61:120:6, 0] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[64:120:6, 0] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[60:120:6, 0] = 60
b1[0] = -63
W1[121:180:6, 1] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[124:180:6, 1] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[120:180:6, 1] = 60
b1[1] = -63
W1[181:240:6, 2] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[184:240:6, 2] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[180:240:6, 2] = 60
b1[2] = -63
W1[241:300:6, 3] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[244:300:6, 3] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[240:300:6, 3] = 60
b1[3] = -63
W1[301:360:6, 4] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[304:360:6, 4] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[300:360:6, 4] = 60
b1[4] = -63
W1[361:420:6, 5] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[364:420:6, 5] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[360:420:6, 5] = 60
b1[5] = -63
W1[421:480:6, 6] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[424:480:6, 6] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[420:480:6, 6] = 60
b1[6] = -63
W1[481:540:6, 7] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[484:540:6, 7] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[480:540:6, 7] = 60
b1[7] = -63
# --- units 8-15: threat in the row ABOVE me; unit 8 = I am in row 1 (no row above) ---
W1[60:120:6, 8] = 60
b1[8] = -30
W1[61:120:6, 9] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[64:120:6, 9] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[120:180:6, 9] = 60
b1[9] = -63
W1[121:180:6, 10] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[124:180:6, 10] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[180:240:6, 10] = 60
b1[10] = -63
W1[181:240:6, 11] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[184:240:6, 11] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[240:300:6, 11] = 60
b1[11] = -63
W1[241:300:6, 12] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[244:300:6, 12] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[300:360:6, 12] = 60
b1[12] = -63
W1[301:360:6, 13] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[304:360:6, 13] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[360:420:6, 13] = 60
b1[13] = -63
W1[361:420:6, 14] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[364:420:6, 14] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[420:480:6, 14] = 60
b1[14] = -63
W1[421:480:6, 15] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[424:480:6, 15] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[480:540:6, 15] = 60
b1[15] = -63
# --- units 16-23: threat in the row BELOW me; unit 23 = I am in row 8 (no row below) ---
W1[121:180:6, 16] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[124:180:6, 16] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[60:120:6, 16] = 60
b1[16] = -63
W1[181:240:6, 17] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[184:240:6, 17] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[120:180:6, 17] = 60
b1[17] = -63
W1[241:300:6, 18] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[244:300:6, 18] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[180:240:6, 18] = 60
b1[18] = -63
W1[301:360:6, 19] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[304:360:6, 19] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[240:300:6, 19] = 60
b1[19] = -63
W1[361:420:6, 20] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[364:420:6, 20] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[300:360:6, 20] = 60
b1[20] = -63
W1[421:480:6, 21] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[424:480:6, 21] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[360:420:6, 21] = 60
b1[21] = -63
W1[481:540:6, 22] = [0, 0, 9, 12, 18, 30, 18, 12, 9, 0]
W1[484:540:6, 22] = [0, -3, -3, -12, -15, -21, -15, -12, -3, 0]
W1[420:480:6, 22] = 60
b1[22] = -63
W1[480:540:6, 23] = 60
b1[23] = -30
# --- units 24-27: player in row band {1,2} {3,4} {5,6} {7,8} ---
W1[60:180:6, 24] = 6
b1[24] = -3
W1[180:300:6, 25] = 6
b1[25] = -3
W1[300:420:6, 26] = 6
b1[26] = -3
W1[420:540:6, 27] = 6
b1[27] = -3
# --- units 28-31: approaching gold in those bands (gold channel 2, trail channel 4) ---
W1[62:120:6, 28] = [3, 6, 12, 18, 24, 30, 24, 18, 12, 6]
W1[122:180:6, 28] = [3, 6, 12, 18, 24, 30, 24, 18, 12, 6]
W1[64:120:6, 28] = [0, -3, -6, -12, -18, -24, -18, -12, -6, 0]
W1[124:180:6, 28] = [0, -3, -6, -12, -18, -24, -18, -12, -6, 0]
b1[28] = -3
W1[182:240:6, 29] = [3, 6, 12, 18, 24, 30, 24, 18, 12, 6]
W1[242:300:6, 29] = [3, 6, 12, 18, 24, 30, 24, 18, 12, 6]
W1[184:240:6, 29] = [0, -3, -6, -12, -18, -24, -18, -12, -6, 0]
W1[244:300:6, 29] = [0, -3, -6, -12, -18, -24, -18, -12, -6, 0]
b1[29] = -3
W1[302:360:6, 30] = [3, 6, 12, 18, 24, 30, 24, 18, 12, 6]
W1[362:420:6, 30] = [3, 6, 12, 18, 24, 30, 24, 18, 12, 6]
W1[304:360:6, 30] = [0, -3, -6, -12, -18, -24, -18, -12, -6, 0]
W1[364:420:6, 30] = [0, -3, -6, -12, -18, -24, -18, -12, -6, 0]
b1[30] = -3
W1[422:480:6, 31] = [3, 6, 12, 18, 24, 30, 24, 18, 12, 6]
W1[482:540:6, 31] = [3, 6, 12, 18, 24, 30, 24, 18, 12, 6]
W1[424:480:6, 31] = [0, -3, -6, -12, -18, -24, -18, -12, -6, 0]
W1[484:540:6, 31] = [0, -3, -6, -12, -18, -24, -18, -12, -6, 0]
b1[31] = -3
# ===== layer 2 =====
# v0 threat in my row (OR of units 0-7); v1 threat above (8-15); v2 threat below (16-23)
W2[0:8, 0] = 6
b2[0] = 45
W2[8:16, 1] = 6
b2[1] = 45
W2[16:24, 2] = 6
b2[2] = 45
# v3: in band 2 and gold in band 1
W2[25, 3] = 6
W2[28, 3] = 6
b2[3] = -9
# v4: in band 3 and gold in band 1 or 2
W2[26, 4] = 18
W2[28, 4] = 6
W2[29, 4] = 6
b2[4] = -12
# v5: in band 4 and gold in band 1, 2 or 3
W2[27, 5] = 24
W2[28:31, 5] = 6
b2[5] = -12
# v6: in band 1 and gold in band 2, 3 or 4
W2[24, 6] = 24
W2[29:32, 6] = 6
b2[6] = -12
# v7: in band 2 and gold in band 3 or 4
W2[25, 7] = 18
W2[30, 7] = 6
W2[31, 7] = 6
b2[7] = -12
# v8: in band 3 and gold in band 4
W2[26, 8] = 6
W2[31, 8] = 6
b2[8] = -9
# ===== layer 3 =====
# up (action 2) = 30*[threat here] - 40*[threat above] + 15*[gold above]
W3[0, 2] = 15
W3[1, 2] = -20
W3[3:6, 2] = 7.5
b3[2] = 17.5
# down (action 4) = 30*[threat here] - 40*[threat below] + 15*[gold below]
W3[0, 4] = 15
W3[2, 4] = -20
W3[6:9, 4] = 7.5
b3[4] = 17.5
# no-op
b3[0] = 8
# never move sideways or fire
b3[1] = -50
b3[3] = -50
b3[5] = -50
```
