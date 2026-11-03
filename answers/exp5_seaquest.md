# Exp 5, Seaquest — one-shot network written by Claude Fable 5.1 (no tools, no simulation before scoring)

## Design

Facts used: only the submarine's FRONT cell collides and picks up divers; the front cell is in channel 0 but not in
channel 4, the back cell is in both (the trail channel includes the sub's back). Oxygen only drops below row 0, and
coming back to row 0 kills the sub unless it carries a diver. Channel 1 - channel 2 (danger minus reward) isolates
enemy bullets (fish and enemy subs are in both; divers only in channel 2). Along row 9, cell (9,2) of the gauge
channel is lit only by oxygen (oxygen >= 60); cell (9,8) is lit by the diver bar whenever at least one diver is on
board (or by oxygen >= 180, which cannot coincide with low oxygen).

Strategy: dive to row 1 and hunt there. If an enemy or diver in row 1 is on the left of my front cell, mix "left"
and "fire" 50/50 (a left move turns the sub to face left; bullets then fly at it); same on the right. A bullet in my
row -> drop to row 2 until the row is clear. A diver in row 2 -> go down and wait for it to cross my column, then
come back up. When oxygen is low and a diver is on board -> go up to the surface (refills oxygen, costs one diver);
at the surface -> dive again. Deeper rows (only reachable through sticky actions) -> go up.

Layer 1 (27 units): units 0-9 "my front is at column c AND something rewarding (enemy or diver) is left of c in
row 1" (built as tanh(6*(10*front_c + count_left - 10.5)) with front_c = ch0 - ch4 at (1,c)); units 10-19 the same
for the right; 20/21/22 = sub in row 0/1/2; 23 = oxygen low (cell (9,2) off); 24 = diver bar lit (cell (9,8));
25 = bullet in row 1 (danger - reward over row 1 >= 1); 26 = diver in row 2 (reward - danger over row 2 >= 1).
Layer 2: v0 = OR(units 0-9) target on the left; v1 = OR(units 10-19) target on the right; v2 = surface now
(low oxygen AND diver AND not at row 0); v3 = at row 0; v4 = row 1 with a bullet; v5 = row 2, no bullet, no diver;
v6 = deeper than row 2; v7 = row 1 with a diver in row 2 and no bullet.
Layer 3: left = 30*[v0] - 20, right = 30*[v1] - 20, fire = 30*[v0] + 30*[v1] - 20, up = 50*[v2] + 35*[v5] + 35*[v6]
- 20, down = 40*[v3] + 35*[v4] + 32*[v7] - 20, no-op = 0.

```edits
# ===== layer 1 =====
# units 0-9: front at (1,c) AND an enemy/diver (channel 2) left of c in row 1
W1[60, 0] = 60
W1[64, 0] = -60
b1[0] = -63
W1[66, 1] = 60
W1[70, 1] = -60
W1[62:68:6, 1] = 6
b1[1] = -63
W1[72, 2] = 60
W1[76, 2] = -60
W1[62:74:6, 2] = 6
b1[2] = -63
W1[78, 3] = 60
W1[82, 3] = -60
W1[62:80:6, 3] = 6
b1[3] = -63
W1[84, 4] = 60
W1[88, 4] = -60
W1[62:86:6, 4] = 6
b1[4] = -63
W1[90, 5] = 60
W1[94, 5] = -60
W1[62:92:6, 5] = 6
b1[5] = -63
W1[96, 6] = 60
W1[100, 6] = -60
W1[62:98:6, 6] = 6
b1[6] = -63
W1[102, 7] = 60
W1[106, 7] = -60
W1[62:104:6, 7] = 6
b1[7] = -63
W1[108, 8] = 60
W1[112, 8] = -60
W1[62:110:6, 8] = 6
b1[8] = -63
W1[114, 9] = 60
W1[118, 9] = -60
W1[62:116:6, 9] = 6
b1[9] = -63
# units 10-19: front at (1,c) AND an enemy/diver right of c in row 1
W1[60, 10] = 60
W1[64, 10] = -60
W1[68:120:6, 10] = 6
b1[10] = -63
W1[66, 11] = 60
W1[70, 11] = -60
W1[74:120:6, 11] = 6
b1[11] = -63
W1[72, 12] = 60
W1[76, 12] = -60
W1[80:120:6, 12] = 6
b1[12] = -63
W1[78, 13] = 60
W1[82, 13] = -60
W1[86:120:6, 13] = 6
b1[13] = -63
W1[84, 14] = 60
W1[88, 14] = -60
W1[92:120:6, 14] = 6
b1[14] = -63
W1[90, 15] = 60
W1[94, 15] = -60
W1[98:120:6, 15] = 6
b1[15] = -63
W1[96, 16] = 60
W1[100, 16] = -60
W1[104:120:6, 16] = 6
b1[16] = -63
W1[102, 17] = 60
W1[106, 17] = -60
W1[110:120:6, 17] = 6
b1[17] = -63
W1[108, 18] = 60
W1[112, 18] = -60
W1[116:120:6, 18] = 6
b1[18] = -63
W1[114, 19] = 60
W1[118, 19] = -60
b1[19] = -63
# units 20-22: sub in row 0 / 1 / 2 (channel 0)
W1[0:60:6, 20] = 6
b1[20] = -3
W1[60:120:6, 21] = 6
b1[21] = -3
W1[120:180:6, 22] = 6
b1[22] = -3
# unit 23: oxygen low = gauge cell (9,2) off
W1[557, 23] = -6
b1[23] = 3
# unit 24: diver bar lit = gauge cell (9,8) on
W1[593, 24] = 6
b1[24] = -3
# unit 25: bullet in row 1 = (danger - reward) over row 1 >= 1
W1[61:120:6, 25] = 6
W1[62:120:6, 25] = -6
b1[25] = -3
# unit 26: diver in row 2 = (reward - danger) over row 2 >= 1
W1[122:180:6, 26] = 6
W1[121:180:6, 26] = -6
b1[26] = -3
# ===== layer 2 =====
# v0: target on the left (OR of units 0-9); v1: target on the right (OR of 10-19)
W2[0:10, 0] = 6
b2[0] = 57
W2[10:20, 1] = 6
b2[1] = 57
# v2: surface now = oxygen low AND diver AND not at row 0
W2[23, 2] = 6
W2[24, 2] = 6
W2[20, 2] = -6
b2[2] = -15
# v3: at row 0
W2[20, 3] = 6
# v4: row 1 AND bullet in row 1
W2[21, 4] = 6
W2[25, 4] = 6
b2[4] = -9
# v5: row 2 AND no bullet in row 1 AND no diver in row 2
W2[22, 5] = 6
W2[25, 5] = -6
W2[26, 5] = -6
b2[5] = -15
# v6: deeper than row 2
W2[20, 6] = -6
W2[21, 6] = -6
W2[22, 6] = -6
b2[6] = -15
# v7: row 1 AND diver in row 2 AND no bullet in row 1
W2[21, 7] = 6
W2[26, 7] = 6
W2[25, 7] = -6
b2[7] = -15
# ===== layer 3 (logit = sum w*(h+1)/2 - 20) =====
# left (1)
W3[0, 1] = 15
b3[1] = -5
# right (3)
W3[1, 3] = 15
b3[3] = -5
# fire (5)
W3[0, 5] = 15
W3[1, 5] = 15
b3[5] = 10
# up (2)
W3[2, 2] = 25
W3[5, 2] = 17.5
W3[6, 2] = 17.5
b3[2] = 40
# down (4)
W3[3, 4] = 20
W3[4, 4] = 17.5
W3[7, 4] = 16
b3[4] = 33.5
# no-op (0)
b3[0] = 0
```
