# Exp 5, Breakout — one-shot network written by Claude Fable 5.1 (no tools, no simulation before scoring)

## Design

Facts from the source that drive the design:
- The paddle moves first, then the ball. The ball is caught when it is about to enter row 9 and either
  `ball_x == pos` (branch 1: only the vertical direction flips, the ball keeps its horizontal direction) or
  `new_x == pos` (branch 2: the ball also reverses horizontally).
- Ball and paddle both move one column per step, so the paddle can never gain on the ball except at the side walls,
  where the ball spends an extra step in the wall column.
- Sticky actions (10%) repeat the PREVIOUS action, so any step where the required action changes is a hazard.
  The sticky-proof regime is "the paddle is one column behind the ball and moves in the ball's direction every
  step": every catch is then branch 1 with `pos` arriving exactly under the ball, the same action as before, and
  the regime reproduces itself after the bounce.

Policy (dx = ball column minus trail column, k = lead of the paddle in the ball's direction):
- dx = 0 (ball just reflected off a side wall, or episode start): move toward the centre (left if the ball is in the
  right half, right otherwise). After a wall this puts the paddle under the ball again.
- Moving, not the catch step: if the paddle is ahead of the ball (k >= 1) -> no-op (shed the lead); otherwise move
  with the ball (+dx).
- Catch step (ball in row 8 coming down, trail in row 7): k = 1 -> step back onto the ball column (-dx); k = 0 ->
  no-op; k <= -1 -> +dx. All three put the paddle exactly on the ball column so the ball keeps its horizontal
  direction (branch 1) and the paddle ends one behind: the sticky-proof regime.
- Just-bounced step (ball in row 8 with trail in row 8): if the paddle is ON the ball column the catch was branch 2
  (the ball has reversed although the trail still points the old way) -> move against the trail direction;
  otherwise follow the trail direction.
- Up, down and fire get a bias of -50 so only no-op / left / right are ever sampled.

Hidden units (gain 6, margins 0.5, so every unit is a +-1 logical value):
- h1[0] R = ball moving right (ball_x - trail_x - 0.5 > 0); h1[1] L = moving left; h1[2] A = paddle right of the ball
  (pos - ball_x - 0.5 > 0); h1[3] B = paddle left of the ball; h1[4] C = catch step (ball row 8 AND trail row 7);
  h1[5] V = just bounced (ball row 8 AND trail row 8); h1[6] H = ball in columns 5-9 (ball_x - 4.5 > 0).
  Columns are read out with weights 6*c on every cell of a column (one-hot inputs make this a linear read-out).
- h2[0..17]: conjunctions of the above (bias -6*(n-0.5) for n literals), one per case of the decision table.
- Output: logit_a = sum over the conjunctions that select action a of 10*(h2+1) = 20 if that case is active, 0
  otherwise (W3 = 10, b3 = 10 * number of selectors). Exactly one conjunction is active in every state.

```edits
# ---------- first layer ----------
# unit 0: R = moving right = tanh(6*(ball_x - trail_x - 0.5)); ball = channel 3, trail = channel 4
W1[9:600:60, 0] = 6
W1[15:600:60, 0] = 12
W1[21:600:60, 0] = 18
W1[27:600:60, 0] = 24
W1[33:600:60, 0] = 30
W1[39:600:60, 0] = 36
W1[45:600:60, 0] = 42
W1[51:600:60, 0] = 48
W1[57:600:60, 0] = 54
W1[10:600:60, 0] = -6
W1[16:600:60, 0] = -12
W1[22:600:60, 0] = -18
W1[28:600:60, 0] = -24
W1[34:600:60, 0] = -30
W1[40:600:60, 0] = -36
W1[46:600:60, 0] = -42
W1[52:600:60, 0] = -48
W1[58:600:60, 0] = -54
b1[0] = -3
# unit 1: L = moving left = tanh(6*(trail_x - ball_x - 0.5))
W1[9:600:60, 1] = -6
W1[15:600:60, 1] = -12
W1[21:600:60, 1] = -18
W1[27:600:60, 1] = -24
W1[33:600:60, 1] = -30
W1[39:600:60, 1] = -36
W1[45:600:60, 1] = -42
W1[51:600:60, 1] = -48
W1[57:600:60, 1] = -54
W1[10:600:60, 1] = 6
W1[16:600:60, 1] = 12
W1[22:600:60, 1] = 18
W1[28:600:60, 1] = 24
W1[34:600:60, 1] = 30
W1[40:600:60, 1] = 36
W1[46:600:60, 1] = 42
W1[52:600:60, 1] = 48
W1[58:600:60, 1] = 54
b1[1] = -3
# unit 2: A = paddle right of ball = tanh(6*(pos - ball_x - 0.5)); paddle = channel 0
W1[6:600:60, 2] = 6
W1[12:600:60, 2] = 12
W1[18:600:60, 2] = 18
W1[24:600:60, 2] = 24
W1[30:600:60, 2] = 30
W1[36:600:60, 2] = 36
W1[42:600:60, 2] = 42
W1[48:600:60, 2] = 48
W1[54:600:60, 2] = 54
W1[9:600:60, 2] = -6
W1[15:600:60, 2] = -12
W1[21:600:60, 2] = -18
W1[27:600:60, 2] = -24
W1[33:600:60, 2] = -30
W1[39:600:60, 2] = -36
W1[45:600:60, 2] = -42
W1[51:600:60, 2] = -48
W1[57:600:60, 2] = -54
b1[2] = -3
# unit 3: B = paddle left of ball = tanh(6*(ball_x - pos - 0.5))
W1[6:600:60, 3] = -6
W1[12:600:60, 3] = -12
W1[18:600:60, 3] = -18
W1[24:600:60, 3] = -24
W1[30:600:60, 3] = -30
W1[36:600:60, 3] = -36
W1[42:600:60, 3] = -42
W1[48:600:60, 3] = -48
W1[54:600:60, 3] = -54
W1[9:600:60, 3] = 6
W1[15:600:60, 3] = 12
W1[21:600:60, 3] = 18
W1[27:600:60, 3] = 24
W1[33:600:60, 3] = 30
W1[39:600:60, 3] = 36
W1[45:600:60, 3] = 42
W1[51:600:60, 3] = 48
W1[57:600:60, 3] = 54
b1[3] = -3
# unit 4: C = catch step = ball in row 8 AND trail in row 7
W1[483:540:6, 4] = 6
W1[424:480:6, 4] = 6
b1[4] = -9
# unit 5: V = just bounced = ball in row 8 AND trail in row 8
W1[483:540:6, 5] = 6
W1[484:540:6, 5] = 6
b1[5] = -9
# unit 6: H = ball in right half = tanh(6*(ball_x - 4.5))
W1[9:600:60, 6] = 6
W1[15:600:60, 6] = 12
W1[21:600:60, 6] = 18
W1[27:600:60, 6] = 24
W1[33:600:60, 6] = 30
W1[39:600:60, 6] = 36
W1[45:600:60, 6] = 42
W1[51:600:60, 6] = 48
W1[57:600:60, 6] = 54
b1[6] = -27
# ---------- second layer: conjunctions (R=0 L=1 A=2 B=3 C=4 V=5 H=6) ----------
# u0: R & notC & notV & notA -> right
W2[0, 0] = 6
W2[4, 0] = -6
W2[5, 0] = -6
W2[2, 0] = -6
b2[0] = -21
# u1: R & notC & notV & A -> no-op
W2[0, 1] = 6
W2[4, 1] = -6
W2[5, 1] = -6
W2[2, 1] = 6
b2[1] = -21
# u2: L & notC & notV & notB -> left
W2[1, 2] = 6
W2[4, 2] = -6
W2[5, 2] = -6
W2[3, 2] = -6
b2[2] = -21
# u3: L & notC & notV & B -> no-op
W2[1, 3] = 6
W2[4, 3] = -6
W2[5, 3] = -6
W2[3, 3] = 6
b2[3] = -21
# u4: R & C & A -> left (step back onto the ball column)
W2[0, 4] = 6
W2[4, 4] = 6
W2[2, 4] = 6
b2[4] = -15
# u5: R & C & notA & notB -> no-op
W2[0, 5] = 6
W2[4, 5] = 6
W2[2, 5] = -6
W2[3, 5] = -6
b2[5] = -21
# u6: R & C & B -> right
W2[0, 6] = 6
W2[4, 6] = 6
W2[3, 6] = 6
b2[6] = -15
# u7: L & C & B -> right (step back onto the ball column)
W2[1, 7] = 6
W2[4, 7] = 6
W2[3, 7] = 6
b2[7] = -15
# u8: L & C & notA & notB -> no-op
W2[1, 8] = 6
W2[4, 8] = 6
W2[2, 8] = -6
W2[3, 8] = -6
b2[8] = -21
# u9: L & C & A -> left
W2[1, 9] = 6
W2[4, 9] = 6
W2[2, 9] = 6
b2[9] = -15
# u10: R & V & notA & notB -> left (branch-2 bounce: ball really moves left now)
W2[0, 10] = 6
W2[5, 10] = 6
W2[2, 10] = -6
W2[3, 10] = -6
b2[10] = -21
# u11: R & V & B -> right
W2[0, 11] = 6
W2[5, 11] = 6
W2[3, 11] = 6
b2[11] = -15
# u12: R & V & A -> right
W2[0, 12] = 6
W2[5, 12] = 6
W2[2, 12] = 6
b2[12] = -15
# u13: L & V & notA & notB -> right (branch-2 bounce: ball really moves right now)
W2[1, 13] = 6
W2[5, 13] = 6
W2[2, 13] = -6
W2[3, 13] = -6
b2[13] = -21
# u14: L & V & A -> left
W2[1, 14] = 6
W2[5, 14] = 6
W2[2, 14] = 6
b2[14] = -15
# u15: L & V & B -> left
W2[1, 15] = 6
W2[5, 15] = 6
W2[3, 15] = 6
b2[15] = -15
# u16: notR & notL & H -> left (ball at a wall / start, ball on the right)
W2[0, 16] = -6
W2[1, 16] = -6
W2[6, 16] = 6
b2[16] = -15
# u17: notR & notL & notH -> right
W2[0, 17] = -6
W2[1, 17] = -6
W2[6, 17] = -6
b2[17] = -15
# ---------- output layer: logit = 10*(h2+1) summed over selectors ----------
# right (action 3): u0 u6 u7 u11 u12 u13 u17
W3[0, 3] = 10
W3[6, 3] = 10
W3[7, 3] = 10
W3[11, 3] = 10
W3[12, 3] = 10
W3[13, 3] = 10
W3[17, 3] = 10
b3[3] = 70
# left (action 1): u2 u4 u9 u10 u14 u15 u16
W3[2, 1] = 10
W3[4, 1] = 10
W3[9, 1] = 10
W3[10, 1] = 10
W3[14, 1] = 10
W3[15, 1] = 10
W3[16, 1] = 10
b3[1] = 70
# no-op (action 0): u1 u3 u5 u8
W3[1, 0] = 10
W3[3, 0] = 10
W3[5, 0] = 10
W3[8, 0] = 10
b3[0] = 40
# never sample up / down / fire
b3[2] = -50
b3[4] = -50
b3[5] = -50
```
