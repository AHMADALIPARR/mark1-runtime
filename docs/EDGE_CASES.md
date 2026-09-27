<!--
SPDX-License-Identifier: AGPL-3.0-or-later

mark1-runtime
Copyright (C) 2026 SnapKitty Collective

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
-->

# MARK-I Runtime — Edge cases

Every degenerate input the runtime pins down with an assertion. Each row
names the edge, the guaranteed behavior, and the assertion that proves it
in `src/mark1_runtime.adb` (phase `EDGE CASES`).

| # | Edge | Behavior | Assertion |
|---|---|---|---|
| 1 | `sigmoid(+1000)` | Pins to `1.0` **exactly** — the stable branch never evaluates `e^(+1000)` | `edge: sigmoid(+1000) = 1.0 exactly` |
| 2 | `sigmoid(−1000)` | Pins to `0.0` **exactly** — `e^(−1000)` underflows cleanly | `edge: sigmoid(-1000) = 0.0 exactly` |
| 3 | `Sigmoid_D(±1000)` | `0.0` — flat tails, no NaN from `s·(1−s)` | `edge: sigmoid derivative vanishes at extremes` |
| 4 | ReLU at the kink | `ReLU(0) = 0`, `ReLU(−2.5) = 0` — non-positive side is hard zero | `edge: relu kink at zero` |
| 5 | `Activate_D(Sigmoid_A, 0.5)` | `0.25` — derivative from the activated output, no re-evaluation | `edge: sigmoid derivative from activated output` |
| 6 | `tanh(50)` | Saturates to `1.0` | `edge: tanh saturates` |
| 7 | BatchNorm, batch of 1 | Variance is 0, `x̂ = 0`, output collapses to β (here `0.0`); running stats still update (`run_μ = 0.1·x`, `run_σ² = 0.9`) | `edge: bn batch-of-1 collapses to beta`, `edge: bn batch-of-1 running stats` |
| 8 | BatchNorm, constant feature | Zero variance across the batch → output is β, no division blowup (ε guard) | `edge: bn constant feature collapses to beta` |
| 9 | `BN_Backward` with `η = 0.0` | Gradients are computed, γ/β are untouched | `edge: bn eta=0 freezes gamma/beta` |
| 10 | Zero vector | `Dot = 0.0`, `Norm2 = 0.0` | `edge: zero vector dot/norm` |
| 11 | `Matmul` 1×1 | `[[2]]·[[3]] = [[6]]` — degenerate GEMM path | `edge: matmul 1x1` |
| 12 | `Axpy` with `α = 0` | `y` is bit-identical afterwards | `edge: axpy alpha=0 leaves y unchanged` |
| 13 | Max-pool tie, uniform map | Deterministic — the first maximum wins (`>` keeps the earliest cell); argmax mask is one-hot at cell 0 | `edge: conv zero-kernel tie pools deterministically` |
| 14 | Attention, single token | One key means one score: `A(1,1) = 1.0` **exactly** — softmax over a single logit is the identity | `edge: attn single token weight is 1.0` |

## Notes

- **Batch-of-1 batchnorm** is the sharpest edge: with `m = 1` the batch
  variance is exactly 0, so `x̂ = 0/√ε = 0` and the layer outputs β
  regardless of input. Gradients still flow (to γ/β and, correctly,
  zero to x). The runtime asserts the forward collapse and the running
  stat values (`0.9·1 + 0.1·0 = 0.9` for variance).
- **The pool tie** documents a real behavioral choice: `if V > Best`
  (strict) means ties keep the *first* cell, and the mask marks exactly
  that cell. Backprop through a tie therefore routes the full delta to
  one cell — deterministic, reproducible, asserted.
- **`Make_Conv2D` seeds kernels** (`0.1·(f+i+j) − 0.2`); the tie test
  zeroes them explicitly via `Set_Kernel` first. Deterministic init is
  itself an edge-case policy: no RNG, no hidden state.
- Adding a new edge: write the degenerate input, assert the exact
  behavior, bump the `pragma Assert (Passed = N)` total in the REPORT
  phase, and add a row here.
