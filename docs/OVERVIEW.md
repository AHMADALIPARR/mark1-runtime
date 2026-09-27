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

# MARK-I Runtime — Overview

Rosenblatt's Mark-I, rebuilt as a pure-Ada neural runtime. A **K array
engine** (vectors, matrices, rank-N tensors, associative memory, learning
transforms) sits under a **stimulus → response ← feedback** learning loop,
with hand-rolled backpropagating networks stacked on top: a 2-layer
sigmoid MLP, an arbitrary-depth custom net, a Conv2D CNN, and batch
normalization.

## Design rules

1. **Ada 2007 only** (ISO/IEC 8652:2007 — the Ada 2005 revision, compiled
   with `gnatmake -gnat05`). No Python, no frameworks, no FFI in the
   engine.
2. **Arrays only.** Every structure is a constrained or unconstrained Ada
   array. No access types in the hot path, no address tricks — this is
   what lets the same source cross-compile to IBM Z / s390x unchanged.
3. **Raw assertions, no test framework.** Verification is `pragma Assert`
   inside the drivers. Build with `-gnata`; if any assertion fails the
   program does not run. The assertion count is the test report.
4. **Deterministic.** No RNG anywhere. Every initialization is a closed
   formula (fan-in scaled), every dataset is fixed. Bit-identical results
   across x86-64 and s390x.
5. **Overflow-safe numerics.** The sigmoid uses the stable branch form;
   batchnorm guards variance with epsilon; edge cases pin exact values
   (see `docs/EDGE_CASES.md`).

## Packages

| Package | Role | Key API |
|---|---|---|
| `Mark1_K` | K array engine | `Vector`, `Matrix`, `Tensor`, `Dot`, `Norm2`, `Axpy`, `Matvec`, `Matmul`, `Outer`, `Transpose`, `Assoc_Memory`, `Hebb`, `Perceptron_Update`, `Sigmoid`/`Sigmoid_D` (scalar, vector, matrix) |
| `Mark1_Rosenblatt` | Perceptron + feedback loop | `Respond`, `Feedback`, `Weights`, `Bias_Of` |
| `Mark1_Backprop` | 2-layer sigmoid MLP, full backprop | `Train`, `Forward` (XOR demo in driver) |
| `Mark1_Net` | Arbitrary-depth net (≤8 layers, ≤32 wide), linear/sigmoid/tanh/relu, SGD | `Make_Net`, `Forward`, `Train_Sample`, `Activate`, `Activate_D` |
| `Mark1_CNN` | Valid Conv2D + ReLU, 2×2 max-pool with argmax masks, filter-major flatten, sigmoid dense head, backprop to kernels | `Conv_Forward`, `MaxPool_Forward`, `Flatten`, `Dense_Forward`/`Dense_Backward`, `Conv_Gradients`, `Conv_Backward` |
| `Mark1_BatchNorm` | Per-feature batch normalization, learnable γ/β, running stats (momentum 0.9), train/infer paths | `BN_Forward_Train`, `BN_Forward_Infer`, `BN_Gradients`, `BN_Backward` |
| `Mark1_Attention` | Multi-head scaled dot-product attention: learned Wq/Wk/Wv per head + Wo, softmax semantic scores, full backprop | `Attn_Forward`, `Attn_Scores`, `Attn_Gradients`, `Attn_Backward` |
| `Mark1_GPU` | RTX 4090 offload seam (`Sgemm`/`Sgemv`); CPU fallback body | `Sgemm`, `Sgemv` |

`Mark1_Backprop.Sigmoid` is a renaming of `Mark1_K.Sigmoid` — one
implementation, no drift.

## Lifecycle driver (`src/mark1_runtime.adb`)

One procedure, five phases, each check a raw assertion:

1. **INIT** — engine checks: dot/norm/axpy, matvec/matmul, outer,
   transpose, associative memory store/recall, Hebb, perceptron update,
   sigmoid family values.
2. **TRAIN: backprop MLP** — 2-layer sigmoid net learns XOR by full
   backpropagation.
3. **TRAIN: custom net** — arbitrary-depth net fits sin(x) (MSE ~2.3e-7).
4. **TRAIN: CNN** — Conv2D + pool + dense learns 8 bar images; numeric
   gradient check guards `Conv_Backward`.
5. **TRAIN: batch normalization** — forward statistics, γ/β, running
   stats, inference path, numeric gradient checks on dX and dγ, and a
   2-4-1 tanh MLP with BN on the hidden activations learning XOR,
   evaluated through the inference path.
6. **EDGE CASES** — saturation, kinks, degenerate batches, frozen
   parameters, degenerate linear algebra, deterministic pool ties.
7. **REPORT** — prints the held-assertion count and re-asserts the total.

## Layout

```
src/
  mark1_k.ads/.adb          K array engine
  mark1_rosenblatt.ads/.adb perceptron + feedback
  mark1_backprop.ads/.adb   2-layer sigmoid MLP
  mark1_net.ads/.adb        arbitrary-depth net
  mark1_cnn.ads/.adb        convolutional net
  mark1_batchnorm.ads/.adb  batch normalization
  mark1_attention.ads/.adb  multi-head attention
  mark1_gpu.ads/.adb        4090 offload seam (CPU fallback)
  mark1_runtime.adb         lifecycle driver (assertions)
  mark1_bench.adb           benchmark driver (timing)
docs/
  OVERVIEW.md  MATH.md  BUILD_AND_VERIFY.md  EDGE_CASES.md  BENCHMARKS.md
build.sh
```

Further reading: `docs/MATH.md` for the equations behind every block,
`docs/BUILD_AND_VERIFY.md` for the toolchain and the s390x recipe,
`docs/EDGE_CASES.md` for the edge-case catalog, `docs/BENCHMARKS.md`
for measured throughput.
