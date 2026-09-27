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

# MARK-I Runtime

Rosenblatt's Mark-I, rebuilt as a pure-Ada runtime: a **K array engine**
(vectors, matrices, rank-N tensors, associative memory, learning transforms)
under a **stimulus → response ← feedback** learning loop.

- Language: **Ada 2007** (ISO/IEC 8652:2007, i.e. the Ada 2005 revision —
  compiled with `gnatmake -gnat05`).
- Verification: **raw `pragma Assert` only**. No Python, no test framework.
  Build with `-gnata`; if any assertion fails the program does not run.
- No OS calls in the engine. One codebase, three targets.

## Layout

| File | What |
|---|---|
| `src/mark1_k.ads/.adb` | K array engine: Vector/Matrix/Tensor, dot/matmul/outer, associative memory, Hebb + perceptron update, sigmoid family (scalar/vector/matrix + derivatives) |
| `src/mark1_rosenblatt.ads/.adb` | Perceptron: `Respond` (stimulus → response), `Feedback` (correction loop) |
| `src/mark1_gpu.ads/.adb` | RTX 4090 offload interface; host fallback body |
| `src/mark1_backprop.ads/.adb` | Backpropagating errors: 2-layer sigmoid MLP, SGD with full backward pass (XOR demo) |
| `src/mark1_net.ads/.adb` | Hand-rolled custom net: arbitrary depth, per-layer activations (linear/sigmoid/tanh/relu), full backprop, SGD (sine demo) |
| `src/mark1_cnn.ads/.adb` | Hand-rolled CNN: Conv2D + 2x2 max-pool + dense head, full backprop to kernels, numeric gradient check (bar-image demo) |
| `src/mark1_batchnorm.ads/.adb` | Batch normalization: per-feature normalize + learnable γ/β, running stats with momentum, train/infer paths, full backward through batch stats (XOR MLP demo) |
| `src/mark1_attention.ads/.adb` | Multi-head attention, hand-rolled: learned Wq/Wk/Wv per head + Wo, scaled dot-product semantic scores with stable softmax, full backprop, numeric gradient checks (library-search + token-routing demos) |
| `src/mark1_runtime.adb` | Lifecycle driver: INIT → TRAIN → EVALUATE → EDGE CASES → REPORT, 69 assertions |
| `src/mark1_bench.adb` | Benchmark driver: MatMul/MatVec scaling, sigmoid, batchnorm, CNN + MLP steps, attention steps (deterministic, checksummed) |

## Docs

| File | What |
|---|---|
| `docs/OVERVIEW.md` | Design rules, package map, lifecycle phases, repo layout |
| `docs/MATH.md` | Every equation the runtime implements, plus the verified demo numbers |
| `docs/BUILD_AND_VERIFY.md` | Toolchain, assertion philosophy, the s390x proot/qemu recipe, 4090 status |
| `docs/EDGE_CASES.md` | Catalog of degenerate inputs and the assertions that pin them |
| `docs/BENCHMARKS.md` | How to run the bench driver + measured throughput |

## Build & run

```sh
./build.sh
```

## Targets

1. **Native (this box)** — `gnatmake` as above, verified.
2. **IBM Z (s390x mainframe)** — the engine is pure Ada 2007 with no
   host dependencies, so it cross-builds cleanly. Verified 2026-09-27
   under `qemu-s390x` in a noble s390x chroot: same 69 assertions hold
   on the mainframe ISA (`file` confirms IBM S/390 ELF; numerics
   bit-identical to x86-64 to 14+ digits).
3. **RTX 4090 (NVIDIA Ada Lovelace)** — `Mark1_GPU` is the offload seam:
   `Sgemm`/`Sgemv` bind to CUDA kernels on the 4090 target; the host
   fallback body keeps every assertion identical on CPU.

## License

AGPL-3.0-or-later — see [LICENSE](LICENSE). Every source file carries
an SPDX header (`SPDX-License-Identifier: AGPL-3.0-or-later`).
Copyright (C) 2026 SnapKitty Collective.
