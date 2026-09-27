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

# MARK-I Runtime — Benchmarks

Driver: `src/mark1_bench.adb`. Deterministic data (closed-form fills, no
RNG); every timed loop ends in a printed checksum so the optimizer
cannot delete the work.

```sh
gnatmake -gnat05 -O2 src/mark1_bench.adb -o mark1_bench
./mark1_bench
```

## Results — 2026-09-27, x86-64, GNAT 13.3.0, `-O2`

Naive triple-loop Ada, no blocking, no SIMD intrinsics — these numbers
are the honest baseline, not a tuned peak.

### Level 3 (MatMul, 2N³ flops)

| N | ms/op | GFLOPS |
|---|---|---|
| 16 | 0.0061 | 1.34 |
| 32 | 0.0957 | 0.68 |
| 64 | 0.2571 | 2.04 |
| 128 | 2.2678 | 1.85 |

(The N=32 dip is cache/TLB noise on a shared box; rerun to confirm on
quiet hardware.)

### Level 2 / activations

| Op | ms/op |
|---|---|
| Matvec 256×256 | 0.3751 |
| Sigmoid, 4096-vector | 0.0345 |

### BatchNorm (64 features)

| Batch | forward ms | backward ms |
|---|---|---|
| 32 | 0.0133 | 0.0415 |
| 1 (edge case) | 0.0012 | 0.0025 |

Backward costs ~3× forward (three passes: stats, dvar/dmu, dx).

### Networks (one full step)

| Net | forward ms | backward ms |
|---|---|---|
| CNN 16×16, 4 filters 3×3, pool→dense | 0.0265 | 0.0273 |
| MLP 1-16-16-1, one SGD `Train_Sample` | 0.0025 (total) | — |
| Attention, 2 heads, d=32, 16 tokens | 0.2096 | 0.8588 |
| Attention, 4 heads, d=64, 32 tokens | 1.7373 | 7.0500 |

## Reading these numbers

- Everything above is single-threaded scalar Ada. The 4090 seam
  (`Mark1_GPU.Sgemm`/`Sgemv`) exists precisely because the Level-3
  path is where a GPU earns its keep: at ~2 GFLOPS here, a 4090's
  cuBLAS SGEMM would be three orders of magnitude faster.
- The BN batch-of-1 row doubles as the perf side of the edge case in
  `docs/EDGE_CASES.md`: the degenerate path is also the cheapest.
- Re-run on your hardware before quoting — `mark1_bench` prints its
  own table; paste it here when the numbers move.
