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

# MARK-I Runtime — Build & verify

## Native build (x86-64)

Prerequisite: GNAT with Ada 2005 support (`-gnat05`). This box uses
GNAT 13.3.0 from apt.

```sh
./build.sh
# which runs:
gnatmake -gnat05 -gnata -gnatW8 -O2 src/mark1_runtime.adb -o mark1_runtime
./mark1_runtime
```

Flags: `-gnat05` (Ada 2005 revision), `-gnata` (assertions on — the
whole verification story), `-gnatW8` (UTF-8), `-O2`.

Expected: zero warnings/errors, then `MARK-I runtime complete:  58
assertions held.` If any assertion fails, the program raises
`Assertion_Error` and stops — a failed check is never silent.

Benchmarks:

```sh
gnatmake -gnat05 -O2 src/mark1_bench.adb -o mark1_bench
./mark1_bench
```

## What the assertions mean

There is no test framework. Each `Check (cond, "name")` in
`src/mark1_runtime.adb` is a raw `pragma Assert`: the program *proves
itself or it does not run*. The final line re-asserts the total count,
so adding a check without bumping the total fails the build of trust —
you cannot silently drop a check.

Phase map: INIT (engine) → TRAIN backprop MLP → TRAIN custom net →
TRAIN CNN → TRAIN batchnorm → EDGE CASES → REPORT.

## IBM Z / s390x

The engine is pure Ada 2007 with no host dependencies (no OS calls, no
address tricks, arrays only), so it cross-builds unchanged.

Verified 2026-09-27 under `qemu-s390x` in an Ubuntu Noble s390x rootfs
(`proot`, since `binfmt_misc` registration is blocked on this box —
this is qemu-user emulation, not physical IBM Z hardware):

```sh
sudo proot -q /usr/bin/qemu-s390x-static \
  -r ~/workspace/mark1-runtime/zroot \
  -b /dev -b /proc \
  -w /root/mark1/src \
  /usr/bin/gnatmake -gnat05 -gnata -gnatW8 -O2 mark1_runtime
```

Result: zero warnings/errors, `file` confirms an ELF 64-bit MSB IBM
S/390 executable, **all assertions held**, and the XOR error, sine MSE,
and CNN gradient values were **bit-identical to x86-64** (14+ digits).
No portability source changes were needed.

## RTX 4090 (NVIDIA Ada Lovelace)

Status: **not implemented**. `Mark1_GPU` is an offload seam
(`Sgemm`/`Sgemv`) with a CPU fallback body — every assertion runs on
CPU today. Real CUDA/cuBLAS execution needs Ada bindings (or a thin C
ABI bridge), device allocation/transfers, and a 4090 build
configuration, verified on actual NVIDIA hardware. The user's "no
python just Ada and Arrays" constraint stands; introducing a C shim is
a tradeoff to decide explicitly before it happens.

## Toolchain recovery note

2026-09-27: apt's HTTP fetcher stalled on this box while the package
index was fetched fine with curl — the two use different code paths.
The fix was pulling the Noble universe `Packages.xz` directly with
curl, resolving the exact GNAT `.deb` URLs, and `dpkg -i`'ing them;
once the indexes completed, `apt-get install -f -y` finished the job.
Durable lesson: when diagnosing apt, check `/var/lib/apt/lists/`, not
only `/var/lib/apt/lists/partial/`.
