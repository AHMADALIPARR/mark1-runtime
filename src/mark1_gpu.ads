--  SPDX-License-Identifier: AGPL-3.0-or-later
--
--  mark1-runtime
--  Copyright (C) 2026 SnapKitty Collective
--
--  This program is free software: you can redistribute it and/or modify
--  it under the terms of the GNU Affero General Public License as published
--  by the Free Software Foundation, either version 3 of the License, or
--  (at your option) any later version.
--
--  This program is distributed in the hope that it will be useful,
--  but WITHOUT ANY WARRANTY; without even the implied warranty of
--  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
--  GNU Affero General Public License for more details.
--
--  You should have received a copy of the GNU Affero General Public License
--  along with this program.  If not, see <https://www.gnu.org/licenses/>.
--

--  MARK-I Runtime -- RTX 4090 offload interface (spec)
--
--  Target: NVIDIA Ada Lovelace (RTX 4090).
--  On the 4090 target these specs bind to CUDA kernels through the vendor
--  toolchain (cuBLAS Sgemm / one-thread-block-per-row Sgemv). The host
--  fallback body keeps the runtime whole on any CPU -- including IBM Z --
--  so the K engine stays one codebase and the offload is a link-time choice.

with Mark1_K; use Mark1_K;

package Mark1_GPU is

   pragma Assertion_Policy (Check);

   GPU_Present : constant Boolean := False;
   --  Set True by the 4090 target build; selects kernel vs host path.

   procedure Sgemm (A, B : Matrix; C : out Matrix);
   --  C := A * B. 4090 target: cuBLAS / custom kernel.

   procedure Sgemv (M : Matrix; V : Vector; R : out Vector);
   --  R := M * V. 4090 target: kernel launch over rows.

end Mark1_GPU;
