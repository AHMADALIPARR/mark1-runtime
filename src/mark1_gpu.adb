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

--  MARK-I Runtime -- RTX 4090 offload interface (host fallback body)
--
--  Compiles and runs on CPU (x86-64, IBM Z s390x, anything with GNAT).
--  The 4090 build replaces this body with CUDA kernel bindings behind the
--  same spec; all assertions stay identical.

package body Mark1_GPU is

   procedure Sgemm (A, B : Matrix; C : out Matrix) is
   begin
      pragma Assert (A'Length (2) = B'Length (1));
      pragma Assert (C'First (1) = A'First (1) and C'Last (1) = A'Last (1));
      pragma Assert (C'First (2) = B'First (2) and C'Last (2) = B'Last (2));
      C := Matmul (A, B);
   end Sgemm;

   procedure Sgemv (M : Matrix; V : Vector; R : out Vector) is
   begin
      pragma Assert (M'Length (2) = V'Length);
      pragma Assert (R'Length = M'Length (1));
      R := Matvec (M, V);
   end Sgemv;

end Mark1_GPU;
