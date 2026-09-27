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

--  MARK-I Runtime -- K Array Engine (spec)
--  Pure Ada 2007 (ISO/IEC 8652:2007, the Ada 2005 revision). No Python.
--  No OS calls. Verification is raw pragma Assert only.

with Ada.Numerics.Generic_Elementary_Functions;

package Mark1_K is

   pragma Assertion_Policy (Check);

   type Real is digits 15;

   package Math is new Ada.Numerics.Generic_Elementary_Functions (Real);

   type Vector is array (Positive range <>) of Real;
   type Matrix is array (Positive range <>, Positive range <>) of Real;

   --------------------
   --  Rank-N tensors  --
   --------------------
   --  Row-major, flat storage + shape vector. This is the array core that
   --  also cross-compiles to IBM Z (s390x): no address tricks, no C.

   type Dim_Array  is array (Positive range <>) of Positive;
   type Dim_Access is access Dim_Array;
   type Data_Array  is array (Positive range <>) of Real;
   type Data_Access is access Data_Array;

   type Tensor is record
      Rank : Positive;
      Dims : Dim_Access;
      Data : Data_Access;
   end record;

   function Element_Count (D : Dim_Array) return Positive;
   function Make_Tensor (D : Dim_Array) return Tensor;
   procedure Free (T : in out Tensor);
   function Lin (T : Tensor; Idx : Dim_Array) return Positive;
   function Get (T : Tensor; Idx : Dim_Array) return Real;
   procedure Set (T : Tensor; Idx : Dim_Array; V : Real);

   -------------------
   --  Level 1/2/3  --
   -------------------

   function Dot (A, B : Vector) return Real;
   function Norm2 (A : Vector) return Real;
   procedure Axpy (Alpha : Real; X : Vector; Y : in out Vector);
   function Matvec (M : Matrix; V : Vector) return Vector;
   function Matmul (A, B : Matrix) return Matrix;
   function Outer (A, B : Vector) return Matrix;
   function Transpose (M : Matrix) return Matrix;

   ------------------------
   --  Associative memory  --
   ------------------------

   type Real_Bank is array (Positive range <>, Positive range <>) of Real;

   type Assoc_Memory (Key_Dim, Val_Dim, Capacity : Positive) is private;

   procedure Store (M : in out Assoc_Memory; Key, Val : Vector);
   function Stored (M : Assoc_Memory) return Natural;
   function Recall (M : Assoc_Memory; Query : Vector) return Vector;

   -------------------------
   --  Learning transforms  --
   -------------------------

   procedure Hebb (W : in out Matrix; X, Y : Vector; Eta : Real);
   procedure Perceptron_Update
     (W     : in out Vector;
      Bias  : in out Real;
      X     : Vector;
      Desired : Integer;
      Eta   : Real);

   --------------------------
   --  Activation functions  --
   --------------------------

   function Sigmoid (X : Real) return Real;
   function Sigmoid (V : Vector) return Vector;
   function Sigmoid (M : Matrix) return Matrix;
   function Sigmoid_D (X : Real) return Real;
   function Sigmoid_D (V : Vector) return Vector;
   function Sigmoid_D (M : Matrix) return Matrix;

private

   type Assoc_Memory (Key_Dim, Val_Dim, Capacity : Positive) is record
      N    : Natural := 0;
      Keys : Real_Bank (1 .. Capacity, 1 .. Key_Dim);
      Vals : Real_Bank (1 .. Capacity, 1 .. Val_Dim);
   end record;

end Mark1_K;
