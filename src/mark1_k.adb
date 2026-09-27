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

--  MARK-I Runtime -- K Array Engine (body)

with Ada.Unchecked_Deallocation;

package body Mark1_K is

   procedure Free_Dims is new Ada.Unchecked_Deallocation (Dim_Array, Dim_Access);
   procedure Free_Data is new Ada.Unchecked_Deallocation (Data_Array, Data_Access);

   function Element_Count (D : Dim_Array) return Positive is
      N : Positive := 1;
   begin
      pragma Assert (D'Length > 0);
      for I in D'Range loop
         N := N * D (I);
      end loop;
      return N;
   end Element_Count;

   function Make_Tensor (D : Dim_Array) return Tensor is
      T : Tensor;
   begin
      pragma Assert (D'Length > 0);
      T.Rank := D'Length;
      T.Dims := new Dim_Array'(D);
      T.Data := new Data_Array'(1 .. Element_Count (D) => 0.0);
      return T;
   end Make_Tensor;

   procedure Free (T : in out Tensor) is
   begin
      Free_Dims (T.Dims);
      Free_Data (T.Data);
   end Free;

   function Lin (T : Tensor; Idx : Dim_Array) return Positive is
      L      : Positive := 1;
      Stride : Positive := 1;
      K      : Positive;
   begin
      pragma Assert (T.Dims /= null and then T.Data /= null);
      pragma Assert (Idx'Length = T.Rank);
      for I in reverse 1 .. T.Rank loop
         K := Idx (Idx'First + I - 1);
         pragma Assert (K in 1 .. T.Dims (I));
         L := L + (K - 1) * Stride;
         Stride := Stride * T.Dims (I);
      end loop;
      pragma Assert (L in T.Data'Range);
      return L;
   end Lin;

   function Get (T : Tensor; Idx : Dim_Array) return Real is
   begin
      return T.Data (Lin (T, Idx));
   end Get;

   procedure Set (T : Tensor; Idx : Dim_Array; V : Real) is
   begin
      T.Data (Lin (T, Idx)) := V;
   end Set;

   function Dot (A, B : Vector) return Real is
      S : Real := 0.0;
   begin
      pragma Assert (A'Length = B'Length);
      for I in A'Range loop
         S := S + A (I) * B (B'First + I - A'First);
      end loop;
      return S;
   end Dot;

   function Norm2 (A : Vector) return Real is
   begin
      return Math.Sqrt (Dot (A, A));
   end Norm2;

   procedure Axpy (Alpha : Real; X : Vector; Y : in out Vector) is
   begin
      pragma Assert (X'Length = Y'Length);
      for I in X'Range loop
         Y (Y'First + I - X'First) :=
           Y (Y'First + I - X'First) + Alpha * X (I);
      end loop;
   end Axpy;

   function Matvec (M : Matrix; V : Vector) return Vector is
      R : Vector (M'Range (1)) := (others => 0.0);
   begin
      pragma Assert (M'Length (2) = V'Length);
      for I in M'Range (1) loop
         for J in M'Range (2) loop
            R (I) := R (I) + M (I, J) * V (V'First + J - M'First (2));
         end loop;
      end loop;
      return R;
   end Matvec;

   function Matmul (A, B : Matrix) return Matrix is
      C : Matrix (A'Range (1), B'Range (2)) := (others => (others => 0.0));
   begin
      pragma Assert (A'Length (2) = B'Length (1));
      for I in A'Range (1) loop
         for K in A'Range (2) loop
            for J in B'Range (2) loop
               C (I, J) :=
                 C (I, J) + A (I, K) * B (B'First (1) + K - A'First (2), J);
            end loop;
         end loop;
      end loop;
      return C;
   end Matmul;

   function Outer (A, B : Vector) return Matrix is
      R : Matrix (A'Range, B'Range);
   begin
      for I in A'Range loop
         for J in B'Range loop
            R (I, J) := A (I) * B (J);
         end loop;
      end loop;
      return R;
   end Outer;

   function Transpose (M : Matrix) return Matrix is
      T : Matrix (M'Range (2), M'Range (1));
   begin
      for I in M'Range (1) loop
         for J in M'Range (2) loop
            T (J, I) := M (I, J);
         end loop;
      end loop;
      return T;
   end Transpose;

   procedure Store (M : in out Assoc_Memory; Key, Val : Vector) is
   begin
      pragma Assert (Key'Length = M.Key_Dim);
      pragma Assert (Val'Length = M.Val_Dim);
      pragma Assert (M.N < M.Capacity);
      M.N := M.N + 1;
      for J in 1 .. M.Key_Dim loop
         M.Keys (M.N, J) := Key (Key'First + J - 1);
      end loop;
      for J in 1 .. M.Val_Dim loop
         M.Vals (M.N, J) := Val (Val'First + J - 1);
      end loop;
   end Store;

   function Stored (M : Assoc_Memory) return Natural is
   begin
      return M.N;
   end Stored;

   function Recall (M : Assoc_Memory; Query : Vector) return Vector is
      Best   : Positive := 1;
      Best_S : Real     := 0.0;
      S      : Real;
   begin
      pragma Assert (M.N > 0);
      pragma Assert (Query'Length = M.Key_Dim);
      for K in 1 .. M.N loop
         S := 0.0;
         for J in 1 .. M.Key_Dim loop
            S := S + M.Keys (K, J) * Query (Query'First + J - 1);
         end loop;
         if K = 1 or else S > Best_S then
            Best_S := S;
            Best   := K;
         end if;
      end loop;
      declare
         R : Vector (1 .. M.Val_Dim);
      begin
         for J in 1 .. M.Val_Dim loop
            R (J) := M.Vals (Best, J);
         end loop;
         return R;
      end;
   end Recall;

   procedure Hebb (W : in out Matrix; X, Y : Vector; Eta : Real) is
   begin
      pragma Assert (W'Length (1) = Y'Length);
      pragma Assert (W'Length (2) = X'Length);
      pragma Assert (Eta >= 0.0);
      for I in W'Range (1) loop
         for J in W'Range (2) loop
            W (I, J) :=
              W (I, J) + Eta
              * Y (Y'First + I - W'First (1))
              * X (X'First + J - W'First (2));
         end loop;
      end loop;
   end Hebb;

   procedure Perceptron_Update
     (W       : in out Vector;
      Bias    : in out Real;
      X       : Vector;
      Desired : Integer;
      Eta     : Real)
   is
      S    : Real := 0.0;
      Resp : Integer;
   begin
      pragma Assert (W'Length = X'Length);
      pragma Assert (Desired = 1 or Desired = -1);
      pragma Assert (Eta > 0.0);
      for I in W'Range loop
         S := S + W (I) * X (X'First + I - W'First);
      end loop;
      S := S + Bias;
      if S >= 0.0 then
         Resp := 1;
      else
         Resp := -1;
      end if;
      if Resp /= Desired then
         for I in W'Range loop
            W (I) := W (I) + Eta * Real (Desired) * X (X'First + I - W'First);
         end loop;
         Bias := Bias + Eta * Real (Desired);
      end if;
   end Perceptron_Update;

   --------------------------
   --  Activation functions  --
   --------------------------

   function Sigmoid (X : Real) return Real is
   begin
      --  Overflow-safe form: Exp never sees a large positive argument.
      if X >= 0.0 then
         return 1.0 / (1.0 + Math.Exp (-X));
      else
         declare
            E : constant Real := Math.Exp (X);
         begin
            return E / (1.0 + E);
         end;
      end if;
   end Sigmoid;

   function Sigmoid_D (X : Real) return Real is
      S : constant Real := Sigmoid (X);
   begin
      return S * (1.0 - S);
   end Sigmoid_D;

   function Sigmoid (V : Vector) return Vector is
      R : Vector (V'Range);
   begin
      for I in V'Range loop
         R (I) := Sigmoid (V (I));
      end loop;
      return R;
   end Sigmoid;

   function Sigmoid (M : Matrix) return Matrix is
      R : Matrix (M'Range (1), M'Range (2));
   begin
      for I in M'Range (1) loop
         for J in M'Range (2) loop
            R (I, J) := Sigmoid (M (I, J));
         end loop;
      end loop;
      return R;
   end Sigmoid;

   function Sigmoid_D (V : Vector) return Vector is
      R : Vector (V'Range);
   begin
      for I in V'Range loop
         R (I) := Sigmoid_D (V (I));
      end loop;
      return R;
   end Sigmoid_D;

   function Sigmoid_D (M : Matrix) return Matrix is
      R : Matrix (M'Range (1), M'Range (2));
   begin
      for I in M'Range (1) loop
         for J in M'Range (2) loop
            R (I, J) := Sigmoid_D (M (I, J));
         end loop;
      end loop;
      return R;
   end Sigmoid_D;

end Mark1_K;
