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

--  MARK-I Runtime -- Convolutional neural network (body)

package body Mark1_CNN is

   function Out_H (C : Conv2D) return Positive is
   begin
      return C.In_H - C.KH + 1;
   end Out_H;

   function Out_W (C : Conv2D) return Positive is
   begin
      return C.In_W - C.KW + 1;
   end Out_W;

   function Make_Conv2D
     (Num_F, In_H, In_W, KH, KW : Positive) return Conv2D
   is
      C : Conv2D (Num_F, In_H, In_W, KH, KW);
   begin
      pragma Assert (KH <= In_H and KW <= In_W);
      --  Fixed asymmetric seed: reproducible, breaks symmetry, no RNG.
      for F in 1 .. Num_F loop
         for I in 1 .. KH loop
            for J in 1 .. KW loop
               C.K (F, I, J) := 0.1 * Real (F + I + J) - 0.2;
            end loop;
         end loop;
         C.B (F) := 0.0;
      end loop;
      return C;
   end Make_Conv2D;

   function Make_Conv_State (Num_F, Map_H, Map_W : Positive)
     return Conv_State
   is
   begin
      return S : Conv_State (Num_F, Map_H, Map_W, Map_H * Map_W) do
         null;
      end return;
   end Make_Conv_State;

   function Make_Pool_State (Num_F, Map_H, Map_W : Positive)
     return Pool_State
   is
   begin
      return P : Pool_State (Num_F, Map_H, Map_W,
                             Map_H * Map_W, Map_H * Map_W * 4) do
         null;
      end return;
   end Make_Pool_State;

   function Get_Kernel
     (C : Conv2D; F, I, J : Positive) return Real
   is
   begin
      return C.K (F, I, J);
   end Get_Kernel;

   procedure Set_Kernel
     (C : in out Conv2D; F, I, J : Positive; V : Real)
   is
   begin
      C.K (F, I, J) := V;
   end Set_Kernel;

   procedure Conv_Forward (C : Conv2D; S : in out Conv_State; X : Matrix) is
      OH  : constant Positive := Out_H (C);
      OW  : constant Positive := Out_W (C);
      Acc : Real;
   begin
      pragma Assert (S.Num_F = C.Num_F);
      pragma Assert (S.Map_H = OH and S.Map_W = OW);
      pragma Assert (X'Length (1) = C.In_H and X'Length (2) = C.In_W);
      for F in 1 .. C.Num_F loop
         for R in 1 .. OH loop
            for Cc in 1 .. OW loop
               Acc := C.B (F);
               for I in 1 .. C.KH loop
                  for J in 1 .. C.KW loop
                     Acc := Acc + C.K (F, I, J)
                       * X (X'First (1) + R + I - 2,
                            X'First (2) + Cc + J - 2);
                  end loop;
               end loop;
               S.Z (F, (R - 1) * OW + Cc) := Acc;
               if Acc > 0.0 then
                  S.A (F, (R - 1) * OW + Cc) := Acc;
               else
                  S.A (F, (R - 1) * OW + Cc) := 0.0;
               end if;
            end loop;
         end loop;
      end loop;
   end Conv_Forward;

   procedure MaxPool_Forward (S_In : Conv_State; P : in out Pool_State) is
      W_In : constant Positive := S_In.Map_W;
   begin
      pragma Assert (P.Num_F = S_In.Num_F);
      pragma Assert (S_In.Map_H = 2 * P.Map_H);
      pragma Assert (S_In.Map_W = 2 * P.Map_W);
      for F in 1 .. P.Num_F loop
         for R in 1 .. P.Map_H loop
            for Cc in 1 .. P.Map_W loop
               declare
                  Best   : Real := Real'First;
                  Best_K : Natural := 0;
                  V      : Real;
                  PO     : constant Positive := (R - 1) * P.Map_W + Cc;
               begin
                  for DR in 0 .. 1 loop
                     for DC in 0 .. 1 loop
                        V := S_In.A
                          (F, (2 * R - 2 + DR) * W_In + (2 * Cc - 1 + DC));
                        if V > Best then
                           Best := V;
                           Best_K := DR * 2 + DC;
                        end if;
                     end loop;
                  end loop;
                  P.A (F, PO) := Best;
                  for K in 0 .. 3 loop
                     if K = Best_K then
                        P.Mask (F, (PO - 1) * 4 + K + 1) := 1.0;
                     else
                        P.Mask (F, (PO - 1) * 4 + K + 1) := 0.0;
                     end if;
                  end loop;
               end;
            end loop;
         end loop;
      end loop;
   end MaxPool_Forward;

   function Flatten (P : Pool_State) return Vector is
      N : constant Positive := P.Num_F * P.Map_H * P.Map_W;
      R : Vector (1 .. N);
   begin
      for F in 1 .. P.Num_F loop
         for K in 1 .. P.Map_H * P.Map_W loop
            R ((F - 1) * P.Map_H * P.Map_W + K) := P.A (F, K);
         end loop;
      end loop;
      return R;
   end Flatten;

   function Make_Dense (Out_Dim, In_Dim : Positive) return Dense_Layer is
      D : Dense_Layer (Out_Dim, In_Dim);
   begin
      for O in 1 .. Out_Dim loop
         for I in 1 .. In_Dim loop
            --  Fan-in scaled: keeps the sigmoid out of saturation.
            D.W (O, I) := (0.1 * Real (O + I) - 0.15) / Real (In_Dim);
         end loop;
         D.B (O) := 0.0;
      end loop;
      return D;
   end Make_Dense;

   procedure Dense_Forward (D : Dense_Layer; X : Vector; Y : out Vector) is
      Acc : Real;
   begin
      pragma Assert (X'Length = D.In_Dim);
      pragma Assert (Y'Length = D.Out_Dim);
      for O in 1 .. D.Out_Dim loop
         Acc := D.B (O);
         for I in 1 .. D.In_Dim loop
            Acc := Acc + D.W (O, I) * X (X'First + I - 1);
         end loop;
         Y (Y'First + O - 1) := 1.0 / (1.0 + Math.Exp (-Acc));
      end loop;
   end Dense_Forward;

   procedure Dense_Backward
     (D      : in out Dense_Layer;
      X      : Vector;
      Y      : Vector;
      Target : Vector;
      Eta    : Real;
      D_X    : out Vector)
   is
      D_Out : Vector (1 .. D.Out_Dim);
      Acc   : Real;
   begin
      pragma Assert (X'Length = D.In_Dim);
      pragma Assert (Y'Length = D.Out_Dim);
      pragma Assert (Target'Length = D.Out_Dim);
      pragma Assert (D_X'Length = D.In_Dim);
      pragma Assert (Eta >= 0.0);
      for O in 1 .. D.Out_Dim loop
         D_Out (O) :=
           (Y (Y'First + O - 1) - Target (Target'First + O - 1))
           * Y (Y'First + O - 1) * (1.0 - Y (Y'First + O - 1));
      end loop;
      for I in 1 .. D.In_Dim loop
         Acc := 0.0;
         for O in 1 .. D.Out_Dim loop
            Acc := Acc + D.W (O, I) * D_Out (O);
         end loop;
         D_X (D_X'First + I - 1) := Acc;
      end loop;
      if Eta > 0.0 then
         for O in 1 .. D.Out_Dim loop
            for I in 1 .. D.In_Dim loop
               D.W (O, I) :=
                 D.W (O, I) - Eta * D_Out (O) * X (X'First + I - 1);
            end loop;
            D.B (O) := D.B (O) - Eta * D_Out (O);
         end loop;
      end if;
   end Dense_Backward;

   procedure Conv_Gradients
     (C      : Conv2D;
      S      : Conv_State;
      P      : Pool_State;
      X      : Matrix;
      D_Pool : Real_Bank;
      DK     : out Real_3D;
      DB     : out Vector)
   is
      OH     : constant Positive := Out_H (C);
      OW     : constant Positive := Out_W (C);
      W_In   : constant Positive := S.Map_W;
      D_Conv : Real_Bank (1 .. C.Num_F, 1 .. OH * OW) :=
                 (others => (others => 0.0));
      PO, In_Flat : Positive;
      Kdx : Natural;
      Acc : Real;
   begin
      pragma Assert (S.Num_F = C.Num_F and P.Num_F = C.Num_F);
      pragma Assert (S.Map_H = OH and S.Map_W = OW);
      pragma Assert (S.Map_H = 2 * P.Map_H and S.Map_W = 2 * P.Map_W);
      pragma Assert (D_Pool'Length (1) = C.Num_F);
      pragma Assert (D_Pool'Length (2) = P.Map_H * P.Map_W);
      pragma Assert (DK'Length (1) = C.Num_F);
      pragma Assert (DK'Length (2) = C.KH);
      pragma Assert (DK'Length (3) = C.KW);
      pragma Assert (DB'Length = C.Num_F);

      --  1. Route pool deltas back through the argmax mask, times ReLU'.
      for F in 1 .. C.Num_F loop
         for R in 1 .. P.Map_H loop
            for Cc in 1 .. P.Map_W loop
               PO := (R - 1) * P.Map_W + Cc;
               for DR in 0 .. 1 loop
                  for DC in 0 .. 1 loop
                     Kdx := DR * 2 + DC;
                     In_Flat := (2 * R - 2 + DR) * W_In + (2 * Cc - 1 + DC);
                     if P.Mask (F, (PO - 1) * 4 + Kdx + 1) > 0.5 then
                        D_Conv (F, In_Flat) := D_Conv (F, In_Flat)
                          + D_Pool (D_Pool'First (1) + F - 1,
                                    D_Pool'First (2) + PO - 1);
                     end if;
                  end loop;
               end loop;
            end loop;
         end loop;
         for Q in 1 .. OH * OW loop
            if S.Z (F, Q) <= 0.0 then
               D_Conv (F, Q) := 0.0;
            end if;
         end loop;
      end loop;

      --  2. Kernel gradients: cross-correlation of X with D_Conv.
      for F in 1 .. C.Num_F loop
         for I in 1 .. C.KH loop
            for J in 1 .. C.KW loop
               Acc := 0.0;
               for R in 1 .. OH loop
                  for Cc in 1 .. OW loop
                     Acc := Acc
                       + X (X'First (1) + R + I - 2,
                            X'First (2) + Cc + J - 2)
                       * D_Conv (F, (R - 1) * OW + Cc);
                  end loop;
               end loop;
               DK (DK'First (1) + F - 1,
                   DK'First (2) + I - 1,
                   DK'First (3) + J - 1) := Acc;
            end loop;
         end loop;
         Acc := 0.0;
         for Q in 1 .. OH * OW loop
            Acc := Acc + D_Conv (F, Q);
         end loop;
         DB (DB'First + F - 1) := Acc;
      end loop;
   end Conv_Gradients;

   procedure Conv_Backward
     (C      : in out Conv2D;
      S      : Conv_State;
      P      : Pool_State;
      X      : Matrix;
      D_Pool : Real_Bank;
      Eta    : Real)
   is
      DK : Real_3D (1 .. C.Num_F, 1 .. C.KH, 1 .. C.KW);
      DB : Vector (1 .. C.Num_F);
   begin
      pragma Assert (Eta >= 0.0);
      Conv_Gradients (C, S, P, X, D_Pool, DK, DB);
      if Eta > 0.0 then
         for F in 1 .. C.Num_F loop
            for I in 1 .. C.KH loop
               for J in 1 .. C.KW loop
                  C.K (F, I, J) := C.K (F, I, J) - Eta * DK (F, I, J);
               end loop;
            end loop;
            C.B (F) := C.B (F) - Eta * DB (F);
         end loop;
      end if;
   end Conv_Backward;

end Mark1_CNN;
