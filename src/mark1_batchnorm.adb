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

--  MARK-I Runtime -- Batch normalization (body)

package body Mark1_BatchNorm is

   function Make_BatchNorm (Features : Positive) return BatchNorm is
      B : BatchNorm (Features);
   begin
      return B;
   end Make_BatchNorm;

   function Get_Gamma (B : BatchNorm; F : Positive) return Real is
   begin
      return B.Gamma (F);
   end Get_Gamma;

   function Get_Beta (B : BatchNorm; F : Positive) return Real is
   begin
      return B.Beta (F);
   end Get_Beta;

   procedure Set_Gamma (B : in out BatchNorm; F : Positive; V : Real) is
   begin
      B.Gamma (F) := V;
   end Set_Gamma;

   procedure Set_Beta (B : in out BatchNorm; F : Positive; V : Real) is
   begin
      B.Beta (F) := V;
   end Set_Beta;

   function Get_Run_Mean (B : BatchNorm; F : Positive) return Real is
   begin
      return B.Run_Mean (F);
   end Get_Run_Mean;

   function Get_Run_Var (B : BatchNorm; F : Positive) return Real is
   begin
      return B.Run_Var (F);
   end Get_Run_Var;

   procedure BN_Forward_Train
     (B : in out BatchNorm; X : Real_Bank; Y : out Real_Bank)
   is
      M    : constant Positive := X'Length (2);
      FX   : constant Integer := X'First (1);
      CX   : constant Integer := X'First (2);
      FY   : constant Integer := Y'First (1);
      CY   : constant Integer := Y'First (2);
      Mu, Va, Inv, Xh : Real;
   begin
      pragma Assert (X'Length (1) = B.Features);
      pragma Assert (Y'Length (1) = B.Features and Y'Length (2) = M);
      for F in 1 .. B.Features loop
         Mu := 0.0;
         for C in 1 .. M loop
            Mu := Mu + X (FX + F - 1, CX + C - 1);
         end loop;
         Mu := Mu / Real (M);
         Va := 0.0;
         for C in 1 .. M loop
            Va := Va + (X (FX + F - 1, CX + C - 1) - Mu) ** 2;
         end loop;
         Va := Va / Real (M);
         Inv := 1.0 / Math.Sqrt (Va + Epsilon);
         B.Run_Mean (F) := Momentum * B.Run_Mean (F)
                           + (1.0 - Momentum) * Mu;
         B.Run_Var (F) := Momentum * B.Run_Var (F)
                          + (1.0 - Momentum) * Va;
         for C in 1 .. M loop
            Xh := (X (FX + F - 1, CX + C - 1) - Mu) * Inv;
            Y (FY + F - 1, CY + C - 1) := B.Gamma (F) * Xh + B.Beta (F);
         end loop;
      end loop;
   end BN_Forward_Train;

   procedure BN_Forward_Infer
     (B : BatchNorm; X : Real_Bank; Y : out Real_Bank)
   is
      M    : constant Positive := X'Length (2);
      FX   : constant Integer := X'First (1);
      CX   : constant Integer := X'First (2);
      FY   : constant Integer := Y'First (1);
      CY   : constant Integer := Y'First (2);
      Inv, Xh : Real;
   begin
      pragma Assert (X'Length (1) = B.Features);
      pragma Assert (Y'Length (1) = B.Features and Y'Length (2) = M);
      for F in 1 .. B.Features loop
         Inv := 1.0 / Math.Sqrt (B.Run_Var (F) + Epsilon);
         for C in 1 .. M loop
            Xh := (X (FX + F - 1, CX + C - 1) - B.Run_Mean (F)) * Inv;
            Y (FY + F - 1, CY + C - 1) := B.Gamma (F) * Xh + B.Beta (F);
         end loop;
      end loop;
   end BN_Forward_Infer;

   procedure BN_Gradients
     (B       : BatchNorm;
      X       : Real_Bank;
      D_Y     : Real_Bank;
      D_X     : out Real_Bank;
      D_Gamma : out Vector;
      D_Beta  : out Vector)
   is
      M     : constant Positive := X'Length (2);
      FX    : constant Integer := X'First (1);
      CX    : constant Integer := X'First (2);
      FD    : constant Integer := D_Y'First (1);
      CD    : constant Integer := D_Y'First (2);
      FXO   : constant Integer := D_X'First (1);
      CXO   : constant Integer := D_X'First (2);
      Mu, Va, Inv, Xh, Dvar, Dmu, Sc : Real;
   begin
      pragma Assert (X'Length (1) = B.Features);
      pragma Assert (D_Y'Length (1) = B.Features and D_Y'Length (2) = M);
      pragma Assert (D_X'Length (1) = B.Features and D_X'Length (2) = M);
      pragma Assert (D_Gamma'Length = B.Features);
      pragma Assert (D_Beta'Length = B.Features);
      for F in 1 .. B.Features loop
         --  Batch statistics of the forward pass, recomputed from X.
         Mu := 0.0;
         for C in 1 .. M loop
            Mu := Mu + X (FX + F - 1, CX + C - 1);
         end loop;
         Mu := Mu / Real (M);
         Va := 0.0;
         for C in 1 .. M loop
            Va := Va + (X (FX + F - 1, CX + C - 1) - Mu) ** 2;
         end loop;
         Va := Va / Real (M);
         Inv := 1.0 / Math.Sqrt (Va + Epsilon);
         Sc := Va + Epsilon;

         D_Beta (D_Beta'First + F - 1) := 0.0;
         D_Gamma (D_Gamma'First + F - 1) := 0.0;
         Dvar := 0.0;
         Dmu := 0.0;
         for C in 1 .. M loop
            Xh := (X (FX + F - 1, CX + C - 1) - Mu) * Inv;
            D_Beta (D_Beta'First + F - 1) :=
              D_Beta (D_Beta'First + F - 1)
              + D_Y (FD + F - 1, CD + C - 1);
            D_Gamma (D_Gamma'First + F - 1) :=
              D_Gamma (D_Gamma'First + F - 1)
              + D_Y (FD + F - 1, CD + C - 1) * Xh;
            Dvar := Dvar + D_Y (FD + F - 1, CD + C - 1)
                    * B.Gamma (F) * (X (FX + F - 1, CX + C - 1) - Mu)
                    * (-0.5) / (Sc * Math.Sqrt (Sc));
            Dmu := Dmu + D_Y (FD + F - 1, CD + C - 1)
                   * B.Gamma (F) * (-Inv);
         end loop;
         for C in 1 .. M loop
            D_X (FXO + F - 1, CXO + C - 1) :=
              D_Y (FD + F - 1, CD + C - 1) * B.Gamma (F) * Inv
              + Dvar * 2.0 * (X (FX + F - 1, CX + C - 1) - Mu) / Real (M)
              + Dmu / Real (M);
         end loop;
      end loop;
   end BN_Gradients;

   procedure BN_Backward
     (B    : in out BatchNorm;
      X    : Real_Bank;
      D_Y  : Real_Bank;
      Eta  : Real;
      D_X  : out Real_Bank)
   is
      Dg : Vector (1 .. B.Features);
      Db : Vector (1 .. B.Features);
   begin
      pragma Assert (Eta >= 0.0);
      BN_Gradients (B, X, D_Y, D_X, Dg, Db);
      if Eta > 0.0 then
         for F in 1 .. B.Features loop
            B.Gamma (F) := B.Gamma (F) - Eta * Dg (F);
            B.Beta (F)  := B.Beta (F) - Eta * Db (F);
         end loop;
      end if;
   end BN_Backward;

end Mark1_BatchNorm;
