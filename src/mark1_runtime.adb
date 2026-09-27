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

--  MARK-I Runtime -- lifecycle driver
--  INIT -> TRAIN -> EVALUATE -> REPORT. Every check below is a raw
--  pragma Assert: the program proves itself or it does not run.

with Ada.Text_IO;      use Ada.Text_IO;
with Mark1_K;          use Mark1_K;
with Mark1_Rosenblatt; use Mark1_Rosenblatt;
with Mark1_Backprop;   use Mark1_Backprop;
with Mark1_Net;        use Mark1_Net;
with Mark1_CNN;        use Mark1_CNN;
with Mark1_BatchNorm;  use Mark1_BatchNorm;
with Mark1_Attention;  use Mark1_Attention;
with Mark1_GPU;

procedure Mark1_Runtime is

   pragma Assertion_Policy (Check);

   Passed : Natural := 0;

   procedure Check (Cond : Boolean; Name : String) is
   begin
      pragma Assert (Cond);
      Passed := Passed + 1;
      Put_Line ("  [ok] " & Name);
   end Check;

   procedure Phase (Name : String) is
   begin
      Put_Line ("");
      Put_Line ("-- " & Name & " --");
   end Phase;

begin
   Put_Line ("+--------------------------------+");
   Put_Line ("|       MARK-I RUNTIME           |");
   Put_Line ("|   Ada 2007 / K array engine    |");
   Put_Line ("+--------------------------------+");

   ---------------------------------------------------------
   Phase ("INIT : K array engine");
   ---------------------------------------------------------
   declare
      A : Vector := (1.0, 2.0, 3.0);
      B : Vector := (4.0, 5.0, 6.0);
   begin
      Check (Dot (A, B) = 32.0, "dot product = 32");
      Check (Norm2 ((3.0, 4.0)) = 5.0, "norm2 (3,4) = 5");
   end;

   declare
      A : Matrix (1 .. 2, 1 .. 2) := ((1.0, 2.0), (3.0, 4.0));
      B : Matrix (1 .. 2, 1 .. 2) := ((5.0, 6.0), (7.0, 8.0));
      C : Matrix (1 .. 2, 1 .. 2);
   begin
      C := Matmul (A, B);
      Check (C = ((19.0, 22.0), (43.0, 50.0)), "matmul 2x2");
      Check (Matvec (A, (1.0, 1.0)) = (3.0, 7.0), "matvec");
      Check (Transpose (A) = ((1.0, 3.0), (2.0, 4.0)), "transpose");
      Check (Outer ((1.0, 2.0), (3.0, 4.0)) =
               ((3.0, 4.0), (6.0, 8.0)),
             "outer product");
   end;

   declare
      T : Tensor := Make_Tensor ((2, 3));
   begin
      Check (T.Rank = 2, "tensor rank 2");
      Check (Element_Count ((2, 3)) = 6, "tensor element count 6");
      Set (T, (2, 3), 7.5);
      Check (Get (T, (2, 3)) = 7.5, "tensor set/get");
      Check (Lin (T, (2, 3)) = 6, "tensor linear index row-major");
      Free (T);
   end;

   declare
      M : Assoc_Memory (Key_Dim => 2, Val_Dim => 2, Capacity => 8);
      R : Vector (1 .. 2);
   begin
      Store (M, (1.0, 0.0), (9.0, 9.0));
      Store (M, (0.0, 1.0), (7.0, 7.0));
      Check (Stored (M) = 2, "assoc store count");
      R := Recall (M, (0.9, 0.1));
      Check (R = (9.0, 9.0), "assoc recall nearest key");
   end;

   declare
      W : Matrix (1 .. 2, 1 .. 2) := (others => (others => 0.0));
   begin
      Hebb (W, (1.0, 0.0), (0.0, 1.0), 0.5);
      Check (W (2, 1) = 0.5, "hebbian update dw = eta*y*x");
   end;

   declare
      A : Matrix (1 .. 2, 1 .. 2) := ((1.0, 2.0), (3.0, 4.0));
      B : Matrix (1 .. 2, 1 .. 2) := ((5.0, 6.0), (7.0, 8.0));
      C : Matrix (1 .. 2, 1 .. 2);
      R : Vector (1 .. 2);
   begin
      Mark1_GPU.Sgemm (A, B, C);
      Check (C = ((19.0, 22.0), (43.0, 50.0)), "gpu sgemm path");
      Mark1_GPU.Sgemv (A, (1.0, 1.0), R);
      Check (R = (3.0, 7.0), "gpu sgemv path");
   end;

   ---------------------------------------------------------
   Phase ("TRAIN : rosenblatt feedback loop");
   ---------------------------------------------------------
   declare
      P : Perceptron := Make (2);
      type Sample is record
         X : Vector (1 .. 2);
         D : Integer;
      end record;
      Data : array (1 .. 4) of Sample :=
        (((0.0, 0.0), -1),
         ((0.0, 1.0), -1),
         ((1.0, 0.0), -1),
         ((1.0, 1.0),  1));
      Errors : Natural := 0;
   begin
      for Epoch in 1 .. 100 loop
         Errors := 0;
         for S in Data'Range loop
            if Respond (P, Data (S).X) /= Data (S).D then
               Errors := Errors + 1;
            end if;
            Feedback (P, Data (S).X, Data (S).D);  -- stimulus -> response
                                                   --        ^         |
                                                   --        +- feedback
         end loop;
         exit when Errors = 0;
      end loop;
      Put_Line ("  converged, final bias = " & Real'Image (Bias_Of (P)));
      Check (Errors = 0, "rosenblatt converged on AND");
   end;

   ---------------------------------------------------------
   Phase ("EVALUATE");
   ---------------------------------------------------------
   declare
      P : Perceptron := Make (2);
      Data_X : array (1 .. 4) of Vector (1 .. 2) :=
        ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0));
      Data_D : array (1 .. 4) of Integer := (-1, -1, -1, 1);
   begin
      for Epoch in 1 .. 100 loop
         for S in Data_X'Range loop
            Feedback (P, Data_X (S), Data_D (S));
         end loop;
         exit when Respond (P, (0.0, 0.0)) = -1
           and then Respond (P, (0.0, 1.0)) = -1
           and then Respond (P, (1.0, 0.0)) = -1
           and then Respond (P, (1.0, 1.0)) = 1;
      end loop;
      Check (Respond (P, (1.0, 1.0)) = 1, "AND(1,1) = +1");
      Check (Respond (P, (0.0, 1.0)) = -1, "AND(0,1) = -1");
      Check (Respond (P, (0.0, 0.0)) = -1, "AND(0,0) = -1");
   end;
   declare
      In_V : Vector (1 .. 3) := (0.0, 1000.0, -1000.0);
      Sv   : Vector (1 .. 3);
   begin
      Check (abs (Mark1_K.Sigmoid (0.0) - 0.5) < 1.0e-12,
             "sigmoid(0) = 0.5");
      Check (Mark1_K.Sigmoid (1000.0) > 1.0 - 1.0e-12
             and then Mark1_K.Sigmoid (-1000.0) < 1.0e-12,
             "sigmoid saturates, no overflow");
      Check (abs (Sigmoid_D (0.0) - 0.25) < 1.0e-12, "sigmoid'(0) = 0.25");
      Sv := Mark1_K.Sigmoid (In_V);
      Check (abs (Sv (1) - 0.5) < 1.0e-12
             and then Sv (2) > 1.0 - 1.0e-12
             and then Sv (3) < 1.0e-12,
             "sigmoid vector element-wise");
   end;

   ---------------------------------------------------------
   Phase ("TRAIN : backpropagating errors (XOR)");
   ---------------------------------------------------------
   declare
      Net : MLP := Make_MLP (2, 4, 1);
      Xs  : array (1 .. 4) of Vector (1 .. 2) :=
        ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0));
      Ts  : array (1 .. 4) of Vector (1 .. 1) :=
        ((1 => 0.0), (1 => 1.0), (1 => 1.0), (1 => 0.0));
      Max_Err : Real;
      E       : Real;
   begin
      for Epoch in 1 .. 20_000 loop
         for S in Xs'Range loop
            Train_Sample (Net, Xs (S), Ts (S), 0.5);
         end loop;
      end loop;
      Max_Err := 0.0;
      for S in Xs'Range loop
         E := abs (Output_Of (Net, Xs (S)) (1) - Ts (S) (1));
         if E > Max_Err then
            Max_Err := E;
         end if;
      end loop;
      Put_Line ("  max abs error after backprop:" & Real'Image (Max_Err));
      Check (Max_Err < 0.2, "mlp xor converged, max err < 0.2");
      Check (abs (Output_Of (Net, (0.0, 0.0)) (1) - 0.0) < 0.2, "XOR(0,0) ~ 0");
      Check (abs (Output_Of (Net, (0.0, 1.0)) (1) - 1.0) < 0.2, "XOR(0,1) ~ 1");
      Check (abs (Output_Of (Net, (1.0, 0.0)) (1) - 1.0) < 0.2, "XOR(1,0) ~ 1");
      Check (abs (Output_Of (Net, (1.0, 1.0)) (1) - 0.0) < 0.2, "XOR(1,1) ~ 0");
   end;

   ---------------------------------------------------------
   Phase ("TRAIN : custom hand-rolled net (sine)");
   ---------------------------------------------------------
   declare
      Net : Custom_Net :=
        Make_Net ((1, 12, 12, 1), (Linear_A, Tanh_A, Tanh_A, Linear_A));
      St    : Net_State (Depth => 4, Width => 12);
      N_Pts : constant := 24;
      Pi    : constant Real := 3.14159265358979;
      type Pt is record
         X, T : Real;
      end record;
      Data : array (1 .. N_Pts) of Pt;
      Xin  : Vector (1 .. 1);
      Tin  : Vector (1 .. 1);
      Mse  : Real := 0.0;
      E    : Real;
   begin
      for I in 1 .. N_Pts loop
         Data (I).X := -Pi + 2.0 * Pi * Real (I - 1) / Real (N_Pts - 1);
         Data (I).T := Math.Sin (Data (I).X);
      end loop;
      for Epoch in 1 .. 8_000 loop
         for I in 1 .. N_Pts loop
            Xin (1) := Data (I).X / Pi;  -- scale input to [-1, 1]
            Tin (1) := Data (I).T;
            Train_Sample (Net, St, Xin, Tin, 0.05);
         end loop;
      end loop;
      for I in 1 .. N_Pts loop
         Xin (1) := Data (I).X / Pi;
         Forward (Net, St, Xin);
         E := Output_Of (Net, St) (1) - Data (I).T;
         Mse := Mse + E * E;
      end loop;
      Mse := Mse / Real (N_Pts);
      Put_Line ("  sine MSE:" & Real'Image (Mse));
      Check (Mse < 0.02, "custom net learned sine, mse < 0.02");
      Xin (1) := 0.0;
      Forward (Net, St, Xin);
      Check (abs (Output_Of (Net, St) (1) - 0.0) < 0.15, "sine(0) ~ 0");
      Xin (1) := 0.5;  -- (pi/2) / pi
      Forward (Net, St, Xin);
      Check (abs (Output_Of (Net, St) (1) - 1.0) < 0.15, "sine(pi/2) ~ 1");
      Xin (1) := -0.5;
      Forward (Net, St, Xin);
      Check (abs (Output_Of (Net, St) (1) + 1.0) < 0.15, "sine(-pi/2) ~ -1");
   end;

   ---------------------------------------------------------
   Phase ("TRAIN : convolutional neural network (bars)");
   ---------------------------------------------------------
   declare
      C  : Conv2D := Make_Conv2D (4, 8, 8, 3, 3);
      CS : Conv_State := Make_Conv_State (4, 6, 6);
      PS : Pool_State := Make_Pool_State (4, 3, 3);
      D  : Dense_Layer := Make_Dense (1, 36);

      function Bar_Image (Horizontal : Boolean; Pos : Positive) return Matrix is
         M : Matrix (1 .. 8, 1 .. 8) := (others => (others => 0.0));
      begin
         pragma Assert (Pos in 2 .. 6);
         if Horizontal then
            for Cc in 1 .. 8 loop
               M (Pos, Cc) := 1.0;
               M (Pos + 1, Cc) := 1.0;
            end loop;
         else
            for R in 1 .. 8 loop
               M (R, Pos) := 1.0;
               M (R, Pos + 1) := 1.0;
            end loop;
         end if;
         return M;
      end Bar_Image;

      procedure Forward_All (Img : Matrix; Fv : out Vector; Yv : out Vector) is
      begin
         Conv_Forward (C, CS, Img);
         MaxPool_Forward (CS, PS);
         Fv := Flatten (PS);
         Dense_Forward (D, Fv, Yv);
      end Forward_All;
   begin
      --  Numeric gradient check on K(1,1,1) before training.
      declare
         Img : Matrix := Bar_Image (True, 3);
         Fv  : Vector (1 .. 36);
         Yv  : Vector (1 .. 1);
         Tv  : Vector (1 .. 1) := (1 => 0.0);
         Dx  : Vector (1 .. 36);
         Dp  : Real_Bank (1 .. 4, 1 .. 9);
         DK  : Real_3D (1 .. 4, 1 .. 3, 1 .. 3);
         DB  : Vector (1 .. 4);
         Eps : constant Real := 1.0e-4;
         Old, Lp, Lm, Num, Ana, E : Real;

         function Loss return Real is
            F2 : Vector (1 .. 36);
            Y2 : Vector (1 .. 1);
         begin
            Forward_All (Img, F2, Y2);
            E := Y2 (1) - Tv (1);
            return 0.5 * E * E;  --  matches Dense_Backward's Delta
         end Loss;
      begin
         Forward_All (Img, Fv, Yv);
         Dense_Backward (D, Fv, Yv, Tv, 0.0, Dx);  -- deltas, no update
         for F in 1 .. 4 loop
            for Q in 1 .. 9 loop
               Dp (F, Q) := Dx ((F - 1) * 9 + Q);
            end loop;
         end loop;
         Conv_Gradients (C, CS, PS, Img, Dp, DK, DB);
         Ana := DK (1, 1, 1);
         Old := Get_Kernel (C, 1, 1, 1);
         Set_Kernel (C, 1, 1, 1, Old + Eps);
         Lp := Loss;
         Set_Kernel (C, 1, 1, 1, Old - Eps);
         Lm := Loss;
         Set_Kernel (C, 1, 1, 1, Old);
         Num := (Lp - Lm) / (2.0 * Eps);
         Put_Line ("  cnn grad check: analytic =" & Real'Image (Ana)
                   & " numeric =" & Real'Image (Num));
         Check (abs (Ana - Num) < 1.0e-3, "cnn gradient check dK(1,1,1)");
      end;

      --  Train: horizontal bars -> 0, vertical bars -> 1, positions 2..5.
      declare
         Img     : Matrix (1 .. 8, 1 .. 8);
         Fv      : Vector (1 .. 36);
         Yv      : Vector (1 .. 1);
         Tv      : Vector (1 .. 1);
         Dx      : Vector (1 .. 36);
         Dp      : Real_Bank (1 .. 4, 1 .. 9);
         Correct : Natural := 0;
         Total   : Natural := 0;

         procedure Train_One (Img_In : Matrix; T : Real) is
         begin
            Tv (1) := T;
            Forward_All (Img_In, Fv, Yv);
            Dense_Backward (D, Fv, Yv, Tv, 0.1, Dx);
            for F in 1 .. 4 loop
               for Q in 1 .. 9 loop
                  Dp (F, Q) := Dx ((F - 1) * 9 + Q);
               end loop;
            end loop;
            Conv_Backward (C, CS, PS, Img_In, Dp, 0.1);
         end Train_One;
      begin
         for Epoch in 1 .. 1_500 loop
            for Pos in 2 .. 5 loop
               Img := Bar_Image (True, Pos);
               Train_One (Img, 0.0);
               Img := Bar_Image (False, Pos);
               Train_One (Img, 1.0);
            end loop;
         end loop;
         for Pos in 2 .. 5 loop
            Forward_All (Bar_Image (True, Pos), Fv, Yv);
            Total := Total + 1;
            if Yv (1) < 0.5 then
               Correct := Correct + 1;
            end if;
            Forward_All (Bar_Image (False, Pos), Fv, Yv);
            Total := Total + 1;
            if Yv (1) >= 0.5 then
               Correct := Correct + 1;
            end if;
         end loop;
         Put_Line ("  cnn bars:" & Natural'Image (Correct)
                   & " /" & Natural'Image (Total));
         Check (Correct = Total, "cnn classified all 8 bar images");
         Forward_All (Bar_Image (True, 3), Fv, Yv);
         Check (Yv (1) < 0.3, "cnn horizontal bar -> 0");
         Forward_All (Bar_Image (False, 4), Fv, Yv);
         Check (Yv (1) > 0.7, "cnn vertical bar -> 1");
      end;
   end;

   ---------------------------------------------------------
   Phase ("TRAIN : batch normalization");
   ---------------------------------------------------------
   --  Unit checks on a fixed 2-feature x 4-sample batch.
   declare
      B  : BatchNorm := Make_BatchNorm (2);
      X  : Real_Bank (1 .. 2, 1 .. 4) :=
        ((1.0, 2.0, 3.0, 4.0),
         (2.0, 4.0, 6.0, 8.0));
      Y  : Real_Bank (1 .. 2, 1 .. 4);

      function Mean_Of (M : Real_Bank; F : Positive) return Real is
         S : Real := 0.0;
      begin
         for C in 1 .. 4 loop
            S := S + M (F, C);
         end loop;
         return S / 4.0;
      end Mean_Of;

      function Var_Of (M : Real_Bank; F : Positive) return Real is
         Mu : constant Real := Mean_Of (M, F);
         S  : Real := 0.0;
      begin
         for C in 1 .. 4 loop
            S := S + (M (F, C) - Mu) ** 2;
         end loop;
         return S / 4.0;
      end Var_Of;
   begin
      BN_Forward_Train (B, X, Y);
      Check (abs (Mean_Of (Y, 1)) < 1.0e-9
             and then abs (Mean_Of (Y, 2)) < 1.0e-9,
             "bn forward: zero mean per feature");
      Check (abs (Var_Of (Y, 1) - 1.0) < 1.0e-4
             and then abs (Var_Of (Y, 2) - 1.0) < 1.0e-4,
             "bn forward: unit variance per feature");

      Set_Gamma (B, 1, 2.0);
      Set_Gamma (B, 2, 2.0);
      Set_Beta (B, 1, 1.0);
      Set_Beta (B, 2, 1.0);
      BN_Forward_Train (B, X, Y);
      Check (abs (Mean_Of (Y, 1) - 1.0) < 1.0e-9
             and then abs (Var_Of (Y, 1) - 4.0) < 1.0e-3
             and then abs (Mean_Of (Y, 2) - 1.0) < 1.0e-9
             and then abs (Var_Of (Y, 2) - 4.0) < 1.0e-3,
             "bn forward: gamma/beta scale and shift");
   end;
   --  Running statistics + inference path.
   declare
      B2 : BatchNorm := Make_BatchNorm (2);
      X  : Real_Bank (1 .. 2, 1 .. 4) :=
        ((1.0, 2.0, 3.0, 4.0),
         (2.0, 4.0, 6.0, 8.0));
      Y  : Real_Bank (1 .. 2, 1 .. 4);
   begin
      BN_Forward_Train (B2, X, Y);
      --  Feature 1: batch mean 2.5, batch var 1.25; momentum 0.9.
      --  Feature 2: batch mean 5.0, batch var 5.0.
      --  Run_Var starts at 1.0: 0.9 * 1.0 + 0.1 * batch_var.
      Check (abs (Get_Run_Mean (B2, 1) - 0.25) < 1.0e-9
             and then abs (Get_Run_Var (B2, 1) - 1.025) < 1.0e-9
             and then abs (Get_Run_Mean (B2, 2) - 0.5) < 1.0e-9
             and then abs (Get_Run_Var (B2, 2) - 1.4) < 1.0e-9,
             "bn running stats momentum update");
      BN_Forward_Infer (B2, X, Y);
      Check (abs (Y (1, 1) - 0.75 / Math.Sqrt (1.025 + Epsilon)) < 1.0e-9
             and then
             abs (Y (2, 4) - 7.5 / Math.Sqrt (1.4 + Epsilon)) < 1.0e-9,
             "bn infer uses running stats");
   end;
   --  Analytic vs numeric gradients.
   declare
      B3  : BatchNorm := Make_BatchNorm (2);
      Xg  : Real_Bank (1 .. 2, 1 .. 4) :=
        ((1.0, 2.0, 3.0, 4.0),
         (2.0, 4.0, 6.0, 8.0));
      Dyg : Real_Bank (1 .. 2, 1 .. 4) :=
        ((0.5, -1.0, 0.25, 2.0),
         (-0.5, 1.5, -2.0, 0.75));
      Dxg : Real_Bank (1 .. 2, 1 .. 4);
      Dg  : Vector (1 .. 2);
      Db  : Vector (1 .. 2);
      Eps : constant Real := 1.0e-5;
      Ag, Ng, Lp, Lm, Old : Real;

      function Loss return Real is
         Bc : BatchNorm := B3;
         Yl : Real_Bank (1 .. 2, 1 .. 4);
         S  : Real := 0.0;
      begin
         BN_Forward_Train (Bc, Xg, Yl);
         for F in 1 .. 2 loop
            for C in 1 .. 4 loop
               S := S + Dyg (F, C) * Yl (F, C);
            end loop;
         end loop;
         return S;
      end Loss;
   begin
      BN_Gradients (B3, Xg, Dyg, Dxg, Dg, Db);
      Ag := Dxg (1, 1);
      Old := Xg (1, 1);
      Xg (1, 1) := Old + Eps;
      Lp := Loss;
      Xg (1, 1) := Old - Eps;
      Lm := Loss;
      Xg (1, 1) := Old;
      Ng := (Lp - Lm) / (2.0 * Eps);
      Put_Line ("  bn grad check dX: analytic =" & Real'Image (Ag)
                & " numeric =" & Real'Image (Ng));
      Check (abs (Ag - Ng) < 1.0e-4, "bn gradient check dX(1,1)");
      Ag := Dg (1);
      Old := Get_Gamma (B3, 1);
      Set_Gamma (B3, 1, Old + Eps);
      Lp := Loss;
      Set_Gamma (B3, 1, Old - Eps);
      Lm := Loss;
      Set_Gamma (B3, 1, Old);
      Ng := (Lp - Lm) / (2.0 * Eps);
      Put_Line ("  bn grad check dGamma: analytic =" & Real'Image (Ag)
                & " numeric =" & Real'Image (Ng));
      Check (abs (Ag - Ng) < 1.0e-4, "bn gradient check dGamma(1)");
   end;
   --  End-to-end: 2-4-1 tanh MLP with batchnorm on the hidden
   --  activations learns XOR over the full 4-sample batch.
   declare
      BH : BatchNorm := Make_BatchNorm (4);
      W1 : Matrix (1 .. 4, 1 .. 2);
      B1 : Vector (1 .. 4) := (others => 0.0);
      W2 : Matrix (1 .. 1, 1 .. 4);
      B2 : Real := 0.0;
      Xd : Real_Bank (1 .. 2, 1 .. 4) :=
        ((0.0, 0.0, 1.0, 1.0),
         (0.0, 1.0, 0.0, 1.0));
      Td : Vector (1 .. 4) := (0.0, 1.0, 1.0, 0.0);
      H, Hn, Dh_N, Dh : Real_Bank (1 .. 4, 1 .. 4);
      Yd   : Vector (1 .. 4);
      Dlt  : Vector (1 .. 4);
      Acc  : Real;
      Ok   : Natural;
   begin
      for O in 1 .. 4 loop
         for I in 1 .. 2 loop
            W1 (O, I) := (0.1 * Real (O + I) - 0.2) / 2.0;
         end loop;
      end loop;
      for I in 1 .. 4 loop
         W2 (1, I) := (0.1 * Real (1 + I) - 0.15) / 4.0;
      end loop;
      for Epoch in 1 .. 3_000 loop
         for O in 1 .. 4 loop
            for C in 1 .. 4 loop
               Acc := B1 (O);
               for I in 1 .. 2 loop
                  Acc := Acc + W1 (O, I) * Xd (I, C);
               end loop;
               H (O, C) := Math.Tanh (Acc);
            end loop;
         end loop;
         BN_Forward_Train (BH, H, Hn);
         for C in 1 .. 4 loop
            Acc := B2;
            for O in 1 .. 4 loop
               Acc := Acc + W2 (1, O) * Hn (O, C);
            end loop;
            Yd (C) := Mark1_K.Sigmoid (Acc);
            Dlt (C) := (Yd (C) - Td (C)) * Yd (C) * (1.0 - Yd (C));
         end loop;
         for O in 1 .. 4 loop
            for C in 1 .. 4 loop
               Dh_N (O, C) := W2 (1, O) * Dlt (C);
            end loop;
         end loop;
         BN_Backward (BH, H, Dh_N, 0.5, Dh);
         for O in 1 .. 4 loop
            for I in 1 .. 2 loop
               Acc := 0.0;
               for C in 1 .. 4 loop
                  Acc := Acc + Dh (O, C) * (1.0 - H (O, C) ** 2) * Xd (I, C);
               end loop;
               W1 (O, I) := W1 (O, I) - 0.5 * Acc;
            end loop;
            Acc := 0.0;
            for C in 1 .. 4 loop
               Acc := Acc + Dh (O, C) * (1.0 - H (O, C) ** 2);
            end loop;
            B1 (O) := B1 (O) - 0.5 * Acc;
         end loop;
         for O in 1 .. 4 loop
            Acc := 0.0;
            for C in 1 .. 4 loop
               Acc := Acc + Dlt (C) * Hn (O, C);
            end loop;
            W2 (1, O) := W2 (1, O) - 0.5 * Acc;
         end loop;
         Acc := 0.0;
         for C in 1 .. 4 loop
            Acc := Acc + Dlt (C);
         end loop;
         B2 := B2 - 0.5 * Acc;
      end loop;
      --  Evaluate through the inference path.
      for O in 1 .. 4 loop
         for C in 1 .. 4 loop
            Acc := B1 (O);
            for I in 1 .. 2 loop
               Acc := Acc + W1 (O, I) * Xd (I, C);
            end loop;
            H (O, C) := Math.Tanh (Acc);
         end loop;
      end loop;
      BN_Forward_Infer (BH, H, Hn);
      Ok := 0;
      for C in 1 .. 4 loop
         Acc := B2;
         for O in 1 .. 4 loop
            Acc := Acc + W2 (1, O) * Hn (O, C);
         end loop;
         if (Mark1_K.Sigmoid (Acc) >= 0.5) = (Td (C) > 0.5) then
            Ok := Ok + 1;
         end if;
      end loop;
      Put_Line ("  bn xor:" & Natural'Image (Ok) & " / 4");
      Check (Ok = 4, "bn mlp learned xor");
   end;

   ---------------------------------------------------------
   Phase ("TRAIN : multi-head attention (semantic scoring)");
   ---------------------------------------------------------
   --  Library search: 4 tokens, 2 heads. Head 1 scores similarity in
   --  dims 1..4, so query token 1 retrieves document 3; head 2 scores
   --  dims 5..8, so the same query retrieves document 4. The scores
   --  vary per query/key pair -- that variance IS the semantics.
   declare
      MA : Multi_Head := Make_Multi_Head (2, 8, 4, 4);
      X  : Real_Bank (1 .. 8, 1 .. 4) :=
        (1 => (1.00, 0.00, 0.95, 0.10),
         2 => (0.10, 1.00, 0.12, 0.00),
         3 => (0.10, 0.00, 0.08, 1.00),
         4 => (0.00, 0.90, 0.02, 0.10),
         5 => (0.10, 0.90, 0.05, 0.12),
         6 => (0.00, 0.10, 0.90, 0.05),
         7 => (0.10, 0.00, 0.10, 0.08),
         8 => (0.90, 0.05, 0.00, 0.95));
      A1 : Real_Bank (1 .. 4, 1 .. 4);
      A2 : Real_Bank (1 .. 4, 1 .. 4);
      Rs : Real;
   begin
      --  Scale 2.0 projections: scores separate enough to read.
      for I in 1 .. 4 loop
         for J in 1 .. 8 loop
            if J = I then
               Set_Wq (MA, 1, I, J, 2.0);
               Set_Wk (MA, 1, I, J, 2.0);
            else
               Set_Wq (MA, 1, I, J, 0.0);
               Set_Wk (MA, 1, I, J, 0.0);
            end if;
            if J = I + 4 then
               Set_Wq (MA, 2, I, J, 2.0);
               Set_Wk (MA, 2, I, J, 2.0);
            else
               Set_Wq (MA, 2, I, J, 0.0);
               Set_Wk (MA, 2, I, J, 0.0);
            end if;
         end loop;
      end loop;
      Attn_Scores (MA, 1, X, A1);
      Attn_Scores (MA, 2, X, A2);
      Put_Line ("  attn head 1 scores (query x key):");
      for I in 1 .. 4 loop
         Put ("   ");
         for J in 1 .. 4 loop
            Put (Real'Image (A1 (I, J)));
         end loop;
         New_Line;
      end loop;
      Rs := 0.0;
      for I in 1 .. 4 loop
         for J in 1 .. 4 loop
            Rs := Rs + A1 (I, J);
         end loop;
      end loop;
      Check (abs (Rs - 4.0) < 1.0e-9, "attn scores: rows sum to 1");
      Check (A1 (1, 3) > A1 (1, 2) and then A1 (1, 3) > A1 (1, 4),
             "attn head 1 retrieves the matching document");
      Check (A1 (1, 3) - A1 (1, 2) > 0.3,
             "attn scores vary between query/key pairs");
      Check (A2 (1, 4) > A2 (1, 3) and then A2 (1, 4) > A2 (1, 2),
             "attn head 2 retrieves a different document");
   end;
   --  Analytic vs numeric gradients through softmax, Q/K/V and Wo.
   declare
      MG  : Multi_Head := Make_Multi_Head (2, 4, 2, 2);
      Xg  : Real_Bank (1 .. 4, 1 .. 3) :=
        (1 => (0.5, -0.2, 0.8),
         2 => (-0.3, 0.7, 0.1),
         3 => (0.9, 0.4, -0.5),
         4 => (0.2, -0.8, 0.6));
      Dyg : Real_Bank (1 .. 4, 1 .. 3) :=
        (1 => (0.5, -1.0, 0.25),
         2 => (-0.5, 1.5, -2.0),
         3 => (1.0, 0.5, -0.75),
         4 => (-1.5, 0.25, 1.0));
      Dxg : Real_Bank (1 .. 4, 1 .. 3);
      DWq : Matrix (1 .. 4, 1 .. 4);
      DWk : Matrix (1 .. 4, 1 .. 4);
      DWv : Matrix (1 .. 4, 1 .. 4);
      DWo : Matrix (1 .. 4, 1 .. 4);
      Eps : constant Real := 1.0e-5;
      Ag, Ng, Lp, Lm, Old : Real;

      function Loss return Real is
         Yl : Real_Bank (1 .. 4, 1 .. 3);
         S  : Real := 0.0;
      begin
         Attn_Forward (MG, Xg, Yl);
         for I in 1 .. 4 loop
            for C in 1 .. 3 loop
               S := S + Dyg (I, C) * Yl (I, C);
            end loop;
         end loop;
         return S;
      end Loss;
   begin
      Attn_Gradients (MG, Xg, Dyg, Dxg, DWq, DWk, DWv, DWo);
      Ag := DWq (1, 1);
      Old := Get_Wq (MG, 1, 1, 1);
      Set_Wq (MG, 1, 1, 1, Old + Eps);
      Lp := Loss;
      Set_Wq (MG, 1, 1, 1, Old - Eps);
      Lm := Loss;
      Set_Wq (MG, 1, 1, 1, Old);
      Ng := (Lp - Lm) / (2.0 * Eps);
      Put_Line ("  attn grad check dWq: analytic =" & Real'Image (Ag)
                & " numeric =" & Real'Image (Ng));
      Check (abs (Ag - Ng) < 1.0e-4, "attn gradient check dWq");
      Ag := DWo (2, 3);
      Old := Get_Wo (MG, 2, 3);
      Set_Wo (MG, 2, 3, Old + Eps);
      Lp := Loss;
      Set_Wo (MG, 2, 3, Old - Eps);
      Lm := Loss;
      Set_Wo (MG, 2, 3, Old);
      Ng := (Lp - Lm) / (2.0 * Eps);
      Put_Line ("  attn grad check dWo: analytic =" & Real'Image (Ag)
                & " numeric =" & Real'Image (Ng));
      Check (abs (Ag - Ng) < 1.0e-4, "attn gradient check dWo");
      Ag := Dxg (1, 2);
      Old := Xg (1, 2);
      Xg (1, 2) := Old + Eps;
      Lp := Loss;
      Xg (1, 2) := Old - Eps;
      Lm := Loss;
      Xg (1, 2) := Old;
      Ng := (Lp - Lm) / (2.0 * Eps);
      Put_Line ("  attn grad check dX: analytic =" & Real'Image (Ag)
                & " numeric =" & Real'Image (Ng));
      Check (abs (Ag - Ng) < 1.0e-4, "attn gradient check dX");
   end;
   --  End-to-end: each position learns to retrieve a different token.
   --  The value path starts as identity, so the three learned weight
   --  sets must learn the semantic routing themselves.
   declare
      ML : Multi_Head := Make_Multi_Head (2, 4, 4, 4);
      Xl : Real_Bank (1 .. 4, 1 .. 3) :=
        (1 => (1.0, 0.0, 0.2),
         2 => (0.0, 1.0, 0.3),
         3 => (0.0, 0.0, 0.9),
         4 => (0.2, 0.1, 0.4));
      Tl : Real_Bank (1 .. 4, 1 .. 3);
      Yl : Real_Bank (1 .. 4, 1 .. 3);
      Dl : Real_Bank (1 .. 4, 1 .. 3);
      Ox : Real_Bank (1 .. 4, 1 .. 3);
      As : Real_Bank (1 .. 3, 1 .. 3);
      Bs : Real_Bank (1 .. 3, 1 .. 3);
      Max_Err, Spread, Hdiff : Real;
   begin
      for H in 1 .. 2 loop
         for I in 1 .. 4 loop
            for J in 1 .. 4 loop
               if I = J then
                  Set_Wv (ML, H, I, J, 1.0);
               else
                  Set_Wv (ML, H, I, J, 0.0);
               end if;
            end loop;
         end loop;
      end loop;
      for I in 1 .. 4 loop
         for J in 1 .. 8 loop
            if J = I or else J = I + 4 then
               Set_Wo (ML, I, J, 0.5);
            else
               Set_Wo (ML, I, J, 0.0);
            end if;
         end loop;
      end loop;
      --  Permutation targets: pos1 -> tok3, pos2 -> tok1, pos3 -> tok2.
      --  Uniform attention cannot solve this; the scores must route.
      for I in 1 .. 4 loop
         for C in 1 .. 3 loop
            Tl (I, C) := Xl (I, 1 + ((C + 1) mod 3));
         end loop;
      end loop;
      for Epoch in 1 .. 1_500 loop
         Attn_Forward (ML, Xl, Yl);
         for I in 1 .. 4 loop
            for C in 1 .. 3 loop
               Dl (I, C) := Yl (I, C) - Tl (I, C);
            end loop;
         end loop;
         Attn_Backward (ML, Xl, Dl, 0.1, Ox);
      end loop;
      Attn_Forward (ML, Xl, Yl);
      Max_Err := 0.0;
      for I in 1 .. 4 loop
         for C in 1 .. 3 loop
            if abs (Yl (I, C) - Tl (I, C)) > Max_Err then
               Max_Err := abs (Yl (I, C) - Tl (I, C));
            end if;
         end loop;
      end loop;
      Put_Line ("  attn routing max err:" & Real'Image (Max_Err));
      Check (Max_Err < 0.05, "attn routing: each position retrieves its token");
      Attn_Scores (ML, 1, Xl, As);
      Attn_Scores (ML, 2, Xl, Bs);
      Spread := 0.0;
      Hdiff := 0.0;
      for I in 1 .. 3 loop
         for J in 1 .. 3 loop
            if abs (As (I, J) - 1.0 / 3.0) > Spread then
               Spread := abs (As (I, J) - 1.0 / 3.0);
            end if;
            if abs (As (I, J) - Bs (I, J)) > Hdiff then
               Hdiff := abs (As (I, J) - Bs (I, J));
            end if;
         end loop;
      end loop;
      Check (Spread > 0.1, "attn learned scores vary per query/key pair");
      Check (Hdiff > 0.05, "attn heads learn different score patterns");
   end;

   ---------------------------------------------------------
   Phase ("EDGE CASES");
   ---------------------------------------------------------
   --  Saturation: the overflow-safe sigmoid pins exactly.
   Check (Mark1_K.Sigmoid (1000.0) = 1.0, "edge: sigmoid(+1000) = 1.0 exactly");
   Check (Mark1_K.Sigmoid (-1000.0) = 0.0, "edge: sigmoid(-1000) = 0.0 exactly");
   Check (Sigmoid_D (1000.0) = 0.0
          and then Sigmoid_D (-1000.0) = 0.0,
          "edge: sigmoid derivative vanishes at extremes");
   --  Activation kinks and closed forms.
   Check (Activate (Relu_A, 0.0) = 0.0
          and then Activate (Relu_A, -2.5) = 0.0,
          "edge: relu kink at zero");
   Check (Activate_D (Sigmoid_A, 0.5) = 0.25,
          "edge: sigmoid derivative from activated output");
   Check (Activate (Tanh_A, 50.0) = 1.0,
          "edge: tanh saturates");
   --  Batchnorm with a batch of one: variance is zero, x-hat is zero,
   --  the output collapses to beta.
   declare
      B1 : BatchNorm := Make_BatchNorm (2);
      X1 : Real_Bank (1 .. 2, 1 .. 1) := ((1 => 7.0), (1 => -3.0));
      Y1 : Real_Bank (1 .. 2, 1 .. 1);
   begin
      BN_Forward_Train (B1, X1, Y1);
      Check (Y1 (1, 1) = 0.0 and then Y1 (2, 1) = 0.0,
             "edge: bn batch-of-1 collapses to beta");
      Check (abs (Get_Run_Mean (B1, 1) - 0.7) < 1.0e-9
             and then abs (Get_Run_Var (B1, 1) - 0.9) < 1.0e-9,
             "edge: bn batch-of-1 running stats");
   end;
   --  Batchnorm with a constant feature: zero variance, no NaN.
   declare
      Bc : BatchNorm := Make_BatchNorm (1);
      Xc : Real_Bank (1 .. 1, 1 .. 4) := (1 => (3.0, 3.0, 3.0, 3.0));
      Yc : Real_Bank (1 .. 1, 1 .. 4);
   begin
      BN_Forward_Train (Bc, Xc, Yc);
      Check (Yc (1, 1) = 0.0 and then Yc (1, 4) = 0.0,
             "edge: bn constant feature collapses to beta");
   end;
   --  Eta = 0.0: backward computes gradients, touches no parameters.
   declare
      Bz : BatchNorm := Make_BatchNorm (1);
      Xz : Real_Bank (1 .. 1, 1 .. 4) := (1 => (1.0, 2.0, 3.0, 4.0));
      Dz : Real_Bank (1 .. 1, 1 .. 4) := (1 => (0.5, 0.5, 0.5, 0.5));
      Oz : Real_Bank (1 .. 1, 1 .. 4);
      G0, T0 : Real;
   begin
      G0 := Get_Gamma (Bz, 1);
      T0 := Get_Beta (Bz, 1);
      BN_Backward (Bz, Xz, Dz, 0.0, Oz);
      Check (Get_Gamma (Bz, 1) = G0 and then Get_Beta (Bz, 1) = T0,
             "edge: bn eta=0 freezes gamma/beta");
   end;
   --  Degenerate linear algebra.
   declare
      Zv : Vector (1 .. 3) := (others => 0.0);
      A1 : Matrix (1 .. 1, 1 .. 1) := (1 => (1 => 2.0));
      B1 : Matrix (1 .. 1, 1 .. 1) := (1 => (1 => 3.0));
      C1 : Matrix (1 .. 1, 1 .. 1);
      Xv : Vector (1 .. 3) := (1.0, 2.0, 3.0);
      Yv : Vector (1 .. 3) := (4.0, 5.0, 6.0);
   begin
      Check (Dot (Zv, Zv) = 0.0 and then Norm2 (Zv) = 0.0,
             "edge: zero vector dot/norm");
      C1 := Matmul (A1, B1);
      Check (C1 (1, 1) = 6.0, "edge: matmul 1x1");
      Axpy (0.0, Xv, Yv);
      Check (Yv = (4.0, 5.0, 6.0), "edge: axpy alpha=0 leaves y unchanged");
   end;
   --  Max-pool tie on a uniform map: deterministic, picks the first max.
   --  (Make_Conv2D seeds kernels, so zero them explicitly first.)
   declare
      Cz  : Conv2D := Make_Conv2D (1, 4, 4, 3, 3);
      Sz  : Conv_State := Make_Conv_State (1, 2, 2);
      Pz  : Pool_State := Make_Pool_State (1, 1, 1);
      Img : Matrix (1 .. 4, 1 .. 4) := (others => (others => 5.0));
   begin
      for I in 1 .. 3 loop
         for J in 1 .. 3 loop
            Set_Kernel (Cz, 1, I, J, 0.0);
         end loop;
      end loop;
      Conv_Forward (Cz, Sz, Img);
      MaxPool_Forward (Sz, Pz);
      Check (Flatten (Pz) (1) = 0.0,
             "edge: conv zero-kernel tie pools deterministically");
   end;

   --  Attention with a single token: the only score is 1.0 exactly.
   declare
      M1 : Multi_Head := Make_Multi_Head (1, 4, 2, 2);
      X1 : Real_Bank (1 .. 4, 1 .. 1) :=
        (1 => (1 => 0.5), 2 => (1 => -0.3), 3 => (1 => 0.8), 4 => (1 => 0.1));
      S1 : Real_Bank (1 .. 1, 1 .. 1);
   begin
      Attn_Scores (M1, 1, X1, S1);
      Check (S1 (1, 1) = 1.0, "edge: attn single token weight is 1.0");
   end;

   ---------------------------------------------------------
   Phase ("REPORT");
   ---------------------------------------------------------
   Put_Line ("");
   Put_Line ("MARK-I runtime complete: " & Natural'Image (Passed)
             & " assertions held.");
   pragma Assert (Passed = 69);

end Mark1_Runtime;
