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

--  MARK-I Runtime -- benchmarks (driver)
--
--  Timing harness for the K array engine and every learning block:
--  MatMul scaling, MatVec, Sigmoid, BatchNorm forward/backward
--  (including the batch-of-1 edge case), one CNN step, one MLP
--  SGD step. Deterministic data, no RNG. Checksums defeat
--  dead-code elimination.

with Ada.Text_IO;      use Ada.Text_IO;
with Ada.Real_Time;    use Ada.Real_Time;
with Mark1_K;          use Mark1_K;
with Mark1_Net;        use Mark1_Net;
with Mark1_CNN;        use Mark1_CNN;
with Mark1_BatchNorm;  use Mark1_BatchNorm;
with Mark1_Attention;  use Mark1_Attention;

procedure Mark1_Bench is

   function Ms (A, B : Time) return Float is
   begin
      return Float (To_Duration (B - A)) * 1000.0;
   end Ms;

   procedure Fill (M : out Matrix) is
   begin
      for I in M'Range (1) loop
         for J in M'Range (2) loop
            M (I, J) := Real ((I * J) mod 13) * 0.05 - 0.3;
         end loop;
      end loop;
   end Fill;

   procedure Bench_Matmul (N : Positive; Reps : Positive) is
      A, B, C : Matrix (1 .. N, 1 .. N);
      T0, T1  : Time;
      Per_Op  : Float;
   begin
      Fill (A);
      Fill (B);
      T0 := Clock;
      for R in 1 .. Reps loop
         C := Matmul (A, B);
      end loop;
      T1 := Clock;
      Per_Op := Ms (T0, T1) / Float (Reps);
      Put_Line ("  matmul" & Positive'Image (N)
                & ":" & Float'Image (Per_Op) & " ms/op,"
                & Float'Image (2.0 * Float (N) ** 3 / (Per_Op / 1000.0)
                               / 1.0e9) & " GFLOPS"
                & "  chk" & Real'Image (C (1, 1) + C (N, N)));
   end Bench_Matmul;

   procedure Bench_Matvec (N : Positive; Reps : Positive) is
      A  : Matrix (1 .. N, 1 .. N);
      V  : Vector (1 .. N);
      W  : Vector (1 .. N) := (others => 0.0);
      T0, T1 : Time;
      Per_Op : Float;
   begin
      Fill (A);
      for I in 1 .. N loop
         V (I) := Real (I mod 7) * 0.1 - 0.3;
      end loop;
      T0 := Clock;
      for R in 1 .. Reps loop
         W := Matvec (A, V);
      end loop;
      T1 := Clock;
      Per_Op := Ms (T0, T1) / Float (Reps);
      Put_Line ("  matvec" & Positive'Image (N)
                & ":" & Float'Image (Per_Op) & " ms/op"
                & "  chk" & Real'Image (W (1) + W (N)));
   end Bench_Matvec;

   procedure Bench_Sigmoid (N : Positive; Reps : Positive) is
      V, W : Vector (1 .. N);
      T0, T1 : Time;
      Per_Op : Float;
   begin
      for I in 1 .. N loop
         V (I) := Real (I mod 21) * 0.2 - 2.0;
      end loop;
      T0 := Clock;
      for R in 1 .. Reps loop
         W := Sigmoid (V);
      end loop;
      T1 := Clock;
      Per_Op := Ms (T0, T1) / Float (Reps);
      Put_Line ("  sigmoid" & Positive'Image (N)
                & ":" & Float'Image (Per_Op) & " ms/op"
                & "  chk" & Real'Image (W (1) + W (N)));
   end Bench_Sigmoid;

   procedure Bench_BN (F, Batch : Positive; Reps : Positive) is
      B        : BatchNorm := Make_BatchNorm (F);
      X, Y     : Real_Bank (1 .. F, 1 .. Batch);
      DY, DX   : Real_Bank (1 .. F, 1 .. Batch);
      Dg, Db   : Vector (1 .. F);
      T0, T1   : Time;
      Fwd, Bwd : Float;
   begin
      for I in 1 .. F loop
         for J in 1 .. Batch loop
            X (I, J) := Real ((I * J) mod 17) * 0.05 - 0.4;
            DY (I, J) := Real ((I + J) mod 11) * 0.05 - 0.25;
         end loop;
      end loop;
      T0 := Clock;
      for R in 1 .. Reps loop
         BN_Forward_Train (B, X, Y);
      end loop;
      T1 := Clock;
      Fwd := Ms (T0, T1) / Float (Reps);
      T0 := Clock;
      for R in 1 .. Reps loop
         BN_Gradients (B, X, DY, DX, Dg, Db);
      end loop;
      T1 := Clock;
      Bwd := Ms (T0, T1) / Float (Reps);
      Put_Line ("  batchnorm f" & Positive'Image (F)
                & " b" & Positive'Image (Batch)
                & ": fwd" & Float'Image (Fwd)
                & " ms, bwd" & Float'Image (Bwd) & " ms"
                & "  chk" & Real'Image (Y (1, 1) + DX (F, Batch)));
   end Bench_BN;

   procedure Bench_CNN (Reps : Positive) is
      C   : Conv2D := Make_Conv2D (4, 16, 16, 3, 3);
      S   : Conv_State := Make_Conv_State (4, 14, 14);
      P   : Pool_State := Make_Pool_State (4, 7, 7);
      Img : Matrix (1 .. 16, 1 .. 16);
      D   : Dense_Layer := Make_Dense (1, 4 * 7 * 7);
      Xv  : Vector (1 .. 4 * 7 * 7);
      Yv  : Vector (1 .. 1);
      Tgv : Vector (1 .. 1) := (1 => 1.0);
      Dp  : Real_Bank (1 .. 4, 1 .. 49);
      Dx  : Vector (1 .. 4 * 7 * 7);
      T0, T1   : Time;
      Fwd, Bwd : Float;
   begin
      Fill (Img);
      for F in 1 .. 4 loop
         for K in 1 .. 49 loop
            Dp (F, K) := Real ((F * K) mod 9) * 0.02 - 0.08;
         end loop;
      end loop;
      T0 := Clock;
      for R in 1 .. Reps loop
         Conv_Forward (C, S, Img);
         MaxPool_Forward (S, P);
         Xv := Flatten (P);
         Dense_Forward (D, Xv, Yv);
      end loop;
      T1 := Clock;
      Fwd := Ms (T0, T1) / Float (Reps);
      T0 := Clock;
      for R in 1 .. Reps loop
         Dense_Backward (D, Xv, Yv, Tgv, 0.0, Dx);
         Conv_Backward (C, S, P, Img, Dp, 0.0);
      end loop;
      T1 := Clock;
      Bwd := Ms (T0, T1) / Float (Reps);
      Put_Line ("  cnn 16x16/4f step: fwd" & Float'Image (Fwd)
                & " ms, bwd" & Float'Image (Bwd) & " ms"
                & "  chk" & Real'Image (Yv (1) + Dx (1)));
   end Bench_CNN;

   procedure Bench_MLP (Reps : Positive) is
      Net : Custom_Net :=
        Make_Net ((1, 16, 16, 1), (Linear_A, Tanh_A, Tanh_A, Sigmoid_A));
      St  : Net_State (Depth => 4, Width => 16);
      X   : Vector (1 .. 1) := (1 => 0.5);
      Tg  : Vector (1 .. 1) := (1 => 0.8);
      T0, T1 : Time;
      Per_Op : Float;
   begin
      T0 := Clock;
      for R in 1 .. Reps loop
         Train_Sample (Net, St, X, Tg, 0.05);
      end loop;
      T1 := Clock;
      Per_Op := Ms (T0, T1) / Float (Reps);
      Put_Line ("  mlp 1-16-16-1 sgd step:" & Float'Image (Per_Op)
                & " ms" & "  chk" & Real'Image (Output_Of (Net, St) (1)));
   end Bench_MLP;

   procedure Bench_Attn
     (Heads, D_Model, D_K, T, Reps : Positive)
   is
      M        : Multi_Head := Make_Multi_Head (Heads, D_Model, D_K, D_K);
      X, Y     : Real_Bank (1 .. D_Model, 1 .. T);
      DY, DX   : Real_Bank (1 .. D_Model, 1 .. T);
      DWq      : Matrix (1 .. Heads * D_K, 1 .. D_Model);
      DWk      : Matrix (1 .. Heads * D_K, 1 .. D_Model);
      DWv      : Matrix (1 .. Heads * D_K, 1 .. D_Model);
      DWo      : Matrix (1 .. D_Model, 1 .. Heads * D_K);
      T0, T1   : Time;
      Fwd, Bwd : Float;
   begin
      for I in 1 .. D_Model loop
         for J in 1 .. T loop
            X (I, J) := Real ((I * J) mod 13) * 0.05 - 0.3;
            DY (I, J) := Real ((I + J) mod 7) * 0.05 - 0.15;
         end loop;
      end loop;
      T0 := Clock;
      for R in 1 .. Reps loop
         Attn_Forward (M, X, Y);
      end loop;
      T1 := Clock;
      Fwd := Ms (T0, T1) / Float (Reps);
      T0 := Clock;
      for R in 1 .. Reps loop
         Attn_Gradients (M, X, DY, DX, DWq, DWk, DWv, DWo);
      end loop;
      T1 := Clock;
      Bwd := Ms (T0, T1) / Float (Reps);
      Put_Line ("  attn h" & Positive'Image (Heads)
                & " d" & Positive'Image (D_Model)
                & " t" & Positive'Image (T)
                & ": fwd" & Float'Image (Fwd)
                & " ms, bwd" & Float'Image (Bwd) & " ms"
                & "  chk" & Real'Image (Y (1, 1) + DX (D_Model, T)));
   end Bench_Attn;

begin
   Put_Line ("MARK-I benchmarks (-O2, deterministic data)");
   Put_Line ("-- level 3 --");
   Bench_Matmul (16, 2000);
   Bench_Matmul (32, 500);
   Bench_Matmul (64, 100);
   Bench_Matmul (128, 15);
   Put_Line ("-- level 2 / activations --");
   Bench_Matvec (256, 2000);
   Bench_Sigmoid (4096, 2000);
   Put_Line ("-- batchnorm --");
   Bench_BN (64, 32, 500);
   Bench_BN (64, 1, 500);
   Put_Line ("-- networks --");
   Bench_CNN (100);
   Bench_MLP (200);
   Put_Line ("-- attention --");
   Bench_Attn (2, 32, 16, 16, 100);
   Bench_Attn (4, 64, 16, 32, 50);
   Put_Line ("done.");
end Mark1_Bench;
