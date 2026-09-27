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

--  MARK-I Runtime -- Multi-head attention (body)

package body Mark1_Attention is

   function Make_Multi_Head
     (Heads, D_Model, D_K, D_V : Positive) return Multi_Head
   is
      M : Multi_Head (Heads, D_Model, D_K, D_V,
                      Heads * D_K, Heads * D_V, Heads * D_V);
   begin
      --  Deterministic fan-in init, no RNG.
      for H in 1 .. Heads loop
         for I in 1 .. D_K loop
            for J in 1 .. D_Model loop
               M.Wq ((H - 1) * D_K + I, J) :=
                 (0.1 * Real (H + I + J) - 0.15) / Real (D_Model);
               M.Wk ((H - 1) * D_K + I, J) :=
                 (0.1 * Real (H + I + J + 1) - 0.15) / Real (D_Model);
            end loop;
         end loop;
         for I in 1 .. D_V loop
            for J in 1 .. D_Model loop
               M.Wv ((H - 1) * D_V + I, J) :=
                 (0.1 * Real (H + I + J + 2) - 0.15) / Real (D_Model);
            end loop;
         end loop;
      end loop;
      for I in 1 .. D_Model loop
         for J in 1 .. Heads * D_V loop
            M.Wo (I, J) :=
              (0.1 * Real (I + J) - 0.15) / Real (Heads * D_V);
         end loop;
      end loop;
      return M;
   end Make_Multi_Head;

   function QK_At (M : Multi_Head; H, I : Positive) return Positive is
   begin
      return (H - 1) * M.D_K + I;
   end QK_At;

   function V_At (M : Multi_Head; H, I : Positive) return Positive is
   begin
      return (H - 1) * M.D_V + I;
   end V_At;

   function Get_Wq (M : Multi_Head; H, I, J : Positive) return Real is
   begin
      return M.Wq (QK_At (M, H, I), J);
   end Get_Wq;

   function Get_Wk (M : Multi_Head; H, I, J : Positive) return Real is
   begin
      return M.Wk (QK_At (M, H, I), J);
   end Get_Wk;

   function Get_Wv (M : Multi_Head; H, I, J : Positive) return Real is
   begin
      return M.Wv (V_At (M, H, I), J);
   end Get_Wv;

   function Get_Wo (M : Multi_Head; I, J : Positive) return Real is
   begin
      return M.Wo (I, J);
   end Get_Wo;

   procedure Set_Wq (M : in out Multi_Head; H, I, J : Positive; V : Real) is
   begin
      M.Wq (QK_At (M, H, I), J) := V;
   end Set_Wq;

   procedure Set_Wk (M : in out Multi_Head; H, I, J : Positive; V : Real) is
   begin
      M.Wk (QK_At (M, H, I), J) := V;
   end Set_Wk;

   procedure Set_Wv (M : in out Multi_Head; H, I, J : Positive; V : Real) is
   begin
      M.Wv (V_At (M, H, I), J) := V;
   end Set_Wv;

   procedure Set_Wo (M : in out Multi_Head; I, J : Positive; V : Real) is
   begin
      M.Wo (I, J) := V;
   end Set_Wo;

   --  Q = Wq_h . X, K = Wk_h . X, V = Wv_h . X.
   procedure Head_Projections
     (M : Multi_Head; H, T : Positive; X : Real_Bank;
      Q, K : out Matrix; V : out Matrix)
   is
      FX  : constant Integer := X'First (1);
      CX  : constant Integer := X'First (2);
      Acc : Real;
   begin
      for I in 1 .. M.D_K loop
         for C in 1 .. T loop
            Acc := 0.0;
            for J in 1 .. M.D_Model loop
               Acc := Acc + M.Wq (QK_At (M, H, I), J)
                      * X (FX + J - 1, CX + C - 1);
            end loop;
            Q (I, C) := Acc;
            Acc := 0.0;
            for J in 1 .. M.D_Model loop
               Acc := Acc + M.Wk (QK_At (M, H, I), J)
                      * X (FX + J - 1, CX + C - 1);
            end loop;
            K (I, C) := Acc;
         end loop;
      end loop;
      for I in 1 .. M.D_V loop
         for C in 1 .. T loop
            Acc := 0.0;
            for J in 1 .. M.D_Model loop
               Acc := Acc + M.Wv (V_At (M, H, I), J)
                      * X (FX + J - 1, CX + C - 1);
            end loop;
            V (I, C) := Acc;
         end loop;
      end loop;
   end Head_Projections;

   --  Scores S = Q' . K / sqrt(d), then row-wise softmax with
   --  max subtraction. A (I, J): query I attends to key J.
   procedure Head_Scores
     (M : Multi_Head; Q, K : Matrix; T : Positive; A : out Matrix)
   is
      S     : Matrix (1 .. T, 1 .. T);
      Acc   : Real;
      Mx    : Real;
      Den   : Real;
      Scale : constant Real := 1.0 / Math.Sqrt (Real (M.D_K));
   begin
      for I in 1 .. T loop
         for J in 1 .. T loop
            Acc := 0.0;
            for D in 1 .. M.D_K loop
               Acc := Acc + Q (D, I) * K (D, J);
            end loop;
            S (I, J) := Acc * Scale;
         end loop;
      end loop;
      for I in 1 .. T loop
         Mx := S (I, 1);
         for J in 2 .. T loop
            if S (I, J) > Mx then
               Mx := S (I, J);
            end if;
         end loop;
         Den := 0.0;
         for J in 1 .. T loop
            Den := Den + Math.Exp (S (I, J) - Mx);
         end loop;
         for J in 1 .. T loop
            A (I, J) := Math.Exp (S (I, J) - Mx) / Den;
         end loop;
      end loop;
   end Head_Scores;

   procedure Attn_Scores
     (M : Multi_Head; H : Positive; X : Real_Bank; A : out Real_Bank)
   is
      T : constant Positive := X'Length (2);
      Q : Matrix (1 .. M.D_K, 1 .. T);
      K : Matrix (1 .. M.D_K, 1 .. T);
      V : Matrix (1 .. M.D_V, 1 .. T);
      Am : Matrix (1 .. T, 1 .. T);
   begin
      pragma Assert (H <= M.Heads);
      pragma Assert (X'Length (1) = M.D_Model);
      pragma Assert (A'Length (1) = T and A'Length (2) = T);
      Head_Projections (M, H, T, X, Q, K, V);
      Head_Scores (M, Q, K, T, Am);
      for I in 1 .. T loop
         for J in 1 .. T loop
            A (A'First (1) + I - 1, A'First (2) + J - 1) := Am (I, J);
         end loop;
      end loop;
   end Attn_Scores;

   procedure Attn_Forward (M : Multi_Head; X : Real_Bank; Y : out Real_Bank) is
      T   : constant Positive := X'Length (2);
      HO  : Matrix (1 .. M.O_Cols, 1 .. T);
      Q   : Matrix (1 .. M.D_K, 1 .. T);
      K   : Matrix (1 .. M.D_K, 1 .. T);
      V   : Matrix (1 .. M.D_V, 1 .. T);
      A   : Matrix (1 .. T, 1 .. T);
      Acc : Real;
      FY  : constant Integer := Y'First (1);
      CY  : constant Integer := Y'First (2);
   begin
      pragma Assert (X'Length (1) = M.D_Model);
      pragma Assert (Y'Length (1) = M.D_Model and Y'Length (2) = T);
      for H in 1 .. M.Heads loop
         Head_Projections (M, H, T, X, Q, K, V);
         Head_Scores (M, Q, K, T, A);
         --  Head output: V . A'  (mix values by the scores)
         for D in 1 .. M.D_V loop
            for I in 1 .. T loop
               Acc := 0.0;
               for J in 1 .. T loop
                  Acc := Acc + V (D, J) * A (I, J);
               end loop;
               HO (V_At (M, H, D), I) := Acc;
            end loop;
         end loop;
      end loop;
      --  Y = Wo . concat(heads)
      for I in 1 .. M.D_Model loop
         for C in 1 .. T loop
            Acc := 0.0;
            for J in 1 .. M.O_Cols loop
               Acc := Acc + M.Wo (I, J) * HO (J, C);
            end loop;
            Y (FY + I - 1, CY + C - 1) := Acc;
         end loop;
      end loop;
   end Attn_Forward;

   procedure Attn_Gradients
     (M                : Multi_Head;
      X, D_Y           : Real_Bank;
      D_X              : out Real_Bank;
      D_Wq, D_Wk, D_Wv : out Matrix;
      D_Wo             : out Matrix)
   is
      T     : constant Positive := X'Length (2);
      FX    : constant Integer := X'First (1);
      CX    : constant Integer := X'First (2);
      FD    : constant Integer := D_Y'First (1);
      CD    : constant Integer := D_Y'First (2);
      FXO   : constant Integer := D_X'First (1);
      CXO   : constant Integer := D_X'First (2);
      HO    : Matrix (1 .. M.O_Cols, 1 .. T);
      D_HO  : Matrix (1 .. M.O_Cols, 1 .. T);
      Q     : Matrix (1 .. M.D_K, 1 .. T);
      K     : Matrix (1 .. M.D_K, 1 .. T);
      V     : Matrix (1 .. M.D_V, 1 .. T);
      A     : Matrix (1 .. T, 1 .. T);
      D_A   : Matrix (1 .. T, 1 .. T);
      D_S   : Matrix (1 .. T, 1 .. T);
      D_Q   : Matrix (1 .. M.D_K, 1 .. T);
      D_K   : Matrix (1 .. M.D_K, 1 .. T);
      D_V   : Matrix (1 .. M.D_V, 1 .. T);
      Acc   : Real;
      Row   : Real;
      Scale : constant Real := 1.0 / Math.Sqrt (Real (M.D_K));
   begin
      pragma Assert (X'Length (1) = M.D_Model);
      pragma Assert (D_Y'Length (1) = M.D_Model and D_Y'Length (2) = T);
      pragma Assert (D_X'Length (1) = M.D_Model and D_X'Length (2) = T);
      pragma Assert (D_Wq'Length (1) = M.QK_Rows
                     and D_Wq'Length (2) = M.D_Model);
      pragma Assert (D_Wk'Length (1) = M.QK_Rows
                     and D_Wk'Length (2) = M.D_Model);
      pragma Assert (D_Wv'Length (1) = M.V_Rows
                     and D_Wv'Length (2) = M.D_Model);
      pragma Assert (D_Wo'Length (1) = M.D_Model
                     and D_Wo'Length (2) = M.O_Cols);

      --  Recompute the forward pass (no activation cache, like BN).
      for H in 1 .. M.Heads loop
         Head_Projections (M, H, T, X, Q, K, V);
         Head_Scores (M, Q, K, T, A);
         for D in 1 .. M.D_V loop
            for I in 1 .. T loop
               Acc := 0.0;
               for J in 1 .. T loop
                  Acc := Acc + V (D, J) * A (I, J);
               end loop;
               HO (V_At (M, H, D), I) := Acc;
            end loop;
         end loop;
      end loop;

      --  dWo = D_Y . HO',  dHO = Wo' . D_Y
      for I in 1 .. M.D_Model loop
         for J in 1 .. M.O_Cols loop
            Acc := 0.0;
            for C in 1 .. T loop
               Acc := Acc + D_Y (FD + I - 1, CD + C - 1) * HO (J, C);
            end loop;
            D_Wo (D_Wo'First (1) + I - 1, D_Wo'First (2) + J - 1) := Acc;
         end loop;
      end loop;
      for J in 1 .. M.O_Cols loop
         for C in 1 .. T loop
            Acc := 0.0;
            for I in 1 .. M.D_Model loop
               Acc := Acc + M.Wo (I, J) * D_Y (FD + I - 1, CD + C - 1);
            end loop;
            D_HO (J, C) := Acc;
         end loop;
      end loop;

      --  D_X accumulates over heads.
      for I in 1 .. M.D_Model loop
         for C in 1 .. T loop
            D_X (FXO + I - 1, CXO + C - 1) := 0.0;
         end loop;
      end loop;

      for H in 1 .. M.Heads loop
         Head_Projections (M, H, T, X, Q, K, V);
         Head_Scores (M, Q, K, T, A);

         --  dV = D_Oh . A,  dA = D_Oh' . V
         for D in 1 .. M.D_V loop
            for J in 1 .. T loop
               Acc := 0.0;
               for I in 1 .. T loop
                  Acc := Acc + D_HO (V_At (M, H, D), I) * A (I, J);
               end loop;
               D_V (D, J) := Acc;
            end loop;
         end loop;
         for I in 1 .. T loop
            for J in 1 .. T loop
               Acc := 0.0;
               for D in 1 .. M.D_V loop
                  Acc := Acc + D_HO (V_At (M, H, D), I) * V (D, J);
               end loop;
               D_A (I, J) := Acc;
            end loop;
         end loop;

         --  Softmax backward:
         --  dS (I, J) = A (I, J) * (dA (I, J) - sum_k dA (I, k) A (I, k))
         for I in 1 .. T loop
            Row := 0.0;
            for Kk in 1 .. T loop
               Row := Row + D_A (I, Kk) * A (I, Kk);
            end loop;
            for J in 1 .. T loop
               D_S (I, J) := A (I, J) * (D_A (I, J) - Row);
            end loop;
         end loop;

         --  dQ = K . dS' / sqrt(d),  dK = Q . dS / sqrt(d)
         for D in 1 .. M.D_K loop
            for I in 1 .. T loop
               Acc := 0.0;
               for J in 1 .. T loop
                  Acc := Acc + K (D, J) * D_S (I, J);
               end loop;
               D_Q (D, I) := Acc * Scale;
            end loop;
         end loop;
         for D in 1 .. M.D_K loop
            for J in 1 .. T loop
               Acc := 0.0;
               for I in 1 .. T loop
                  Acc := Acc + Q (D, I) * D_S (I, J);
               end loop;
               D_K (D, J) := Acc * Scale;
            end loop;
         end loop;

         --  Weight gradients: dW = dProj . X'
         for I in 1 .. M.D_K loop
            for J in 1 .. M.D_Model loop
               Acc := 0.0;
               for C in 1 .. T loop
                  Acc := Acc + D_Q (I, C) * X (FX + J - 1, CX + C - 1);
               end loop;
               D_Wq (D_Wq'First (1) + QK_At (M, H, I) - 1,
                     D_Wq'First (2) + J - 1) := Acc;
               Acc := 0.0;
               for C in 1 .. T loop
                  Acc := Acc + D_K (I, C) * X (FX + J - 1, CX + C - 1);
               end loop;
               D_Wk (D_Wk'First (1) + QK_At (M, H, I) - 1,
                     D_Wk'First (2) + J - 1) := Acc;
            end loop;
         end loop;
         for I in 1 .. M.D_V loop
            for J in 1 .. M.D_Model loop
               Acc := 0.0;
               for C in 1 .. T loop
                  Acc := Acc + D_V (I, C) * X (FX + J - 1, CX + C - 1);
               end loop;
               D_Wv (D_Wv'First (1) + V_At (M, H, I) - 1,
                     D_Wv'First (2) + J - 1) := Acc;
            end loop;
         end loop;

         --  dX += Wq_h' . dQ + Wk_h' . dK + Wv_h' . dV
         for J in 1 .. M.D_Model loop
            for C in 1 .. T loop
               Acc := 0.0;
               for I in 1 .. M.D_K loop
                  Acc := Acc + M.Wq (QK_At (M, H, I), J) * D_Q (I, C)
                         + M.Wk (QK_At (M, H, I), J) * D_K (I, C);
               end loop;
               for I in 1 .. M.D_V loop
                  Acc := Acc + M.Wv (V_At (M, H, I), J) * D_V (I, C);
               end loop;
               D_X (FXO + J - 1, CXO + C - 1) :=
                 D_X (FXO + J - 1, CXO + C - 1) + Acc;
            end loop;
         end loop;
      end loop;
   end Attn_Gradients;

   procedure Attn_Backward
     (M      : in out Multi_Head;
      X, D_Y : Real_Bank;
      Eta    : Real;
      D_X    : out Real_Bank)
   is
      D_Wq : Matrix (1 .. M.QK_Rows, 1 .. M.D_Model);
      D_Wk : Matrix (1 .. M.QK_Rows, 1 .. M.D_Model);
      D_Wv : Matrix (1 .. M.V_Rows, 1 .. M.D_Model);
      D_Wo : Matrix (1 .. M.D_Model, 1 .. M.O_Cols);
   begin
      pragma Assert (Eta >= 0.0);
      Attn_Gradients (M, X, D_Y, D_X, D_Wq, D_Wk, D_Wv, D_Wo);
      if Eta > 0.0 then
         for I in 1 .. M.QK_Rows loop
            for J in 1 .. M.D_Model loop
               M.Wq (I, J) := M.Wq (I, J) - Eta * D_Wq (I, J);
               M.Wk (I, J) := M.Wk (I, J) - Eta * D_Wk (I, J);
            end loop;
         end loop;
         for I in 1 .. M.V_Rows loop
            for J in 1 .. M.D_Model loop
               M.Wv (I, J) := M.Wv (I, J) - Eta * D_Wv (I, J);
            end loop;
         end loop;
         for I in 1 .. M.D_Model loop
            for J in 1 .. M.O_Cols loop
               M.Wo (I, J) := M.Wo (I, J) - Eta * D_Wo (I, J);
            end loop;
         end loop;
      end if;
   end Attn_Backward;

end Mark1_Attention;
