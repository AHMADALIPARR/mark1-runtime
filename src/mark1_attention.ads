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

--  MARK-I Runtime -- Multi-head attention (spec)
--
--  Scaled dot-product multi-head self-attention, hand-rolled.
--  Semantic scoring, library-search style: each query token scores
--  every key token (Q . K / sqrt(d)), the scores become a softmax
--  distribution, and the output is the values mixed by those weights.
--  Three learned weight sets per head (Wq, Wk, Wv) plus the output
--  projection Wo. Full backpropagation with numeric gradient checks.
--  Pure Ada 2005, arrays, raw assertions.

with Mark1_K; use Mark1_K;

package Mark1_Attention is

   pragma Assertion_Policy (Check);

   --  QK_Rows = Heads * D_K, V_Rows = Heads * D_V, O_Cols = Heads * D_V.
   --  Flat discriminants: a discriminant must appear alone in a
   --  component constraint (same rule as Mark1_CNN's states).
   type Multi_Head
     (Heads, D_Model, D_K, D_V, QK_Rows, V_Rows, O_Cols : Positive)
     is private;

   function Make_Multi_Head
     (Heads, D_Model, D_K, D_V : Positive) return Multi_Head;

   --  The three learned weight sets (+ output projection), per head H.
   function Get_Wq (M : Multi_Head; H, I, J : Positive) return Real;
   function Get_Wk (M : Multi_Head; H, I, J : Positive) return Real;
   function Get_Wv (M : Multi_Head; H, I, J : Positive) return Real;
   function Get_Wo (M : Multi_Head; I, J : Positive) return Real;
   procedure Set_Wq (M : in out Multi_Head; H, I, J : Positive; V : Real);
   procedure Set_Wk (M : in out Multi_Head; H, I, J : Positive; V : Real);
   procedure Set_Wv (M : in out Multi_Head; H, I, J : Positive; V : Real);
   procedure Set_Wo (M : in out Multi_Head; I, J : Positive; V : Real);

   --  Forward: X (D_Model x T, columns are tokens) -> Y, same shape.
   procedure Attn_Forward (M : Multi_Head; X : Real_Bank; Y : out Real_Bank);

   --  Semantic scores of head H: A (T x T), rows sum to 1.
   --  A (I, J) = how much query token I attends to key token J.
   procedure Attn_Scores
     (M : Multi_Head; H : Positive; X : Real_Bank; A : out Real_Bank);

   --  Gradients. D_Wq/D_Wk : (Heads*D_K x D_Model),
   --  D_Wv : (Heads*D_V x D_Model), D_Wo : (D_Model x Heads*D_V).
   procedure Attn_Gradients
     (M                : Multi_Head;
      X, D_Y           : Real_Bank;
      D_X              : out Real_Bank;
      D_Wq, D_Wk, D_Wv : out Matrix;
      D_Wo             : out Matrix);

   --  Full backward: gradients, then the SGD step when Eta > 0.0.
   procedure Attn_Backward
     (M      : in out Multi_Head;
      X, D_Y : Real_Bank;
      Eta    : Real;
      D_X    : out Real_Bank);

private

   type Multi_Head
     (Heads, D_Model, D_K, D_V, QK_Rows, V_Rows, O_Cols : Positive)
   is record
      Wq : Matrix (1 .. QK_Rows, 1 .. D_Model);
      Wk : Matrix (1 .. QK_Rows, 1 .. D_Model);
      Wv : Matrix (1 .. V_Rows, 1 .. D_Model);
      Wo : Matrix (1 .. D_Model, 1 .. O_Cols);
   end record;

end Mark1_Attention;
