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

--  MARK-I Runtime -- Batch normalization (spec)
--
--  Per-feature batch normalization for mini-batch training:
--  normalize each feature to zero mean / unit variance over the
--  batch, then apply a learnable scale (gamma) and shift (beta).
--  Running statistics with momentum feed the inference path.
--  Pure Ada 2005, arrays, raw assertions.

with Mark1_K; use Mark1_K;

package Mark1_BatchNorm is

   pragma Assertion_Policy (Check);

   Epsilon  : constant Real := 1.0e-5;
   Momentum : constant Real := 0.9;

   type BatchNorm (Features : Positive) is private;

   function Make_BatchNorm (Features : Positive) return BatchNorm;

   function Get_Gamma (B : BatchNorm; F : Positive) return Real;
   function Get_Beta  (B : BatchNorm; F : Positive) return Real;
   procedure Set_Gamma (B : in out BatchNorm; F : Positive; V : Real);
   procedure Set_Beta  (B : in out BatchNorm; F : Positive; V : Real);
   function Get_Run_Mean (B : BatchNorm; F : Positive) return Real;
   function Get_Run_Var  (B : BatchNorm; F : Positive) return Real;

   --  Training forward over X : (F0 .. F0+Features-1, C0 .. C0+Batch-1).
   --  Normalizes with batch statistics and updates running statistics.
   --  Y must match X's shape.
   procedure BN_Forward_Train
     (B : in out BatchNorm; X : Real_Bank; Y : out Real_Bank);

   --  Inference forward: normalizes with the running statistics.
   procedure BN_Forward_Infer
     (B : BatchNorm; X : Real_Bank; Y : out Real_Bank);

   --  Gradients for the last training forward. D_Y : upstream deltas,
   --  same shape as X. D_Gamma/D_Beta : (1 .. Features).
   procedure BN_Gradients
     (B       : BatchNorm;
      X       : Real_Bank;
      D_Y     : Real_Bank;
      D_X     : out Real_Bank;
      D_Gamma : out Vector;
      D_Beta  : out Vector);

   --  Full backward: gradients, then the SGD step when Eta > 0.0.
   procedure BN_Backward
     (B    : in out BatchNorm;
      X    : Real_Bank;
      D_Y  : Real_Bank;
      Eta  : Real;
      D_X  : out Real_Bank);

private

   type BatchNorm (Features : Positive) is record
      Gamma    : Vector (1 .. Features) := (others => 1.0);
      Beta     : Vector (1 .. Features) := (others => 0.0);
      Run_Mean : Vector (1 .. Features) := (others => 0.0);
      Run_Var  : Vector (1 .. Features) := (others => 1.0);
   end record;

end Mark1_BatchNorm;
