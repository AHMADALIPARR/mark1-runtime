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

--  MARK-I Runtime -- Convolutional neural network (spec)
--
--  Hand-rolled Conv2D + 2x2 max-pool + dense head, with full
--  backpropagation down to the kernels. A numeric gradient check
--  guards the backward pass. Pure Ada 2005, arrays, raw assertions.

with Mark1_K; use Mark1_K;

package Mark1_CNN is

   pragma Assertion_Policy (Check);

   type Real_3D is
     array (Positive range <>, Positive range <>, Positive range <>) of Real;

   --  Valid convolution: In_H x In_W --> Num_F maps of
   --  (In_H - KH + 1) x (In_W - KW + 1), ReLU activation.
   type Conv2D (Num_F, In_H, In_W, KH, KW : Positive) is private;
   type Conv_State (Num_F, Map_H, Map_W, Cells : Positive) is private;

   function Make_Conv2D
     (Num_F, In_H, In_W, KH, KW : Positive) return Conv2D;

   function Make_Conv_State (Num_F, Map_H, Map_W : Positive)
     return Conv_State;

   function Out_H (C : Conv2D) return Positive;
   function Out_W (C : Conv2D) return Positive;

   function Get_Kernel
     (C : Conv2D; F, I, J : Positive) return Real;
   procedure Set_Kernel
     (C : in out Conv2D; F, I, J : Positive; V : Real);

   procedure Conv_Forward (C : Conv2D; S : in out Conv_State; X : Matrix);

   --  2x2 max-pool, stride 2. Input maps must have even sides.
   --  The argmax mask is kept inside P for the backward pass.
   type Pool_State (Num_F, Map_H, Map_W, Cells, Mask_Cells : Positive)
     is private;

   function Make_Pool_State (Num_F, Map_H, Map_W : Positive)
     return Pool_State;

   procedure MaxPool_Forward (S_In : Conv_State; P : in out Pool_State);

   --  Filter-major, row-major flatten: (F-1)*H*W + (R-1)*W + Cc
   function Flatten (P : Pool_State) return Vector;

   --  Single sigmoid dense layer.
   type Dense_Layer (Out_Dim, In_Dim : Positive) is private;

   function Make_Dense (Out_Dim, In_Dim : Positive) return Dense_Layer;

   procedure Dense_Forward (D : Dense_Layer; X : Vector; Y : out Vector);

   --  Backward pass. Eta = 0.0 computes D_X without touching weights.
   procedure Dense_Backward
     (D      : in out Dense_Layer;
      X      : Vector;
      Y      : Vector;
      Target : Vector;
      Eta    : Real;
      D_X    : out Vector);

   --  Analytic kernel/bias gradients for the current forward state.
   --  D_Pool : deltas at the pool output, Real_Bank (1..Num_F, 1..Map_H*Map_W).
   procedure Conv_Gradients
     (C      : Conv2D;
      S      : Conv_State;
      P      : Pool_State;
      X      : Matrix;
      D_Pool : Real_Bank;
      DK     : out Real_3D;
      DB     : out Vector);

   --  Full conv backward: gradients, then SGD step. Eta = 0.0: no update.
   procedure Conv_Backward
     (C      : in out Conv2D;
      S      : Conv_State;
      P      : Pool_State;
      X      : Matrix;
      D_Pool : Real_Bank;
      Eta    : Real);

private

   type Conv2D (Num_F, In_H, In_W, KH, KW : Positive) is record
      K : Real_3D (1 .. Num_F, 1 .. KH, 1 .. KW) :=
            (others => (others => (others => 0.0)));
      B : Vector (1 .. Num_F) := (others => 0.0);
   end record;

   type Conv_State (Num_F, Map_H, Map_W, Cells : Positive) is record
      Z : Real_Bank (1 .. Num_F, 1 .. Cells) :=
            (others => (others => 0.0));
      A : Real_Bank (1 .. Num_F, 1 .. Cells) :=
            (others => (others => 0.0));
   end record;

   type Pool_State (Num_F, Map_H, Map_W, Cells, Mask_Cells : Positive)
   is record
      A    : Real_Bank (1 .. Num_F, 1 .. Cells) :=
              (others => (others => 0.0));
      Mask : Real_Bank (1 .. Num_F, 1 .. Mask_Cells) :=
              (others => (others => 0.0));
   end record;

   type Dense_Layer (Out_Dim, In_Dim : Positive) is record
      W : Matrix (1 .. Out_Dim, 1 .. In_Dim) :=
            (others => (others => 0.0));
      B : Vector (1 .. Out_Dim) := (others => 0.0);
   end record;

end Mark1_CNN;
