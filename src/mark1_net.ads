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

--  MARK-I Runtime -- Hand-rolled custom neural network (spec)
--
--  A from-scratch feedforward network: arbitrary depth, per-layer
--  activation choice, full error backpropagation, SGD. No frameworks,
--  no Python -- just Ada arrays and raw assertions.

with Mark1_K; use Mark1_K;

package Mark1_Net is

   pragma Assertion_Policy (Check);

   Max_Depth : constant Positive := 8;
   Max_Width : constant Positive := 32;

   type Size_Array is array (Positive range <>) of Positive;

   type Act_Fn is (Linear_A, Sigmoid_A, Tanh_A, Relu_A);
   type Act_Array is array (Positive range <>) of Act_Fn;

   type Custom_Net (Depth : Positive; Width : Positive) is private;
   type Net_State (Depth : Positive; Width : Positive) is private;

   function Activate (F : Act_Fn; X : Real) return Real;
   function Activate_D (F : Act_Fn; Y : Real) return Real;
   --  Derivative w.r.t. the activated output Y (cheap, no re-evaluation).

   --  Sizes = neurons per layer, e.g. (1, 12, 12, 1).
   --  Acts  = activation per layer; Acts (1) is ignored (input layer).
   function Make_Net (Sizes : Size_Array; Acts : Act_Array) return Custom_Net;

   --  Forward pass; all layer activations land in S for the backward pass.
   procedure Forward (N : Custom_Net; S : in out Net_State; X : Vector);

   function Output_Of (N : Custom_Net; S : Net_State) return Vector;

   --  One SGD step: forward, backpropagate errors, descend the gradient.
   procedure Train_Sample
     (N      : in out Custom_Net;
      S      : in out Net_State;
      X      : Vector;
      Target : Vector;
      Eta    : Real);

private

   type Weight_3D is
     array (Positive range <>, Positive range <>, Positive range <>) of Real;

   type Custom_Net (Depth : Positive; Width : Positive) is record
      Sizes : Size_Array (1 .. Depth);
      Acts  : Act_Array (1 .. Depth);
      W     : Weight_3D (1 .. Depth, 1 .. Width, 1 .. Width);
      B     : Real_Bank (1 .. Depth, 1 .. Width);
   end record;

   type Net_State (Depth : Positive; Width : Positive) is record
      A : Real_Bank (1 .. Depth, 1 .. Width);
   end record;

end Mark1_Net;
