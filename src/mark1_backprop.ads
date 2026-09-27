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

--  MARK-I Runtime -- Backpropagating errors (spec)
--
--  Two-layer sigmoid perceptron (In -> Hidden -> Out). The learning
--  transform the 1958 Mark-I never had: output errors are propagated
--  backwards through the weights, layer by layer, and every step down
--  the gradient is guarded by raw assertions.

with Mark1_K; use Mark1_K;

package Mark1_Backprop is

   pragma Assertion_Policy (Check);

   type MLP (In_Dim, Hidden_Dim, Out_Dim : Positive) is private;

   function Make_MLP
     (In_Dim, Hidden_Dim, Out_Dim : Positive) return MLP;

   function Sigmoid (X : Real) return Real renames Mark1_K.Sigmoid;

   --  Forward pass: H = sig(W1 X + B1); Y = sig(W2 H + B2)
   procedure Forward (N : MLP; X : Vector; H, Y : out Vector);

   function Output_Of (N : MLP; X : Vector) return Vector;

   --  One SGD step on a single sample: errors flow Y -> H -> W.
   procedure Train_Sample
     (N      : in out MLP;
      X      : Vector;
      Target : Vector;
      Eta    : Real);

private

   type MLP (In_Dim, Hidden_Dim, Out_Dim : Positive) is record
      W1 : Matrix (1 .. Hidden_Dim, 1 .. In_Dim)   := (others => (others => 0.0));
      B1 : Vector (1 .. Hidden_Dim)                := (others => 0.0);
      W2 : Matrix (1 .. Out_Dim, 1 .. Hidden_Dim)  := (others => (others => 0.0));
      B2 : Vector (1 .. Out_Dim)                   := (others => 0.0);
   end record;

end Mark1_Backprop;
