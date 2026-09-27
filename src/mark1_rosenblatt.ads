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

--  MARK-I Runtime -- Rosenblatt learning (spec)
--  stimulus -> response, with the feedback loop from the diagram:
--      stimulus --> response
--          ^          |
--          +-- feedback

with Mark1_K; use Mark1_K;

package Mark1_Rosenblatt is

   pragma Assertion_Policy (Check);

   type Perceptron (Dim : Positive) is private;

   function Make (Dim : Positive; Learning_Rate : Real := 1.0) return Perceptron;
   function Weights (P : Perceptron) return Vector;
   function Bias_Of (P : Perceptron) return Real;
   function Learning_Rate_Of (P : Perceptron) return Real;

   --  stimulus -> response : sign(W . X + b), answers +1 or -1
   function Respond (P : Perceptron; Stimulus : Vector) return Integer;

   --  feedback : Rosenblatt update, applied only on misclassification
   procedure Feedback
     (P        : in out Perceptron;
      Stimulus : Vector;
      Desired  : Integer);

private

   type Perceptron (Dim : Positive) is record
      W   : Vector (1 .. Dim) := (others => 0.0);
      B   : Real              := 0.0;
      Eta : Real              := 1.0;
   end record;

end Mark1_Rosenblatt;
