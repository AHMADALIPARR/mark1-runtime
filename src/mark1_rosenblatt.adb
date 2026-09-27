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

--  MARK-I Runtime -- Rosenblatt learning (body)

package body Mark1_Rosenblatt is

   function Make (Dim : Positive; Learning_Rate : Real := 1.0)
      return Perceptron
   is
   begin
      pragma Assert (Dim > 0);
      pragma Assert (Learning_Rate > 0.0);
      return (Dim => Dim,
              W   => (1 .. Dim => 0.0),
              B   => 0.0,
              Eta => Learning_Rate);
   end Make;

   function Weights (P : Perceptron) return Vector is
   begin
      return P.W;
   end Weights;

   function Bias_Of (P : Perceptron) return Real is
   begin
      return P.B;
   end Bias_Of;

   function Learning_Rate_Of (P : Perceptron) return Real is
   begin
      return P.Eta;
   end Learning_Rate_Of;

   function Respond (P : Perceptron; Stimulus : Vector) return Integer is
      S : Real;
   begin
      pragma Assert (Stimulus'Length = P.Dim);
      S := Dot (P.W, Stimulus) + P.B;
      if S >= 0.0 then
         return 1;
      else
         return -1;
      end if;
   end Respond;

   procedure Feedback
     (P        : in out Perceptron;
      Stimulus : Vector;
      Desired  : Integer)
   is
   begin
      pragma Assert (Stimulus'Length = P.Dim);
      pragma Assert (Desired = 1 or Desired = -1);
      Perceptron_Update (P.W, P.B, Stimulus, Desired, P.Eta);
   end Feedback;

end Mark1_Rosenblatt;
