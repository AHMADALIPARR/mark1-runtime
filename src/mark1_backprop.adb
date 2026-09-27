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

--  MARK-I Runtime -- Backpropagating errors (body)

package body Mark1_Backprop is

   function Make_MLP
     (In_Dim, Hidden_Dim, Out_Dim : Positive) return MLP
   is
      N : MLP (In_Dim, Hidden_Dim, Out_Dim);
   begin
      --  Fixed asymmetric seed: reproducible, breaks symmetry, no RNG.
      for J in 1 .. Hidden_Dim loop
         for K in 1 .. In_Dim loop
            N.W1 (J, K) := 0.1 * Real (J + K) - 0.15;
         end loop;
         N.B1 (J) := 0.1 * Real (J) - 0.05;
      end loop;
      for I in 1 .. Out_Dim loop
         for J in 1 .. Hidden_Dim loop
            N.W2 (I, J) := 0.1 * Real (I + J + 1) - 0.2;
         end loop;
         N.B2 (I) := 0.0;
      end loop;
      return N;
   end Make_MLP;

   procedure Forward (N : MLP; X : Vector; H, Y : out Vector) is
      S : Real;
   begin
      pragma Assert (X'Length = N.In_Dim);
      pragma Assert (H'Length = N.Hidden_Dim);
      pragma Assert (Y'Length = N.Out_Dim);
      for J in 1 .. N.Hidden_Dim loop
         S := N.B1 (J);
         for K in 1 .. N.In_Dim loop
            S := S + N.W1 (J, K) * X (X'First + K - 1);
         end loop;
         H (H'First + J - 1) := Sigmoid (S);
      end loop;
      for I in 1 .. N.Out_Dim loop
         S := N.B2 (I);
         for J in 1 .. N.Hidden_Dim loop
            S := S + N.W2 (I, J) * H (H'First + J - 1);
         end loop;
         Y (Y'First + I - 1) := Sigmoid (S);
      end loop;
   end Forward;

   function Output_Of (N : MLP; X : Vector) return Vector is
      H : Vector (1 .. N.Hidden_Dim);
      Y : Vector (1 .. N.Out_Dim);
   begin
      Forward (N, X, H, Y);
      return Y;
   end Output_Of;

   procedure Train_Sample
     (N      : in out MLP;
      X      : Vector;
      Target : Vector;
      Eta    : Real)
   is
      H  : Vector (1 .. N.Hidden_Dim);
      Y  : Vector (1 .. N.Out_Dim);
      Delta_O : Vector (1 .. N.Out_Dim);    -- output-layer errors
      Delta_H : Vector (1 .. N.Hidden_Dim); -- hidden-layer errors, propagated back
      S  : Real;
   begin
      pragma Assert (X'Length = N.In_Dim);
      pragma Assert (Target'Length = N.Out_Dim);
      pragma Assert (Eta > 0.0);

      Forward (N, X, H, Y);

      --  Output errors: dL/dY = (Y - T) * sig'(Y)
      for I in Delta_O'Range loop
         Delta_O (I) := (Y (I) - Target (Target'First + I - Delta_O'First))
                  * Y (I) * (1.0 - Y (I));
      end loop;

      --  Backpropagate: Delta_H = (W2' * Delta_O) .* sig'(H)
      for J in Delta_H'Range loop
         S := 0.0;
         for I in Delta_O'Range loop
            S := S + N.W2 (I, J) * Delta_O (I);
         end loop;
         Delta_H (J) := S * H (J) * (1.0 - H (J));
      end loop;

      --  Gradient descent step on both layers
      for I in N.W2'Range (1) loop
         for J in N.W2'Range (2) loop
            N.W2 (I, J) := N.W2 (I, J) - Eta * Delta_O (I) * H (J);
         end loop;
         N.B2 (I) := N.B2 (I) - Eta * Delta_O (I);
      end loop;
      for J in N.W1'Range (1) loop
         for K in N.W1'Range (2) loop
            N.W1 (J, K) := N.W1 (J, K) - Eta * Delta_H (J) * X (X'First + K - 1);
         end loop;
         N.B1 (J) := N.B1 (J) - Eta * Delta_H (J);
      end loop;
   end Train_Sample;

end Mark1_Backprop;
