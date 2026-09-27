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

--  MARK-I Runtime -- Hand-rolled custom neural network (body)

package body Mark1_Net is

   function Activate (F : Act_Fn; X : Real) return Real is
   begin
      case F is
         when Linear_A  => return X;
         when Sigmoid_A => return Sigmoid (X);
         when Tanh_A    => return Math.Tanh (X);
         when Relu_A    =>
            if X > 0.0 then
               return X;
            else
               return 0.0;
            end if;
      end case;
   end Activate;

   function Activate_D (F : Act_Fn; Y : Real) return Real is
   begin
      case F is
         when Linear_A  => return 1.0;
         when Sigmoid_A => return Y * (1.0 - Y);
         when Tanh_A    => return 1.0 - Y * Y;
         when Relu_A    =>
            if Y > 0.0 then
               return 1.0;
            else
               return 0.0;
            end if;
      end case;
   end Activate_D;

   function Make_Net (Sizes : Size_Array; Acts : Act_Array) return Custom_Net is
      Wd : Positive := 1;
   begin
      pragma Assert (Sizes'Length >= 2 and Sizes'Length <= Max_Depth);
      pragma Assert (Acts'Length = Sizes'Length);
      for I in Sizes'Range loop
         pragma Assert (Sizes (I) in 1 .. Max_Width);
         if Sizes (I) > Wd then
            Wd := Sizes (I);
         end if;
      end loop;
      declare
         N : Custom_Net (Depth => Sizes'Length, Width => Wd);
      begin
         for L in 1 .. N.Depth loop
            N.Sizes (L) := Sizes (Sizes'First + L - 1);
            N.Acts (L)  := Acts (Acts'First + L - 1);
         end loop;
         --  Fixed asymmetric seed: reproducible, breaks symmetry, no RNG.
         --  Scaled by fan-in so tanh units start in their active region.
         for L in 2 .. N.Depth loop
            for O in 1 .. N.Sizes (L) loop
               for I in 1 .. N.Sizes (L - 1) loop
                  N.W (L, O, I) :=
                    (0.1 * Real (L + O + I) - 0.2) / Real (N.Sizes (L - 1));
               end loop;
               N.B (L, O) := 0.05 * Real (O) - 0.025;
            end loop;
         end loop;
         return N;
      end;
   end Make_Net;

   procedure Forward (N : Custom_Net; S : in out Net_State; X : Vector) is
      Acc : Real;
   begin
      pragma Assert (S.Depth = N.Depth and S.Width = N.Width);
      pragma Assert (X'Length = N.Sizes (1));
      for I in 1 .. N.Sizes (1) loop
         S.A (1, I) := X (X'First + I - 1);
      end loop;
      for L in 2 .. N.Depth loop
         for O in 1 .. N.Sizes (L) loop
            Acc := N.B (L, O);
            for I in 1 .. N.Sizes (L - 1) loop
               Acc := Acc + N.W (L, O, I) * S.A (L - 1, I);
            end loop;
            S.A (L, O) := Activate (N.Acts (L), Acc);
         end loop;
      end loop;
   end Forward;

   function Output_Of (N : Custom_Net; S : Net_State) return Vector is
      R : Vector (1 .. N.Sizes (N.Depth));
   begin
      pragma Assert (S.Depth = N.Depth and S.Width = N.Width);
      for O in 1 .. N.Sizes (N.Depth) loop
         R (O) := S.A (N.Depth, O);
      end loop;
      return R;
   end Output_Of;

   procedure Train_Sample
     (N      : in out Custom_Net;
      S      : in out Net_State;
      X      : Vector;
      Target : Vector;
      Eta    : Real)
   is
      D   : Real_Bank (1 .. N.Depth, 1 .. N.Width) :=
              (others => (others => 0.0));
      Acc : Real;
      Top : constant Positive := N.Depth;
   begin
      pragma Assert (S.Depth = N.Depth and S.Width = N.Width);
      pragma Assert (X'Length = N.Sizes (1));
      pragma Assert (Target'Length = N.Sizes (Top));
      pragma Assert (Eta > 0.0);

      Forward (N, S, X);

      --  Output errors
      for O in 1 .. N.Sizes (Top) loop
         D (Top, O) := (S.A (Top, O) - Target (Target'First + O - 1))
                       * Activate_D (N.Acts (Top), S.A (Top, O));
      end loop;

      --  Backpropagate errors down the hidden layers
      for L in reverse 2 .. Top - 1 loop
         for I in 1 .. N.Sizes (L) loop
            Acc := 0.0;
            for O in 1 .. N.Sizes (L + 1) loop
               Acc := Acc + N.W (L + 1, O, I) * D (L + 1, O);
            end loop;
            D (L, I) := Acc * Activate_D (N.Acts (L), S.A (L, I));
         end loop;
      end loop;

      --  Descend the gradient
      for L in 2 .. Top loop
         for O in 1 .. N.Sizes (L) loop
            for I in 1 .. N.Sizes (L - 1) loop
               N.W (L, O, I) :=
                 N.W (L, O, I) - Eta * D (L, O) * S.A (L - 1, I);
            end loop;
            N.B (L, O) := N.B (L, O) - Eta * D (L, O);
         end loop;
      end loop;
   end Train_Sample;

end Mark1_Net;
