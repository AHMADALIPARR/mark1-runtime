% SPDX-License-Identifier: AGPL-3.0-or-later
%
% mark1-runtime
% Copyright (C) 2026 SnapKitty Collective
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU Affero General Public License as published
% by the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU Affero General Public License for more details.
%
% You should have received a copy of the GNU Affero General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.
%

% Curried Vector Math
curried_vector_math(A, B, C) :-
    call(A, B, C).

% Dot product
dot_product(A, B, C) :-
    call(dot_product, A, B, C).

% Scalar multiplication
scalar_mul(A, B, C) :-
    call(scalar_mul, A, B, C).

% Matrix addition
matrix_add(A, B, C) :-
    call(matrix_add, A, B, C).

% Curried Attention Kernel
curried_attention_kernel(W_q, W_k, W_v, Q, K, V) :-
    % Partially apply W_q to Q
    call(W_q_to_Q, Q, W_q, A).

% Partially apply W_k to K
    call(W_k_to_K, K, W_k, B).

% Calculate the Attention score
    dot_product(A, B, C).

% Normalize the Attention score
    matrix_add(C, 1, D).
    matrix_mul(D, sqrt(num_elems(V)), E).
    matrix_div(E, num_elems(V), F).
    matrix_exp(F, G).

% Calculate the Attention weights
    matrix_mul(A, G, H).

% Partially apply W_v to V
    call(W_v_to_V, V, W_v, I).

% Calculate the Context vector
    matrix_mul(H, I, Context). % Curried Feed-Forward Block
curried_feed_forward(W_1, W_2, W_3, Input, Output) :-
    call(W_1_to_Input, Input, W_1, A).

    call(W_2_to_A, A, W_2, B).

    call(W_3_to_B, B, W_3, Output). % Execution Pipeline
build_transformer(Input, Output) :-
    % Partially apply network weights
    curried_vector_math(W_q, W_k, W_v).
    curried_attention_kernel(W_q, W_k, W_v, Input, Input, Context).

    % Partially apply feed-forward weights
    curried_feed_forward(W_1, W_2, W_3, Input, Output).
