<!--
SPDX-License-Identifier: AGPL-3.0-or-later

mark1-runtime
Copyright (C) 2026 SnapKitty Collective

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
-->

# MARK-I Runtime — Math reference

Every equation the runtime implements, in one place. Notation: `Real` is
`digits 15` (≈ IEEE binary64). Vectors/matrices are 1-based Ada arrays.

## K engine (`Mark1_K`)

- **Dot:** `d = Σᵢ aᵢ·bᵢ`
- **Norm2:** `‖a‖ = √(Σᵢ aᵢ²)`
- **AXPY:** `y ← α·x + y`
- **Matvec:** `yᵢ = Σⱼ Mᵢⱼ·vⱼ`
- **Matmul:** `Cᵢⱼ = Σₖ Aᵢₖ·Bₖⱼ` (naive triple loop, row-major)
- **Outer:** `Mᵢⱼ = aᵢ·bⱼ`
- **Transpose:** `Tⱼᵢ = Mᵢⱼ`

### Sigmoid (overflow-safe)

```
σ(x) = 1/(1+e^(−x))      for x ≥ 0
σ(x) = e^x/(1+e^x)       for x < 0
```

No branch ever evaluates `e^(+1000)`. `σ(1000) = 1.0` and
`σ(−1000) = 0.0` exactly. Derivative from the activated value:

```
σ'(x) = σ(x)·(1 − σ(x))        (Sigmoid_D takes the activated output)
```

### Learning transforms

- **Hebb:** `W ← W + η·y·xᵀ` (outer product of post- and pre-synaptic activity)
- **Perceptron update:** on misclassification only,
  `w ← w + η·d·x`, `b ← b + η·d`, where `d ∈ {+1, −1}` is the desired
  response and the response is `sign(w·x + b)`.

## Rosenblatt perceptron (`Mark1_Rosenblatt`)

```
response = sign(W·stimulus + b) ∈ {+1, −1}
```

`Feedback` applies the perceptron update above **only on
misclassification** — the correction loop from the diagram:
`stimulus → response`, with `feedback` closing the loop.

## Backprop MLP (`Mark1_Backprop`, `Mark1_Net`)

Two-layer (or arbitrary-depth) feedforward net. For layer `l` with
pre-activation `zˡ = Wˡ·aˡ⁻¹ + bˡ` and activation `aˡ = f(zˡ)`:

```
δᴸ = (aᴸ − t) ⊙ f'(zᴸ)              output error (loss ½‖aᴸ−t‖²)
δˡ = (Wˡ⁺¹ᵀ·δˡ⁺¹) ⊙ f'(zˡ)          backpropagated error
∂L/∂Wˡ = δˡ·(aˡ⁻¹)ᵀ ,  ∂L/∂bˡ = δˡ   gradients
Wˡ ← Wˡ − η·∂L/∂Wˡ                  SGD step
```

`Mark1_Net.Activate_D` takes the *activated* output `y = f(z)` and
returns `f'(z)` cheaply:

| Activation | `Activate` | `Activate_D(F, Y)` |
|---|---|---|
| Linear | `x` | `1` |
| Sigmoid | `σ(x)` | `y·(1−y)` |
| Tanh | `tanh(x)` | `1−y²` |
| ReLU | `max(0, x)` | `1` if `y > 0` else `0` |

The CNN dense head uses loss `½·E²` so that its delta is exactly
`E·σ'(z)` — the numeric gradient check depends on this convention.

## CNN (`Mark1_CNN`)

**Valid Conv2D + ReLU.** For filter `f`, kernel `K`, bias `b`:

```
Z(f, r, c) = b(f) + ΣᵢΣⱼ K(f, i, j)·X(r+i−1, c+j−1)
A(f, r, c) = max(0, Z(f, r, c))
```

Output maps are `(In_H − KH + 1) × (In_W − KW + 1)`. Kernels are seeded
deterministically: `K(f,i,j) = 0.1·(f+i+j) − 0.2` (no RNG).

**2×2 max-pool, stride 2** (even-sided maps only):

```
P(f, r, c) = max over the 2×2 window of A
```

The argmax is stored in a one-hot mask (`P.Mask`) for the backward pass.
Ties resolve deterministically to the first maximum (`>` comparison, so
the earliest cell wins).

**Flatten** is filter-major, row-major:
`flat((f−1)·H·W + (r−1)·W + c) = P(f, r, c)`.

**Conv gradients** (`Conv_Gradients`), given pool-output deltas `D_Pool`:

```
dK(f, i, j) = Σ over output positions of D_up(f, r, c)·X(r+i−1, c+j−1)
dB(f)       = Σ over output positions of D_up(f, r, c)
```

where `D_up` is `D_Pool` scattered back through the argmax mask and the
ReLU gate. `Conv_Backward` then applies the SGD step (`η = 0.0`
computes gradients without touching weights).

## Batch normalization (`Mark1_BatchNorm`)

Per feature `f` over a batch of `m` samples, with `ε = 1e−5`:

```
μ(f)    = (1/m)·Σc x(f, c)
σ²(f)   = (1/m)·Σc (x(f, c) − μ(f))²
x̂(f,c)  = (x(f, c) − μ(f)) / √(σ²(f) + ε)
y(f,c)  = γ(f)·x̂(f, c) + β(f)
```

**Running statistics** (momentum `0.9`), updated on every training forward:

```
run_μ ← 0.9·run_μ + 0.1·μ          (initialized 0)
run_σ² ← 0.9·run_σ² + 0.1·σ²        (initialized 1)
```

**Inference** normalizes with the running statistics instead of batch
statistics.

**Backward** (`BN_Gradients`), given upstream deltas `dy`:

```
dβ = Σc dy(c)
dγ = Σc dy(c)·x̂(c)
dx̂(c) = dy(c)·γ
dσ² = Σc dx̂(c)·(x(c)−μ)·(−½)·(σ²+ε)^(−3/2)
dμ  = Σc dx̂(c)·(−1/√(σ²+ε))        (the dσ²·Σ(x−μ) term is zero)
dx(c) = dx̂(c)/√(σ²+ε) + dσ²·2(x(c)−μ)/m + dμ/m
```

`BN_Backward` calls `BN_Gradients`, then applies SGD to γ/β when
`η > 0.0`. The backward pass recomputes the batch statistics from `X`,
so no forward activations are cached.

## Multi-head attention (`Mark1_Attention`)

`X` is `(d_model × T)`; columns are tokens. Per head `h`, three
learned weight sets project the tokens:

```
Q_h = Wq_h · X        (d_k × T)   -- the search request
K_h = Wk_h · X        (d_k × T)   -- the catalog index
V_h = Wv_h · X        (d_v × T)   -- the catalog contents
```

Semantic scores (the library search): every query scores every key.

```
S_h(i, j) = (Q_h(:,i) · K_h(:,j)) / √(d_k)
A_h(i, j) = exp(S_h(i,j) − max_j S_h(i,j)) / Σ_k exp(S_h(i,k) − max_j …)
```

Rows of `A_h` sum to 1 (max subtraction keeps the softmax stable).
Head output mixes the values by the scores; heads concatenate, then
the output projection `Wo` maps back to model dimension:

```
O_h = V_h · A_h'                      (d_v × T)
Y   = Wo · concat(O_1 … O_H)          (d_model × T)
```

**Backward** (`Attn_Gradients`), given upstream `dY`, recomputing the
forward pass (no activation cache, like BN):

```
dWo   = dY · concat'
dHO   = Wo' · dY
dV    = dO_h · A_h            dA = dO_h' · V_h
dS(i,j) = A(i,j)·(dA(i,j) − Σ_k dA(i,k)·A(i,k))     (softmax)
dQ    = K_h · dS' / √(d_k)    dK = Q_h · dS / √(d_k)
dWq_h = dQ · X'               dWk_h = dK · X'       dWv_h = dV · X'
dX   += Wq_h'·dQ + Wk_h'·dK + Wv_h'·dV            (summed over heads)
```

`Attn_Backward` runs the gradients, then applies SGD to all four
weight sets when `η > 0.0`. Discriminants use explicit flat sizes
(`QK_Rows = Heads·d_k`, …): a discriminant must appear alone in a
component constraint (same rule as the CNN states).

## Demos and their numbers (native x86-64, GNAT 13.3.0, `-O2`)

| Demo | Result |
|---|---|
| XOR (2-layer sigmoid MLP) | max abs error `1.30869741749590E-02` |
| sin(x) regression (custom net) | MSE `2.30559611268966E-07` |
| CNN gradient check | analytic `1.14440168575754E-03` vs numeric `1.14440168674790E-03` (9 digits) |
| CNN bar classification | 8/8 |
| BN gradient checks | dX and dγ analytic vs numeric agree to ~1e-10 |
| BN MLP XOR (train + infer paths) | 4/4 |
| Attention gradient checks | dWq, dWo, dX analytic vs numeric agree to ~1e-10 |
| Attention library search | query retrieves the matching document; scores vary per pair; heads retrieve different documents |
| Attention routing (2 heads, 1500 epochs) | max abs error `8.88178419700125E-16` |
