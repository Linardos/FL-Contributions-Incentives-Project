# LC / OR / MR explainer

## One-line summary

The method treats each client as a player in a cooperative game, uses model performance as the coalition utility, and then allocates value either with:

- Shapley value
- Least core
- A multi-round Shapley accumulation (`lambda-MR`)

The main trick is how coalition utilities are produced cheaply without retraining every coalition from scratch.

## Shared setup

Across the original notebooks, the pipeline is:

1. Split the training set across `N` clients.
2. Inject different label-noise rates per client, so client quality differs.
3. Train a standard FL global model with FedAvg.
4. Track extra submodels so coalition utilities can be estimated.
5. Evaluate each coalition/submodel on a held-out test set.
6. Use those accuracies as the characteristic function `v(S)` of the cooperative game.

Important implementation detail:

- `average_weights(...)` does weighted FedAvg using client data fractions.
- `calculate_gradients(...)` computes each client's local update as `local_weights - global_weights`.
- `update_weights_from_gradients(...)` applies a stored update onto another model state.

## One Round (OR) Approximation

In the `OR` notebooks, we do **not** train every coalition explicitly.

Instead:

1. In each FL round, every client trains locally starting from the current global model.
2. You compute each client's gradient/update relative to that global model.
3. You maintain a separate submodel for each singleton client.
4. Each singleton submodel is updated only with that client's own gradient at every round.
5. After training finishes, for any coalition `S`, you approximate the coalition model by averaging the final singleton submodels of members in `S`.
6. You evaluate that averaged model on the test set and call the resulting accuracy `v(S)`.

So the OR approximation is:

- train single-client trajectories once
- reconstruct larger coalitions afterward by averaging those final singleton models

**This approximation is cheap (trivial compute overhead, model_size x N_clients memory overhead) but can lack fidelity.**

`OR` is the cheap coalition-utility approximation layer. The allocation rule (SV/LC) comes after that.

## Multi-Round (MR) approximation

`MR` is more expensive than OR, but still much cheaper than true, full coalition tracking.

Instead of only tracking singleton submodels and reconstructing coalitions at the end, it tracks a submodel for **every coalition** during training.

Per round:

1. Train all clients locally from the current global model.
2. Compute each client's gradient/update.
3. For every coalition `S`, average the gradients of clients in `S`.
4. Apply that averaged coalition gradient to the coalition's own stored submodel.
5. Baked-in SV calculations in the code:
    1. Evaluate every coalition submodel immediately.
    2. Compute Shapley values from that round's coalition utilities.
    3. Then you aggregate those per-round Shapley values over time:
        - `FedShap = sum_t phi_t`
        - `lambda-MR = sum_t phi_t * decay_t / epoch_sum_t`



So `lambda-MR` is a discounted multi-round contribution score: later rounds count less, and each round is normalized by the total Shapley mass in that round.

## LC: what least core is doing here

`LC` is not a separate training method. It is a different **allocation rule** applied to the same coalition-utility table produced by OR/MR.

Once you have `accuracy_dict`, where each coalition has a utility `v(S)`, you solve:

- variables `x_i >= 0`
- slack variable `e`
- minimize `e`
- subject to:
  - `sum_i x_i = v(N)` for the grand coalition
  - `sum_{i in S} x_i + e >= v(S)` for every coalition `S`

Interpretation:

- `x_i` is the payment/allocation to client `i`
- `e` is the worst coalition dissatisfaction
- minimizing `e` finds the most stable allocation in the least-core sense

TL;DR Shapley gives a fairness-style average marginal contribution. Least core gives the most coalition-stable allocation under the same utility game.

