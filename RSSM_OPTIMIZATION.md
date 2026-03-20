# RSSM Optimization Opportunities

The RSSM (Recurrent State-Space Model) sequential loop is the primary training bottleneck. Each of the 50 timesteps depends on the previous step's output, forcing sequential GPU execution. This document outlines approaches to speed it up.

## Current bottleneck

In `agent.py:_train_world_model`, the RSSM loop runs 50 sequential steps per batch:

```python
for t in range(T):  # T=50
    post_state, prior_state, prior_logits = self.world_model.rssm.observe_step(
        prev_state, actions[:, t], embed[:, t]
    )
```

Each `observe_step` calls `imagine_step` (GRU + prior network + categorical sampling) then the posterior network. This launches ~10 separate CUDA kernels per timestep with idle time between launches. At batch_size=32, these are small operations that are memory-bandwidth-bound, not compute-bound.

The same sequential pattern appears in the imagination loop (`_train_actor_critic`), which runs 15 steps of `imagine_step`.

## Approach 1: Fused CUDA kernel

Fuse the GRU cell + linear layers + LayerNorm + SiLU + categorical sampling into a single CUDA kernel per RSSM step. This eliminates kernel launch overhead and intermediate memory round-trips.

- **Expected speedup**: 2-4x on the RSSM loop
- **Effort**: Medium-high (custom CUDA or Triton kernel)
- **Risk**: Low (no algorithmic change)
- **Reference**: Same idea as FlashAttention but for recurrent steps

Key operations to fuse per step:
1. `pre_gru`: Linear(stoch_flat + action_dim → hidden) + LayerNorm + SiLU
2. `GRUCell(hidden, deter)` — internal: 3 linear projections + sigmoid + tanh
3. `prior_net`: Linear(deter → hidden) + LayerNorm + SiLU + Linear(hidden → stoch_logits)
4. Categorical sampling with straight-through + unimix

A Triton kernel would be the most practical implementation path.

## Approach 2: torch.compile on the RSSM cell

Wrap `observe_step` or `imagine_step` in `torch.compile()` to let PyTorch's compiler fuse operations automatically.

- **Expected speedup**: 1.3-2x
- **Effort**: Low (one-line change + debugging)
- **Risk**: Medium (dictionary state passing and OneHotCategorical may cause graph breaks)

```python
self.world_model.rssm.imagine_step = torch.compile(self.world_model.rssm.imagine_step)
```

May require refactoring state from `dict` to `NamedTuple` or flat tensors to avoid graph breaks. Test with `TORCH_LOGS=graph_breaks` to identify issues.

## Approach 3: Linear recurrence with parallel scan

Replace the GRU with a linear recurrence (e.g., S4, S5, Mamba-style) that supports parallel prefix scan computation: O(log T) sequential depth instead of O(T).

- **Expected speedup**: 10-20x on the sequential dimension
- **Effort**: High (architecture change, revalidation needed)
- **Risk**: High (different inductive bias, may hurt world model quality)
- **References**:
  - S5: "Simplified State Space Layers for Sequence Modeling"
  - Mamba: "Linear-Time Sequence Modeling with Selective State Spaces"

The GRU's nonlinear gates are what prevent parallelization. A linear recurrence `h_t = A * h_{t-1} + B * x_t` can be computed for all T in parallel using an associative scan. The tradeoff is that linear recurrences have weaker expressivity per step, which may require wider hidden states to compensate.

## Approach 4: Reduce sequence length

The simplest option: reduce `--seq_len` from 50 to 25.

- **Expected speedup**: ~2x on the RSSM loop (~1.5x end-to-end)
- **Effort**: None (CLI flag change)
- **Risk**: Low-medium (shorter temporal context may hurt world model predictions for long-horizon dependencies)

Can be combined with any of the above approaches.

## Recommended priority

1. **torch.compile** — try first, lowest effort, may get 1.3-2x for free
2. **Fused Triton kernel** — best effort/reward ratio if compile doesn't work well
3. **Reduce seq_len** — quick experiment to validate the speedup is worth the context tradeoff
4. **Linear recurrence** — biggest potential gain but requires architecture research
