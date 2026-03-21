#!/usr/bin/env python3
"""Analyze theoretical SPS limits by profiling each phase."""
import time
import torch
import numpy as np
from slither_gym.dreamer.agent import DreamerV3Agent, _mamba_wm_forward
from slither_gym.dreamer.replay_buffer import ReplayBuffer
from slither_gym.dreamer.networks import RSSMState

def profile_phases(agent, buf, B, T=50, N=30):
    """Profile each phase of train_step independently."""
    batch = buf.sample(B)
    obs, actions, rewards, conts = agent.transfer_batch(batch)
    torch.cuda.synchronize()

    # Phase 1: Data transfer
    t0 = time.perf_counter()
    for _ in range(N):
        agent.transfer_batch(batch)
    torch.cuda.synchronize()
    xfer_ms = (time.perf_counter() - t0) / N * 1000

    # Phase 2: WM forward only (no backward)
    t0 = time.perf_counter()
    for _ in range(N):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, rl, rwl, cl, kl, fd, fs, fh = agent._compiled_wm_forward(
                agent.world_model.encoder, agent.world_model.rssm,
                agent.world_model.decoder, agent.world_model.reward_head,
                agent.world_model.continue_head,
                obs, actions, rewards, conts, B, T,
                agent.reward_bins, agent.free_nats, agent.kl_scale,
            )
    torch.cuda.synchronize()
    wm_fwd_ms = (time.perf_counter() - t0) / N * 1000

    # Phase 3: WM backward only
    t0 = time.perf_counter()
    for _ in range(N):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss, rl, rwl, cl, kl, fd, fs, fh = agent._compiled_wm_forward(
                agent.world_model.encoder, agent.world_model.rssm,
                agent.world_model.decoder, agent.world_model.reward_head,
                agent.world_model.continue_head,
                obs, actions, rewards, conts, B, T,
                agent.reward_bins, agent.free_nats, agent.kl_scale,
            )
        agent.model_opt.zero_grad(set_to_none=True)
        loss.backward()
    torch.cuda.synchronize()
    wm_fwdbwd_ms = (time.perf_counter() - t0) / N * 1000
    wm_bwd_ms = wm_fwdbwd_ms - wm_fwd_ms

    # Phase 4: WM optimizer step
    t0 = time.perf_counter()
    for _ in range(N):
        agent.model_opt.step()
    torch.cuda.synchronize()
    wm_opt_ms = (time.perf_counter() - t0) / N * 1000

    # Phase 5: WM grad clip
    t0 = time.perf_counter()
    for _ in range(N):
        torch.nn.utils.clip_grad_norm_(agent.world_model.parameters(), 100.0)
    torch.cuda.synchronize()
    wm_clip_ms = (time.perf_counter() - t0) / N * 1000

    # Phase 6: Full AC
    # Need to set up state cache first
    packed_deter = torch.cat([fd, fh.reshape(B, -1)], dim=-1)
    agent._train_state_cache = RSSMState(deter=packed_deter.detach(), stoch=fs.detach())

    t0 = time.perf_counter()
    for _ in range(N):
        agent._train_actor_critic(obs, actions, B, T)
    torch.cuda.synchronize()
    ac_ms = (time.perf_counter() - t0) / N * 1000

    # Phase 7: EMA
    t0 = time.perf_counter()
    for _ in range(N):
        with torch.no_grad():
            for sp, tp in zip(agent.slow_critic.parameters(), agent.critic.parameters()):
                sp.data.lerp_(tp.data, 0.02)
    torch.cuda.synchronize()
    ema_ms = (time.perf_counter() - t0) / N * 1000

    # Full train_step for comparison
    t0 = time.perf_counter()
    for _ in range(N):
        agent.train_step(batch)
    torch.cuda.synchronize()
    full_ms = (time.perf_counter() - t0) / N * 1000
    sps = B * T / (512 * full_ms / 1000)

    return {
        "B": B,
        "full_ms": full_ms,
        "sps": sps,
        "xfer_ms": xfer_ms,
        "wm_fwd_ms": wm_fwd_ms,
        "wm_bwd_ms": wm_bwd_ms,
        "wm_clip_ms": wm_clip_ms,
        "wm_opt_ms": wm_opt_ms,
        "ac_ms": ac_ms,
        "ema_ms": ema_ms,
    }


def main():
    device = "cuda"
    T = 50
    action_dim = 6

    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    print()

    agent = DreamerV3Agent(
        action_dim=action_dim, device=device, use_amp=True,
        compile_models=False, rssm_type="mamba",
    )

    # Count parameters
    wm_params = sum(p.numel() for p in agent.world_model.parameters())
    actor_params = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critic.parameters())
    total_params = wm_params + actor_params + critic_params * 2
    print(f"Parameters: WM={wm_params/1e6:.2f}M  Actor={actor_params/1e6:.2f}M  "
          f"Critic={critic_params/1e6:.2f}M  Total={total_params/1e6:.2f}M")
    print()

    buf = ReplayBuffer(capacity=300_000, seq_len=T, pin_memory=True)
    for _ in range(80):
        ep_len = 200
        ep = {
            "obs": np.random.randint(0, 255, (ep_len, 3, 64, 64), dtype=np.uint8),
            "action": np.eye(action_dim, dtype=np.float32)[np.random.randint(0, action_dim, ep_len)],
            "reward": np.random.randn(ep_len).astype(np.float32),
            "cont": np.ones(ep_len, dtype=np.float32),
        }
        buf.add_episode(ep)

    # Warmup
    batch = buf.sample(128)
    for _ in range(5):
        agent.train_step(batch)
    torch.cuda.synchronize()

    print("=" * 80)
    print(f"{'Phase':<25s} | {'B=128':>10s} | {'B=256':>10s} | {'B=384':>10s} | {'B=512':>10s}")
    print("-" * 80)

    results = {}
    for B in [128, 256, 384, 512]:
        try:
            # Warmup this batch size
            batch = buf.sample(B)
            for _ in range(3):
                agent.train_step(batch)
            torch.cuda.synchronize()

            r = profile_phases(agent, buf, B)
            results[B] = r
        except torch.cuda.OutOfMemoryError:
            results[B] = None
            torch.cuda.empty_cache()

    phases = ["xfer_ms", "wm_fwd_ms", "wm_bwd_ms", "wm_clip_ms", "wm_opt_ms", "ac_ms", "ema_ms", "full_ms"]
    labels = {
        "xfer_ms": "Data transfer",
        "wm_fwd_ms": "WM forward",
        "wm_bwd_ms": "WM backward",
        "wm_clip_ms": "WM grad clip",
        "wm_opt_ms": "WM optimizer",
        "ac_ms": "Actor-critic",
        "ema_ms": "Slow critic EMA",
        "full_ms": "TOTAL (train_step)",
    }
    for phase in phases:
        label = labels[phase]
        vals = []
        for B in [128, 256, 384, 512]:
            r = results.get(B)
            if r:
                vals.append(f"{r[phase]:>7.1f} ms")
            else:
                vals.append(f"{'OOM':>10s}")
        sep = " | ".join(vals)
        bold = "**" if phase == "full_ms" else "  "
        print(f"{bold}{label:<23s}{bold} | {sep}")

    print("-" * 80)
    sps_line = []
    for B in [128, 256, 384, 512]:
        r = results.get(B)
        if r:
            sps_line.append(f"{r['sps']:>7.0f} SPS")
        else:
            sps_line.append(f"{'OOM':>10s}")
    print(f"  {'Theoretical SPS':<23s} | {' | '.join(sps_line)}")

    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Use B=384 as reference
    r = results.get(384) or results.get(256) or results[128]
    B = r["B"]

    print(f"\nAt B={B}, train_step = {r['full_ms']:.1f} ms:")
    print(f"  WM backward is {r['wm_bwd_ms']/r['full_ms']*100:.0f}% of total ({r['wm_bwd_ms']:.1f} ms)")
    print(f"  WM forward is {r['wm_fwd_ms']/r['full_ms']*100:.0f}% of total ({r['wm_fwd_ms']:.1f} ms)")
    print(f"  Actor-critic is {r['ac_ms']/r['full_ms']*100:.0f}% of total ({r['ac_ms']:.1f} ms)")

    overhead = r['full_ms'] - r['wm_fwd_ms'] - r['wm_bwd_ms'] - r['ac_ms'] - r['xfer_ms']
    print(f"  Overhead (clip+opt+ema+misc) = {overhead:.1f} ms")

    # Theoretical limits
    print(f"\n--- Theoretical SPS limits ---")
    print(f"\nFormula: SPS = B × T / (train_ratio × t_train)")
    print(f"  B={B}, T={T}, train_ratio=512")

    # If we could eliminate backward overhead
    no_bwd_overhead = r['wm_fwd_ms'] + r['wm_bwd_ms'] * 0.5 + r['ac_ms'] + r['xfer_ms'] + 5  # 5ms overhead
    sps_half_bwd = B * T / (512 * no_bwd_overhead / 1000)
    print(f"\n  If backward were 2× faster:     {no_bwd_overhead:.0f} ms → {sps_half_bwd:.0f} SPS")

    # If we could use CUDA graphs for everything
    cuda_graph = r['full_ms'] * 0.7  # ~30% overhead from kernel launch
    sps_cg = B * T / (512 * cuda_graph / 1000)
    print(f"  If CUDA graphs worked (−30%):    {cuda_graph:.0f} ms → {sps_cg:.0f} SPS")

    # If we reduced T from 50 to 32
    t32_ms = r['full_ms'] * 32 / 50
    sps_t32 = B * 32 / (512 * t32_ms / 1000)
    print(f"  If T=32 instead of T=50:         {t32_ms:.0f} ms → {sps_t32:.0f} SPS  (same, T cancels)")

    # If we reduced train_ratio from 512 to 256
    sps_tr256 = B * T / (256 * r['full_ms'] / 1000)
    print(f"  If train_ratio=256 (half):       {r['full_ms']:.0f} ms → {sps_tr256:.0f} SPS  (2× but less learning)")

    # Compute bound estimate
    # Rough FLOP count: ~30-50 GFLOPs per train_step at B=384
    # RTX 5090 bf16: ~400+ TFLOPS
    compute_limit_ms = 50e9 / 400e12 * 1000  # 50 GFLOPs / 400 TFLOPS
    sps_compute = B * T / (512 * compute_limit_ms / 1000)
    print(f"\n  Pure compute bound (est):        {compute_limit_ms:.2f} ms → {sps_compute:.0f} SPS")
    print(f"  (assumes ~50 GFLOPs/step, ~400 TFLOPS peak)")

    # Memory bandwidth bound estimate
    # Total data touched per step: ~2 × params (read+write) × 3 (fwd+bwd+opt)
    # + activations
    total_bytes = total_params * 4 * 3 * 2 + B * T * 512 * 4 * 10  # rough
    mem_bw = 1.8e12  # ~1.8 TB/s for GDDR7
    mem_limit_ms = total_bytes / mem_bw * 1000
    print(f"  Pure memory BW bound (est):      {mem_limit_ms:.2f} ms → {B * T / (512 * mem_limit_ms / 1000):.0f} SPS")
    print(f"  (assumes ~{total_bytes/1e9:.1f} GB touched, ~1.8 TB/s BW)")

    # Current efficiency
    efficiency = compute_limit_ms / r['full_ms'] * 100
    print(f"\n  Current compute efficiency:      {efficiency:.1f}%")
    print(f"  Bottleneck: {'memory bandwidth' if efficiency < 20 else 'kernel launch/Python overhead' if efficiency < 50 else 'compute'}")

if __name__ == "__main__":
    main()
