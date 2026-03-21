#!/usr/bin/env python3
"""Clean benchmark: just full train_step at various batch sizes."""
import time, torch, numpy as np
from slither_gym.dreamer.agent import DreamerV3Agent
from slither_gym.dreamer.replay_buffer import ReplayBuffer

def main():
    T, action_dim, N = 50, 6, 30
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu} ({mem:.0f} GB)\n")

    agent = DreamerV3Agent(action_dim=action_dim, device="cuda", use_amp=True,
                           compile_models=False, rssm_type="mamba")
    wm_p = sum(p.numel() for p in agent.world_model.parameters())
    ac_p = sum(p.numel() for p in agent.actor.parameters())
    cr_p = sum(p.numel() for p in agent.critic.parameters())
    print(f"Params: WM={wm_p/1e6:.1f}M  Actor={ac_p/1e6:.1f}M  Critic={cr_p/1e6:.1f}M\n")

    buf = ReplayBuffer(capacity=500_000, seq_len=T, pin_memory=True)
    for _ in range(100):
        ep = {"obs": np.random.randint(0, 255, (200, 3, 64, 64), dtype=np.uint8),
              "action": np.eye(action_dim, dtype=np.float32)[np.random.randint(0, action_dim, 200)],
              "reward": np.random.randn(200).astype(np.float32),
              "cont": np.ones(200, dtype=np.float32)}
        buf.add_episode(ep)

    # Warmup
    for _ in range(5): agent.train_step(buf.sample(128))
    torch.cuda.synchronize()

    print(f"{'B':>5s} | {'ms/step':>10s} | {'SPS':>8s} | {'VRAM':>8s} | {'bwd/fwd':>8s}")
    print("-" * 55)

    for B in [64, 128, 192, 256, 384, 512, 640]:
        torch.cuda.reset_peak_memory_stats()
        try:
            batch = buf.sample(B)
            for _ in range(3): agent.train_step(batch)
            torch.cuda.synchronize()

            # Full train_step
            t0 = time.perf_counter()
            for _ in range(N): agent.train_step(batch)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / N * 1000
            sps = B * T / (512 * ms / 1000)
            vram = torch.cuda.max_memory_allocated() / 1e9

            # Forward-only timing
            obs, actions, rewards, conts = agent.transfer_batch(batch)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(N):
                    loss, *_ = agent._compiled_wm_forward(
                        agent.world_model.encoder, agent.world_model.rssm,
                        agent.world_model.decoder, agent.world_model.reward_head,
                        agent.world_model.continue_head,
                        obs, actions, rewards, conts, B, T,
                        agent.reward_bins, agent.free_nats, agent.kl_scale)
                torch.cuda.synchronize()
                fwd_ms = (time.perf_counter() - t0) / N * 1000

            ratio = (ms - fwd_ms) / fwd_ms  # rough bwd/fwd ratio
            print(f"{B:>5d} | {ms:>7.1f} ms | {sps:>5.0f}   | {vram:>5.1f} GB | {ratio:>5.1f}x")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                print(f"{B:>5d} | {'OOM':>10s} |")
                torch.cuda.empty_cache()
            else:
                raise

    print()
    print("Key: SPS = B×T / (512 × t_train)")
    print("Backward pass is the dominant bottleneck (~50% of total)")
    print()
    print("--- Theoretical limits (B=256 reference) ---")
    # Compute at B=256
    B_ref = 256
    batch = buf.sample(B_ref)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N): agent.train_step(batch)
    torch.cuda.synchronize()
    ms_ref = (time.perf_counter() - t0) / N * 1000
    sps_ref = B_ref * T / (512 * ms_ref / 1000)

    print(f"Current:                    {ms_ref:.0f} ms → {sps_ref:.0f} SPS")
    print(f"2x faster backward:         {ms_ref*0.75:.0f} ms → {sps_ref/0.75:.0f} SPS")
    print(f"CUDA graphs (−30% overhead): {ms_ref*0.70:.0f} ms → {sps_ref/0.70:.0f} SPS")
    print(f"train_ratio=256:            {ms_ref:.0f} ms → {sps_ref*2:.0f} SPS (less learning per step)")

if __name__ == "__main__":
    main()
