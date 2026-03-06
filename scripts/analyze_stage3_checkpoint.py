#!/usr/bin/env python3
"""
Analyze Stage 3 checkpoint: track action embedding divergence, gate movement, temporal gammas.

Usage:
    python scripts/analyze_stage3_checkpoint.py \
        --checkpoint checkpoints/zimage_stage3/world_model_s2_epoch20.pt \
        [--baseline checkpoints/zimage_stage2_gamefactory/world_model_s2_final.pt]
"""

import argparse
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _get_projected_sim(ae_state: dict) -> torch.Tensor:
    """Run action encoder forward pass to get projected 3840-dim embeddings."""
    try:
        from models.zimage_world_model import ZImageActionEncoder
        enc = ZImageActionEncoder(num_actions=17, embedding_dim=512)
        enc.load_state_dict(ae_state)
        enc.eval()
        actions = torch.arange(17).unsqueeze(1)  # (17, 1)
        with torch.no_grad():
            emb = enc(actions)[:, 0].float()  # (17, 3840)
        import torch.nn.functional as F
        emb_norm = F.normalize(emb, dim=-1)
        return emb_norm @ emb_norm.T
    except Exception as e:
        return None


def analyze_checkpoint(ckpt_path: str, baseline_path: str = None):
    print(f"\n{'='*60}")
    print(f"Analyzing: {ckpt_path}")
    print(f"{'='*60}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    baseline = None
    if baseline_path and Path(baseline_path).exists():
        baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)

    epoch = ckpt.get("epoch", "?")
    step = ckpt.get("global_step", "?")
    print(f"Epoch: {epoch}, Global step: {step}")

    # --- 1. Action Encoder Embeddings ---
    ae = ckpt["action_encoder_state_dict"]
    emb = ae["action_embedding.weight"].float()  # (17, 512)
    norms = emb.norm(dim=-1)
    emb_norm = emb / norms.unsqueeze(-1).clamp(min=1e-8)
    sim = emb_norm @ emb_norm.T  # (17, 17) cosine similarities

    mask_off = ~torch.eye(17, dtype=bool)
    off_diag = sim[mask_off]

    # Expected action indices: 0=idle, 1=forward, 2=backward, 3=left, 4=right,
    #                          5=run, 6=jump, 7=attack, 8=interact
    action_names = {0: "idle", 1: "forward", 2: "backward", 3: "left", 4: "right",
                    5: "run", 6: "jump", 7: "attack", 8: "interact"}

    print("\n--- Action Embedding Analysis ---")
    print(f"  Norms: mean={norms.mean():.4f} ± {norms.std():.4f}")
    print(f"  Random baseline cos_sim: ±{1/emb.shape[1]**0.5:.4f}")
    print(f"  Mean off-diagonal cos_sim: {off_diag.mean():.4f} ± {off_diag.std():.4f}")
    print()
    # Key action pairs
    pairs = [(1, 2, "fwd/bwd"), (3, 4, "left/right"), (1, 5, "fwd/run"),
             (0, 6, "idle/jump"), (1, 3, "fwd/left")]
    for i, j, label in pairs:
        print(f"  cos_sim({label}): {sim[i,j].item():.4f}")

    # Most similar / most dissimilar pairs
    sim_flat = sim.clone()
    sim_flat.fill_diagonal_(-2)
    max_idx = sim_flat.argmax()
    i_max, j_max = max_idx // 17, max_idx % 17
    sim_flat.fill_diagonal_(2)
    min_idx = sim_flat.argmin()
    i_min, j_min = min_idx // 17, min_idx % 17
    print(f"\n  Most similar pair: actions {i_max.item()} & {j_max.item()} "
          f"({action_names.get(i_max.item(),'?')}/{action_names.get(j_max.item(),'?')})"
          f" = {sim[i_max, j_max].item():.4f}")
    print(f"  Most dissimilar pair: actions {i_min.item()} & {j_min.item()} "
          f"({action_names.get(i_min.item(),'?')}/{action_names.get(j_min.item(),'?')})"
          f" = {sim[i_min, j_min].item():.4f}")

    # Compare to baseline
    if baseline:
        ae_b = baseline.get("action_encoder_state_dict", {})
        if "action_embedding.weight" in ae_b:
            emb_b = ae_b["action_embedding.weight"].float()
            delta = (emb - emb_b).norm(dim=-1)
            print(f"\n  Embedding drift from baseline: mean={delta.mean():.4f}, max={delta.max():.4f}")

    # --- 2. Action Injection Gates ---
    ai = ckpt["action_injections_state_dict"]
    gate_keys = [k for k in ai.keys() if k.endswith(".gate")]
    print("\n--- ActionInjection Gates ---")
    for k in gate_keys:
        v = ai[k].float()
        print(f"  {k}: raw={v.item():.6f}, sigmoid={torch.sigmoid(v).item():.6f}")

    # Compare gates to baseline
    if baseline:
        ai_b = baseline.get("action_injections_state_dict", {})
        for k in gate_keys:
            if k in ai_b:
                delta = abs(ai[k].item() - ai_b[k].item())
                print(f"    Δ from baseline: {delta:.8f}")

    # to_out weights
    print("\n--- ActionInjection to_out weight stats ---")
    for layer in ["7", "15", "22"]:
        key = f"{layer}.to_out.0.weight"
        if key in ai:
            w = ai[key].float()
            print(f"  Layer {layer}: max={w.abs().max():.6f}, std={w.std():.6f}, "
                  f"mean_abs={w.abs().mean():.6f}")

    # --- 3. Temporal Gammas ---
    td = ckpt["temporal_state_dict"]
    gamma_keys = [k for k in td.keys() if "gamma" in k]
    gammas = [td[k].float().item() for k in gamma_keys]
    print(f"\n--- Temporal Gammas ({len(gamma_keys)} total) ---")
    print(f"  mean={np.mean(gammas):.8f} ± {np.std(gammas):.8f}")
    print(f"  min={min(gammas):.8f}, max={max(gammas):.8f}")

    if baseline:
        td_b = baseline.get("temporal_state_dict", {})
        gamma_b = [td_b[k].float().item() for k in gamma_keys if k in td_b]
        if gamma_b:
            deltas = [abs(td[k].float().item() - td_b[k].float().item())
                      for k in gamma_keys if k in td_b]
            changed = sum(d > 1e-9 for d in deltas)
            print(f"  Changed from baseline: {changed}/{len(gamma_keys)}")
            print(f"  Max delta: {max(deltas):.8f}")

    # --- 4. Projected embedding similarity (3840-dim, what contrastive loss trains) ---
    proj_sim = _get_projected_sim(ae)
    if proj_sim is not None:
        mask_off_p = ~torch.eye(17, dtype=bool)
        off_p = proj_sim[mask_off_p]
        print(f"\n--- Projected (3840-dim) Embeddings [what the model actually uses] ---")
        print(f"  Random baseline: ±{1/3840**0.5:.4f}")
        print(f"  Mean off-diagonal: {off_p.mean():.4f} ± {off_p.std():.4f}")
        for i, j, label in [(1, 2, "fwd/bwd"), (3, 4, "left/right"), (1, 5, "fwd/run"), (0, 6, "idle/jump")]:
            print(f"  cos_sim({label}): {proj_sim[i,j].item():.4f}")
        s = proj_sim.clone(); s.fill_diagonal_(2.0)
        mi = s.argmin(); i_min, j_min = mi//17, mi%17
        s.fill_diagonal_(-2.0)
        ma = s.argmax(); i_max, j_max = ma//17, ma%17
        print(f"  Most dissimilar: {i_min.item()}&{j_min.item()} = {proj_sim[i_min,j_min].item():.4f}")
        print(f"  Most similar:    {i_max.item()}&{j_max.item()} = {proj_sim[i_max,j_max].item():.4f}")
        fwd_bwd_proj = proj_sim[1, 2].item()
    else:
        fwd_bwd_proj = None

    # --- 4b. Injection residual similarity ---
    inj_fwd_bwd = None
    try:
        from models.zimage_world_model import ZImageActionInjectionLayer, ZImageActionEncoder
        import torch.nn.functional as F_fn

        enc = ZImageActionEncoder(num_actions=17, embedding_dim=512)
        enc.load_state_dict(ae)
        enc.eval()

        # Use first injection layer (key "7")
        inj_state = {k[2:]: v for k, v in ai.items() if k.startswith("7.")}
        inj = ZImageActionInjectionLayer(hidden_dim=3840, num_heads=30)
        try:
            inj.load_state_dict(inj_state)
        except Exception:
            inj = None

        if inj is not None:
            inj.eval()
            actions_pair = torch.tensor([[1, 1], [2, 2]])  # fwd, bwd — 2 frames each
            with torch.no_grad():
                emb = enc(actions_pair)  # (2, 2, 3840)
                # Use a fixed mock x (zeros → isolates action-driven component)
                x_mock = torch.zeros(2, 32, 3840)  # (B, seq=32, D)
                cond = emb[:, 0:1, :]  # (B, 1, D) first frame
                _, residual = inj(x_mock, cond, return_residual=True)  # (B, 32, D)
            r = residual.mean(dim=1).float()  # (B, D)
            r_norm = F_fn.normalize(r, dim=-1)
            sim_inj = (r_norm[0] * r_norm[1]).sum().item()
            inj_fwd_bwd = sim_inj
            print(f"\n--- Injection Residual Similarity (layer 7) ---")
            print(f"  cos_sim(fwd_inj, bwd_inj): {sim_inj:.4f}")
            print(f"  Ideal: -1.0 (fully opposite), Bad: +1.0 (identical action signals)")
            print(f"  to_out max: {ai['7.to_out.0.weight'].float().abs().max():.6f}")
    except Exception as e:
        pass

    # --- 5. Summary verdict ---
    fwd_bwd_sim = sim[1, 2].item()
    gate_mean = np.mean([torch.sigmoid(ai[k].float()).item() for k in gate_keys])
    gamma_changed = 0
    if baseline:
        td_b = baseline.get("temporal_state_dict", {})
        gamma_changed = sum(abs(td[k].float().item() - td_b[k].float().item()) > 1e-9
                            for k in gamma_keys if k in td_b)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"  fwd/bwd cos_sim: {fwd_bwd_sim:.4f}  (target: diverge from 0, ideally <-0.3 or >0.3)")
    print(f"  mean gate sigmoid: {gate_mean:.4f}  (target: move significantly from 0.5)")
    print(f"  gammas changed: {gamma_changed}/{len(gamma_keys)}  (0 = bfloat16 floor issue)")

    # Use projected similarity if available (more meaningful)
    sim_for_verdict = fwd_bwd_proj if fwd_bwd_proj is not None else fwd_bwd_sim
    if abs(sim_for_verdict) > 0.3:
        verdict = "WORKING — strong action discrimination in projected space"
    elif abs(sim_for_verdict) > 0.1:
        verdict = "IMPROVING — moderate action discrimination"
    else:
        verdict = "NOT YET — embeddings still at noise level"
    print(f"  projected fwd/bwd: {fwd_bwd_proj:.4f}" if fwd_bwd_proj is not None else "  projected: N/A")
    print(f"  injection residual fwd/bwd: {inj_fwd_bwd:.4f}" if inj_fwd_bwd is not None else "  injection: N/A")
    print(f"  verdict: {verdict}")
    print(f"{'='*60}\n")

    return {
        "fwd_bwd_sim": fwd_bwd_sim,
        "gate_mean": gate_mean,
        "gamma_changed": gamma_changed,
        "off_diag_sim_mean": off_diag.mean().item(),
        "off_diag_sim_std": off_diag.std().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline", default="checkpoints/zimage_stage2_gamefactory/world_model_s2_final.pt")
    args = parser.parse_args()
    analyze_checkpoint(args.checkpoint, args.baseline)


if __name__ == "__main__":
    main()
