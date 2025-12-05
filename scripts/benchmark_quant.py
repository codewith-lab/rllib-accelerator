#!/usr/bin/env python3
"""
Benchmark vanilla vs dynamic-quantized PolicyBackbone inference speed.
"""

import os
import sys
import time

import warnings
warnings.filterwarnings("ignore", message=".*torch.ao.quantization.*")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.policy import PolicyBackbone


def quantize_model(model: nn.Module):
    model.eval()
    return torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )


def run_once(model, batch):
    with torch.no_grad():
        logits, value = model(batch)
    return logits, value


def benchmark(model, batch, steps=100, warmup=10):
    for _ in range(warmup):
        run_once(model, batch)
    start = time.time()
    for _ in range(steps):
        run_once(model, batch)
    end = time.time()
    return (end - start) / steps


def main():
    device = torch.device("cuda")
    in_dim = 64
    out_dim = 32
    hidden_dims = [1024] * 8
    batch = torch.randn(4096, in_dim, device=device)

    vanilla = PolicyBackbone(in_dim, out_dim, hidden_dims, use_residual=True).to(device)
    quantized = quantize_model(PolicyBackbone(in_dim, out_dim, hidden_dims, use_residual=True))

    vanilla_time = benchmark(vanilla, batch)
    quant_time = benchmark(quantized, batch)

    print("==== Vanilla vs Dynamic Quantized (CPU) ====")
    print(f"Vanilla avg step:   {vanilla_time:.6f}s")
    print(f"Quantized avg step: {quant_time:.6f}s")
    speedup = vanilla_time / quant_time if quant_time > 0 else float('inf')
    print(f"Speedup (vanilla / quantized): {speedup:.3f}x")


if __name__ == "__main__":
    main()
