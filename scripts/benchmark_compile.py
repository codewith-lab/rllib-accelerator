#!/usr/bin/env python3
"""
Fixed inference benchmark comparing vanilla PyTorch vs torch.compile for a large MLP.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim=64, out_dim=32, hidden_dim=1024, num_layers=8, use_residual=True):
        super().__init__()
        layers = []
        prev = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev, hidden_dim))
            prev = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.policy_head = nn.Linear(prev, out_dim)
        self.value_head = nn.Linear(prev, 1)
        self.use_residual = use_residual

    def forward(self, x):
        for layer in self.layers:
            residual = x if self.use_residual and x.shape[-1] == layer.out_features else None
            x = F.relu(layer(x))
            if residual is not None:
                x = x + residual
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


def run_step(model, batch, optimizer):
    with torch.no_grad():
        logits, value = model(batch)
    return (logits, value)


def benchmark(model, batch, steps=100, warmup=20):
    for _ in range(warmup):
        run_step(model, batch, None)
    if torch.cuda.is_available() and batch.is_cuda:
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(steps):
        run_step(model, batch, None)
    if torch.cuda.is_available() and batch.is_cuda:
        torch.cuda.synchronize()
    end = time.time()
    return (end - start) / steps


def main():
    device = torch.device("cpu")
    batch = torch.randn(4096, 64, device=device)

    vanilla = MLP().to(device)
    compiled = torch.compile(MLP().to(device), backend="inductor")

    vanilla_time = benchmark(vanilla, batch)
    compiled_time = benchmark(compiled, batch)

    print("==== PyTorch vs torch.compile Benchmark ====")
    print(f"Vanilla average step time:  {vanilla_time:.4f}s")
    print(f"Compiled average step time: {compiled_time:.4f}s")
    speedup = vanilla_time / compiled_time if compiled_time > 0 else float("inf")
    print(f"Speedup (vanilla / compiled): {speedup:.3f}x")


if __name__ == "__main__":
    main()
