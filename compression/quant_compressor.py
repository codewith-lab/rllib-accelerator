import time
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from torch import nn

from compression.base import BaseCompressor
from models.policy import PolicyBackbone


class QuantCompressor(BaseCompressor):
    """Dynamic quantization compressor targeting CPU inference."""

    def __init__(self, diff_threshold: float = 5e-4):
        self.diff_threshold = diff_threshold
        self._meta: Optional[Dict[str, Any]] = None

    def snapshot(self, train_model: Any) -> Dict[str, torch.Tensor]:
        bb = train_model.backbone
        if hasattr(bb, "_orig_mod"):
            bb_to_copy = bb._orig_mod
        else:
            bb_to_copy = bb
        state = {k: v.detach().cpu().clone() for k, v in bb_to_copy.state_dict().items()}
        hidden_dims = getattr(train_model, "hidden_dims", None) or [64, 64]
        self._meta = {
            "in_dim": getattr(train_model, "in_dim", None),
            "num_outputs": getattr(train_model, "num_outputs", None),
            "hidden_dims": list(hidden_dims),
            "use_residual": getattr(train_model, "use_residual", False),
        }
        return state

    def should_recompress(
        self, new_snapshot: Dict[str, torch.Tensor], last_snapshot: Optional[Dict[str, torch.Tensor]]
    ) -> bool:
        if last_snapshot is None:
            return True
        diffs: List[float] = []
        for k in new_snapshot:
            diffs.append((new_snapshot[k] - last_snapshot[k]).abs().mean().item())
        mean_diff = float(np.mean(diffs))
        return mean_diff > self.diff_threshold

    def compress(self, snapshot: Dict[str, torch.Tensor]) -> Tuple[Any, Dict[str, Any]]:
        if self._meta is None:
            raise RuntimeError("QuantCompressor snapshot meta missing.")
        in_dim = self._meta["in_dim"]
        num_outputs = self._meta["num_outputs"]
        hidden_dims = self._meta["hidden_dims"]
        use_residual = self._meta.get("use_residual", False)

        bb = PolicyBackbone(in_dim, num_outputs, hidden_dims, use_residual)
        bb.load_state_dict(snapshot)
        bb.eval()

        t0 = time.time()
        quantized = torch.quantization.quantize_dynamic(
            bb, {nn.Linear}, dtype=torch.qint8
        )
        latency = time.time() - t0
        return quantized, {
            "type": "dynamic-quant",
            "latency": latency,
            "hidden_dims": hidden_dims,
            "use_residual": use_residual,
        }
