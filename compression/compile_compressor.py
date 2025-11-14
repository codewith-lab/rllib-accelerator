# path: compression/compile_compressor.py

import time
import numpy as np
from typing import Any, Dict, Tuple, Optional, List
import torch
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

from compression.base import BaseCompressor
from models.policy import PolicyBackbone  # ⚠️你需要把这个改成你的真实路径


class CompileCompressor(BaseCompressor):
    """
    用于 torch.compile 的压缩器。

    主要功能：
    - 从 train_model.backbone 拍 snapshot（state_dict clone）
    - 基于权重 diff 判断是否需要重新 compile
    - 调用 torch.compile 生成 compiled_backbone
    """

    def __init__(self,
                 backend: str = "inductor",
                 diff_threshold: float = 1e-4):
        """
        参数:
            backend: torch.compile backend（一般用 inductor）
            diff_threshold: 若新旧 snapshot 平均差异大于此阈值则重新编译
        """
        self.backend = backend
        self.diff_threshold = diff_threshold
        self._raw_model: Optional[PolicyBackbone] = None
        self._compiled_model: Optional[Any] = None
        self._meta: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------
    # 1. snapshot
    # ------------------------------------------------------------
    def snapshot(self, train_model: Any) -> Dict[str, torch.Tensor]:
        """复制 backbone 的 state_dict（无梯度，cpu clone）。"""
        bb = train_model.backbone
        if hasattr(bb, "_orig_mod"):
            bb_to_copy = bb._orig_mod
        else:
            bb_to_copy = bb
        state = {
            k: v.detach().cpu().clone()
            for k, v in bb_to_copy.state_dict().items()
        }
        hidden_dims = getattr(train_model, "hidden_dims", None)
        if hidden_dims is None:
            hidden_dims = [64, 64]
        self._meta = {
            "in_dim": getattr(train_model, "in_dim", None),
            "num_outputs": getattr(train_model, "num_outputs", None),
            "hidden_dims": list(hidden_dims),
            "use_residual": getattr(train_model, "use_residual", False),
        }
        return state

    # ------------------------------------------------------------
    # 2. diff 检测
    # ------------------------------------------------------------
    def should_recompress(self,
                          new_snapshot: Dict[str, torch.Tensor],
                          last_snapshot: Dict[str, torch.Tensor]) -> bool:
        """基于参数差分判断是否需要重新编译。"""

        if last_snapshot is None:
            return True  # 第一次必须压缩

        diffs = []
        for k in new_snapshot:
            diff_value = (new_snapshot[k] - last_snapshot[k]).abs().mean().item()
            diffs.append(diff_value)

        mean_diff = float(np.mean(diffs))

        return mean_diff > self.diff_threshold

    # ------------------------------------------------------------
    # 3. compress（torch.compile）
    # ------------------------------------------------------------
    def compress(self, snapshot: Dict[str, torch.Tensor]) -> Tuple[Any, Dict[str, Any]]:
        """执行 torch.compile，返回新的 compiled_backbone。"""

        # 自动推断 backbone 结构（你自己的 backbone，请确保一致）
        if self._meta is None:
            raise RuntimeError("CompileCompressor snapshot meta is missing.")
        in_dim = self._meta["in_dim"]
        num_outputs = self._meta["num_outputs"]
        hidden_dims: List[int] = self._meta["hidden_dims"]
        use_residual: bool = self._meta.get("use_residual", False)

        # 如果已经有编译好的模型，就复用并仅更新权重
        reused = False
        if self._compiled_model is not None and self._raw_model is not None:
            load_start = time.time()
            self._raw_model.load_state_dict(snapshot)
            latency = time.time() - load_start
            compiled_bb = self._compiled_model
            reused = True
        else:
            bb = PolicyBackbone(in_dim, num_outputs, hidden_dims, use_residual)
            bb.load_state_dict(snapshot)

            t0 = time.time()
            compiled_bb = torch.compile(bb, backend=self.backend)
            t1 = time.time()

            latency = t1 - t0
            self._raw_model = bb
            self._compiled_model = compiled_bb

        return compiled_bb, {
            "type": "torch.compile",
            "backend": self.backend,
            "latency": latency,
            "in_dim": in_dim,
            "num_outputs": num_outputs,
            "hidden_dims": hidden_dims,
            "use_residual": use_residual,
            "reused": reused,
        }
