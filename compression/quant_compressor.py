import time
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

from compression.base import BaseCompressor
from models.policy import PolicyBackbone


class QuantCompressor(BaseCompressor):
    """
    Quantization compressor targeting CPU inference.
    
    Modes:
        - dynamic (default): torch.quantization.quantize_dynamic on Linear layers
        - weight_only: int8 weight-only quant (activations stay float)
    """

    def __init__(self,
                 diff_threshold: float = 5e-4,
                 mode: str = "dynamic",
                 trt_calib_batches: int = 4,
                 trt_calib_batch_size: int = 64,
                 calibration_data: Optional[Any] = None,
                 device: str = "cpu"):
        self.diff_threshold = diff_threshold
        self.mode = mode
        self.trt_calib_batches = trt_calib_batches
        self.trt_calib_batch_size = trt_calib_batch_size
        # For TensorRT int8: user-provided calibration data (DataLoader, Dataset, or list/tuple of tensors/ndarrays)
        self.calibration_data = calibration_data
        self.device = self._resolve_device(device)
        self._meta: Optional[Dict[str, Any]] = None
        self._weight_only_backend: Optional[str] = None

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
        if self.mode == "dynamic":
            # Dynamic quant only runs on CPU; ensure model is on CPU even if a GPU exists.
            bb = bb.cpu()
            quantized = torch.quantization.quantize_dynamic(
                bb, {nn.Linear}, dtype=torch.qint8
            )
            qtype = "dynamic-quant"
        elif self.mode == "weight_only":
            quantized = self._apply_weight_only_quant(bb)
            qtype = "weight-only-int8"
        elif self.mode == "tensorrt_int8":
            quantized = self._compile_tensorrt_int8(bb, in_dim)
            qtype = "tensorrt-int8"
        else:
            raise ValueError(f"Unknown quantization mode: {self.mode}")
        latency = time.time() - t0
        meta = {
            "type": qtype,
            "latency": latency,
            "hidden_dims": hidden_dims,
            "use_residual": use_residual,
        }
        if self.mode == "weight_only":
            meta["backend"] = self._weight_only_backend or "custom_cpu"
        return quantized, meta

    @staticmethod
    def _quantize_linear_weights(linear: nn.Linear):
        # Per-output-channel symmetric quantization
        weight = linear.weight.detach().cpu()
        bias = linear.bias.detach().cpu() if linear.bias is not None else None
        # Avoid division by zero
        max_vals = weight.abs().amax(dim=1)
        max_vals = torch.where(max_vals == 0, torch.full_like(max_vals, 1e-8), max_vals)
        scales = max_vals / 127.0
        zero_points = torch.zeros_like(scales, dtype=torch.int64)
        qweight = torch.quantize_per_channel(weight, scales, zero_points, axis=0, dtype=torch.qint8)
        return qweight, bias

    class WeightOnlyLinear(nn.Module):
        def __init__(self, qweight: torch.Tensor, bias: Optional[torch.Tensor]):
            super().__init__()
            self.register_buffer("qweight", qweight)
            if bias is not None:
                self.bias = nn.Parameter(bias)
            else:
                self.bias = None

        def forward(self, x):
            w = self.qweight.dequantize()
            return F.linear(x, w, self.bias)

    def _apply_weight_only_quant(self, model: PolicyBackbone) -> PolicyBackbone:
        if self.device is not None and self.device.type == "cuda":
            quantized = self._apply_weight_only_quant_cuda(model)
            if quantized is not None:
                return quantized
        return self._apply_weight_only_quant_cpu(model)

    def _apply_weight_only_quant_cpu(self, model: PolicyBackbone) -> PolicyBackbone:
        self._weight_only_backend = "custom_cpu"
        # Clone structure, then replace Linear layers with weight-only versions
        quantized_model = PolicyBackbone(
            model.hidden_layers[0].in_features,
            model.policy_head.out_features,
            [layer.out_features for layer in model.hidden_layers],
            model.use_residual,
        )
        quantized_model.eval()

        # Quantize hidden layers
        for idx, layer in enumerate(model.hidden_layers):
            if isinstance(layer, nn.Linear):
                qweight, bias = self._quantize_linear_weights(layer)
                quantized_model.hidden_layers[idx] = self.WeightOnlyLinear(qweight, bias)

        # Quantize policy head
        if isinstance(model.policy_head, nn.Linear):
            qweight, bias = self._quantize_linear_weights(model.policy_head)
            quantized_model.policy_head = self.WeightOnlyLinear(qweight, bias)

        # Quantize value head
        if isinstance(model.value_head, nn.Linear):
            qweight, bias = self._quantize_linear_weights(model.value_head)
            quantized_model.value_head = self.WeightOnlyLinear(qweight, bias)

        return quantized_model

    def _apply_weight_only_quant_cuda(self, model: PolicyBackbone) -> Optional[PolicyBackbone]:
        if not torch.cuda.is_available():
            return None
        device = self.device or torch.device("cuda")
        model = model.to(device).eval()
        torchao_err = None
        try:
            from torchao.quantization import quantize_, int8_weight_only
            quantize_(model, int8_weight_only())
            self._weight_only_backend = "torchao"
            return model
        except Exception as exc:
            torchao_err = exc

        try:
            import bitsandbytes as bnb
        except Exception:
            print(
                "[QuantCompressor] GPU weight-only requested, but torchao/bitsandbytes is unavailable; "
                "falling back to CPU weight-only."
            )
            if torchao_err is not None:
                print(f"[QuantCompressor] torchao error: {torchao_err}")
            return None

        try:
            quantized = self._apply_bnb_weight_only(model, bnb, device)
            self._weight_only_backend = "bitsandbytes"
            return quantized
        except Exception as exc:
            print(f"[QuantCompressor] bitsandbytes weight-only failed: {exc}. Falling back to CPU weight-only.")
            return None

    @staticmethod
    def _apply_bnb_weight_only(model: PolicyBackbone, bnb, device: torch.device) -> PolicyBackbone:
        def to_bnb_linear(layer: nn.Linear) -> nn.Module:
            bnb_linear = bnb.nn.Linear8bitLt(
                layer.in_features,
                layer.out_features,
                bias=layer.bias is not None,
            )
            bnb_linear.load_state_dict(layer.state_dict())
            return bnb_linear

        for idx, layer in enumerate(model.hidden_layers):
            if isinstance(layer, nn.Linear):
                model.hidden_layers[idx] = to_bnb_linear(layer)

        if isinstance(model.policy_head, nn.Linear):
            model.policy_head = to_bnb_linear(model.policy_head)

        if isinstance(model.value_head, nn.Linear):
            model.value_head = to_bnb_linear(model.value_head)

        return model.to(device).eval()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        try:
            resolved = torch.device(device)
            if resolved.type.startswith("cuda") and not torch.cuda.is_available():
                print(f"[QuantCompressor] Device {device} unavailable, fallback to CPU.")
                return torch.device("cpu")
            return resolved
        except (RuntimeError, TypeError):
            print(f"[QuantCompressor] Invalid device {device}, fallback to CPU.")
            return torch.device("cpu")

    def _normalize_sample(self, sample: Any, in_dim: int) -> torch.Tensor:
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample)
        sample = sample.float()
        if sample.ndim == 1:
            sample = sample.unsqueeze(0)
        if sample.shape[-1] != in_dim:
            raise ValueError(f"Calibration sample last dim {sample.shape[-1]} != expected in_dim {in_dim}")
        return sample

    def _get_calibration_loader(self, in_dim: int) -> Optional[DataLoader]:
        data = self.calibration_data
        if data is None:
            return None
        if isinstance(data, DataLoader):
            return data
        if isinstance(data, Dataset):
            return DataLoader(data, batch_size=self.trt_calib_batch_size)
        if isinstance(data, (list, tuple)):
            tensors = [self._normalize_sample(x, in_dim).squeeze(0) for x in data]
            stacked = torch.stack(tensors)
            ds = TensorDataset(stacked)
            return DataLoader(ds, batch_size=self.trt_calib_batch_size)
        return None

    def set_calibration_data(self, data: Any):
        """Set calibration data for TensorRT int8 (DataLoader, Dataset, or list/tuple of tensors/ndarrays)."""
        self.calibration_data = data

    def _compile_tensorrt_int8(self, model: PolicyBackbone, in_dim: int):
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT int8 requires CUDA; GPU not available.")
        try:
            import torch_tensorrt
            from torch_tensorrt import ptq
        except Exception as exc:
            raise RuntimeError("torch-tensorrt is required for TensorRT int8 mode.") from exc

        loader = self._get_calibration_loader(in_dim)
        if loader is None:
            raise RuntimeError(
                "TensorRT int8 mode requires calibration_data (DataLoader, Dataset, or list of tensors/ndarrays). "
                "Provide via QuantCompressor(calibration_data=...) with representative samples."
            )
        try:
            first_batch = next(iter(loader))
        except StopIteration:
            raise RuntimeError("Calibration loader is empty.")
        if isinstance(first_batch, (list, tuple)):
            first_batch = first_batch[0]
        if first_batch.ndim == 1:
            first_batch = first_batch.unsqueeze(0)
        batch_size = first_batch.shape[0]
        if first_batch.shape[-1] != in_dim:
            raise RuntimeError(f"Calibration sample last dim {first_batch.shape[-1]} != expected in_dim {in_dim}")

        calibrator = ptq.DataLoaderCalibrator(
            loader,
            use_cache=False,
            algo_type=ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
            device=torch.device("cuda"),
        )

        compiled = torch_tensorrt.compile(
            model.to("cuda").eval(),
            inputs=[torch_tensorrt.Input((batch_size, in_dim))],
            enabled_precisions={torch.int8, torch.float16, torch.float32},
            calibrator=calibrator,
        )
        return compiled
