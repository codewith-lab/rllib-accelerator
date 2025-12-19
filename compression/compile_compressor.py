# path: compression/compile_compressor.py

import time
import numpy as np
from typing import Any, Dict, Tuple, Optional, List
import torch
from ray.rllib.utils.framework import try_import_torch
torch, nn = try_import_torch()

from compression.base import BaseCompressor
from models.policy import PolicyBackbone  # ⚠️ Update to your actual path if needed


class CompileCompressor(BaseCompressor):
    """
    Compressor for torch.compile.

    Key actions:
    - Snapshot train_model.backbone (state_dict clone)
    - Use weight diff to decide whether to recompile
    - Call torch.compile to generate compiled_backbone
    """

    def __init__(self,
                 backend: str = "inductor",
                 diff_threshold: float = 1e-4,
                 device: str = "cpu",
                 recompile_every: int = 2,
                 sparsity_change_threshold: float = 0.05):
        """
        Args:
            backend: torch.compile backend (typically inductor)
            diff_threshold: Recompile if mean snapshot diff exceeds this threshold
            recompile_every: Force recompile every N compressions (handles sparsity shifts)
            sparsity_change_threshold: Recompile when sparsity change exceeds this threshold
        """
        self.backend = backend
        self.diff_threshold = diff_threshold
        self.device_str = device
        self.recompile_every = recompile_every
        self.sparsity_change_threshold = sparsity_change_threshold
        
        # ✅ Supports weight sync
        # For prune+compile pipelines, weights are synced before masks are reapplied
        self.supports_weight_sync = True
        if torch is not None:
            try:
                resolved = torch.device(device)
                if resolved.type.startswith("cuda") and not torch.cuda.is_available():
                    print(f"[CompileCompressor] ⚠️ Device {device} unavailable, fallback to CPU.")
                    resolved = torch.device("cpu")
                self.device = resolved
            except (RuntimeError, TypeError):
                self.device = torch.device("cpu")
        else:
            self.device = None
        
        self._raw_model: Optional[PolicyBackbone] = None
        self._compiled_model: Optional[Any] = None
        self._meta: Optional[Dict[str, Any]] = None
        
        # 📊 Track compression count and sparsity for recompilation logic
        self._compression_count = 0
        self._last_sparsity = 0.0

    # ------------------------------------------------------------
    # 1. snapshot
    # ------------------------------------------------------------
    def snapshot(self, train_model: Any) -> Dict[str, torch.Tensor]:
        """Copy backbone state_dict (no gradients, CPU clone)."""
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
    # 2. Diff check
    # ------------------------------------------------------------
    def should_recompress(self,
                          new_snapshot: Dict[str, torch.Tensor],
                          last_snapshot: Dict[str, torch.Tensor]) -> bool:
        """Decide whether recompilation is needed based on parameter diffs."""

        if last_snapshot is None:
            return True  # Always compress the first time

        diffs = []
        for k in new_snapshot:
            diff_value = (new_snapshot[k] - last_snapshot[k]).abs().mean().item()
            diffs.append(diff_value)

        mean_diff = float(np.mean(diffs))

        return mean_diff > self.diff_threshold

    # ------------------------------------------------------------
    # 3. compress（torch.compile）
    # ------------------------------------------------------------
    def compress(self, snapshot) -> Tuple[Any, Dict[str, Any]]:
        """
        Run torch.compile and return a new compiled_backbone.

        Supports two input forms:
        1. Dict[str, torch.Tensor] - state_dict (from snapshot)
        2. PolicyBackbone - model object (from upstream compressor, e.g., pruning)
        """
        
        # Detect input type
        if isinstance(snapshot, dict):
            # Case 1: state_dict
            if self._meta is None:
                raise RuntimeError("CompileCompressor snapshot meta is missing.")
            in_dim = self._meta["in_dim"]
            num_outputs = self._meta["num_outputs"]
            hidden_dims: List[int] = self._meta["hidden_dims"]
            use_residual: bool = self._meta.get("use_residual", False)
            state_dict = snapshot
        elif isinstance(snapshot, (PolicyBackbone, torch.nn.Module)):
            # Case 2: model object from upstream (e.g., MaskPruneCompressor output)
            bb_input = snapshot
            
            # Unwrap compile wrapper
            if hasattr(bb_input, "_orig_mod"):
                bb_input = bb_input._orig_mod
            
            # If MaskedPolicyBackbone, extract inner backbone
            if hasattr(bb_input, "backbone"):
                actual_bb = bb_input.backbone
                if hasattr(actual_bb, "_orig_mod"):
                    actual_bb = actual_bb._orig_mod
            else:
                actual_bb = bb_input
            
            if self.device is not None and hasattr(actual_bb, 'to'):
                actual_bb = actual_bb.to(self.device)
            
            # Infer structure from the model (for meta)
            in_dim = actual_bb.hidden_layers[0].in_features if len(actual_bb.hidden_layers) > 0 else 4
            num_outputs = actual_bb.policy_head.out_features
            hidden_dims = [layer.out_features for layer in actual_bb.hidden_layers]
            use_residual = actual_bb.use_residual
            
            # Update meta
            self._meta = {
                "in_dim": in_dim,
                "num_outputs": num_outputs,
                "hidden_dims": hidden_dims,
                "use_residual": use_residual,
            }
            
            # 📊 Calculate current sparsity (for recompilation decision)
            current_sparsity = self._calculate_sparsity(actual_bb)
            sparsity_delta = abs(current_sparsity - self._last_sparsity)
            
            # 🔧 Periodic recompilation logic
            self._compression_count += 1
            force_recompile = False
            recompile_reason = None
            
            # Reason 1: Periodic recompilation (every N compressions)
            if self.recompile_every > 0 and self._compression_count % self.recompile_every == 0:
                force_recompile = True
                recompile_reason = f"periodic (every {self.recompile_every} compressions)"
            
            # Reason 2: Significant sparsity change (e.g., pruning increased zeros)
            if sparsity_delta > self.sparsity_change_threshold:
                force_recompile = True
                recompile_reason = f"sparsity change ({self._last_sparsity*100:.1f}% → {current_sparsity*100:.1f}%)"
            
            if force_recompile:
                print(f"[CompileCompressor] 🔄 Forcing recompilation: {recompile_reason}")
                self._compiled_model = None
                self._raw_model = None
                self._last_sparsity = current_sparsity
            
            # ✅ Try to reuse compiled model if structure matches and no force recompile
            reused = False
            if self._compiled_model is not None and self._raw_model is not None:
                # Check if we can reuse by comparing structure
                try:
                    # Extract state dicts to compare structure
                    old_state = self._raw_model.state_dict()
                    new_state = actual_bb.state_dict()
                    
                    # Check if all keys and shapes match (structure identical)
                    structure_match = True
                    if set(old_state.keys()) != set(new_state.keys()):
                        structure_match = False
                    else:
                        for key in old_state.keys():
                            if old_state[key].shape != new_state[key].shape:
                                structure_match = False
                                break
                    
                    if structure_match:
                        # Structure matches! Just update weights in existing model
                        # This preserves hooks from pruning!
                        t0 = time.time()
                        self._raw_model.load_state_dict(new_state, strict=True)
                        latency = time.time() - t0
                        
                        # Reuse existing compiled version
                        compiled_bb = self._compiled_model
                        actual_bb = self._raw_model  # Use existing model
                        reused = True
                        
                        print(f"[CompileCompressor] ♻️  Reused compiled model (structure unchanged, {latency:.4f}s to update weights)")
                    else:
                        # Structure changed, need fresh compile
                        reused = False
                        self._compiled_model = None
                        self._raw_model = None
                        
                except Exception as exc:
                    # Any error, fallback to fresh compile
                    print(f"[CompileCompressor] ⚠️ Failed to check structure match: {exc}")
                    reused = False
                    self._compiled_model = None
                    self._raw_model = None
            
            if not reused:
                # Fresh compilation needed
                t0 = time.time()
                compiled_bb = torch.compile(actual_bb, backend=self.backend)
                latency = time.time() - t0
                
                # Save references
                self._raw_model = actual_bb
                self._compiled_model = compiled_bb
                print(f"[CompileCompressor] 🔧 Compiled new model ({latency:.4f}s)")
            
            state_dict = None  # Not needed
            
        else:
            raise TypeError(f"CompileCompressor.compress() expects Dict or nn.Module, got {type(snapshot)}")

        # Only attempt reuse when the snapshot is a dict
        if isinstance(snapshot, dict):
            reused = False
            if self._compiled_model is not None and self._raw_model is not None:
                # Check if structures match
                try:
                    load_start = time.time()
                    self._raw_model.load_state_dict(state_dict)
                    latency = time.time() - load_start
                    compiled_bb = self._compiled_model
                    reused = True
                except RuntimeError:
                    # Structure mismatch, need to recompile
                    reused = False
                    self._compiled_model = None
                    self._raw_model = None
            
            if not reused:
                bb = PolicyBackbone(in_dim, num_outputs, hidden_dims, use_residual)
                if self.device is not None:
                    bb = bb.to(self.device)
                bb.load_state_dict(state_dict)

                t0 = time.time()
                compiled_bb = torch.compile(bb, backend=self.backend)
                latency = time.time() - t0

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
    
    def _calculate_sparsity(self, backbone: PolicyBackbone) -> float:
        """
        Calculate sparsity (percentage of near-zero weights) in backbone.
        
        Used to detect when pruning has increased sparsity, triggering recompilation
        to let torch.compile optimize for the new sparse pattern.
        
        Returns:
            float: Sparsity ratio (0.0 = dense, 1.0 = all zeros)
        """
        try:
            total_params = 0
            zero_params = 0
            for layer in backbone.hidden_layers:
                weight = layer.weight.data
                total_params += weight.numel()
                zero_params += (weight.abs() < 1e-8).sum().item()
            
            return zero_params / total_params if total_params > 0 else 0.0
        except Exception:
            return 0.0
