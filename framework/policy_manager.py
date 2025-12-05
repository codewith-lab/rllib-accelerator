# path: framework/policy_manager.py

import threading
import time
import ray
from typing import Any, Dict, Optional

from compression.controller import CompressionController
from compression.pipeline import CompressionPipeline
from compression.policy import CompressionPolicy
from compression.base import BaseCompressor
from enum import Enum
from ray.rllib.utils.framework import try_import_torch
torch, _ = try_import_torch()


# ============================================================
# Compile Mode（你的原始定义保留）
# ============================================================
class CompileMode(Enum):
    NONE = "none"
    SYNC = "sync"
    ASYNC = "async"


# ============================================================
# PolicyManager —— glue RLlib & compression system
# ============================================================
class PolicyManager:
    """
    负责：
        - 管理 pipeline & controller（sync/async）
        - 从 RLlib 训练模型抽取 backbone
        - 按策略触发压缩
        - 异步 swap
        - 把 compiled_backbone 广播到所有 rollout workers

    用法：
        manager = PolicyManager(algo, compressors, CompileMode.SYNC, trigger_every=5)
        manager.maybe_swap(epoch)
        meta = manager.maybe_trigger(epoch)
    """

    def __init__(self,
                 algo,
                 compressors: [BaseCompressor],
                 mode: CompileMode = CompileMode.NONE,
                 trigger_every: int = 5,
                 enable_diff_check: bool = True,
                 infer_output_index: int = 0,
                 compile_training_backbone: bool = False,
                 device: str = "cpu",
                 async_warmup: bool = True):

        self.algo = algo
        self.mode = mode

        if not compressors:
            raise ValueError("PolicyManager requires at least one compressor.")
        if infer_output_index < 0 or infer_output_index >= len(compressors):
            raise ValueError("infer_output_index 超出了 compressors 范围。")

        self.device = self._resolve_device(device)

        self.compressors = compressors
        self.async_warmup = async_warmup

        # compression policy
        self.policy = CompressionPolicy(
            trigger_every=trigger_every,
            enable_diff_check=enable_diff_check
        )

        # pipeline + controller
        self.pipeline = CompressionPipeline(compressors, self.policy)
        self.model_lock = threading.Lock()
        self.controller = CompressionController(self.pipeline, mode, self.model_lock)

        # RLlib 的训练模型
        self.train_model = self.algo.get_policy().model

        self.infer_output_index = infer_output_index
        self.infer_compressor_name = compressors[infer_output_index].__class__.__name__
        self._supports_weight_sync = bool(
            getattr(compressors[infer_output_index], "supports_weight_sync", False)
        )

        # 当前 sampler 正在使用的“推理 backbone”
        self.current_infer_model: Optional[Any] = None

        # 记录最近一次压缩 metadata（latency 等）
        self.last_meta: Optional[Dict[str, Any]] = None

        self._compile_training_backbone_flag = compile_training_backbone
        self._training_backbone_compiled = False
        if self._compile_training_backbone_flag:
            self._compile_training_backbone_once()

    # ------------------------------------------------------------------
    # 广播 compiled_backbone 到 rollout workers
    # ------------------------------------------------------------------
    def _broadcast_inference_model(self, model, warmup=False, update_only=False):
        """
        将给定 inference backbone 设置到所有 rollout worker 的 policy.model 中。
        你的 CustomPolicyNet 需要实现 set_compiled_backbone()。
        """
        workers = self.algo.workers.remote_workers()
        state_dict = None
        if update_only and model is not None:
            try:
                state = model.state_dict()
                state_dict = {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                              for k, v in state.items()}
            except Exception as exc:
                print(f"[PolicyManager] ⚠️ Failed to capture compiled state for update-only swap: {exc}")
                update_only = False
                state_dict = None

        def _set(worker):
            def inner(policy, pid):
                did_update = False
                if update_only and hasattr(policy.model, "update_compiled_backbone_weights") and state_dict is not None:
                    try:
                        policy.model.update_compiled_backbone_weights(state_dict)
                        did_update = True
                    except Exception as exc:
                        print(f"[PolicyManager] ⚠️ update_only failed on worker, fallback to full swap: {exc}")
                if not did_update and hasattr(policy.model, "set_compiled_backbone"):
                    policy.model.set_compiled_backbone(model)
                    if warmup and hasattr(policy.model, "warmup_compiled_backbone"):
                        policy.model.warmup_compiled_backbone()
                return 1
            worker.foreach_policy(inner)
            return 1

        if workers:
            ray.get([w.apply.remote(_set) for w in workers])

        if not update_only:
            print("[Broadcast] 📤 Inference backbone updated on all sampler workers.")

    # ------------------------------------------------------------------
    # 异步模式：在每个 epoch 开头尝试 swap（若异步线程已完成）
    # ------------------------------------------------------------------
    def maybe_swap(self) -> Optional[Dict[str, Any]]:
        if self.mode != CompileMode.ASYNC:
            return None

        outputs, meta = self.controller.try_swap()
        if outputs is None:
            return None

        infer_model = self._select_infer_model(outputs)
        if infer_model is None:
            return None

        self.current_infer_model = infer_model
        self.last_meta = meta

        warmup = (self.mode == CompileMode.ASYNC and self.async_warmup)
        update_only = self._should_update_only(meta)
        t0 = time.time()
        self._broadcast_inference_model(infer_model, warmup=warmup and not update_only, update_only=update_only)
        swap_latency = time.time() - t0
        if meta is None:
            meta = {}
        meta.setdefault("SwapLatency", swap_latency)
        if not update_only:
            print("[AsyncCompile] 🔁 Swapped inference model.")
        return meta

    def push_weight_update(self):
        """
        将训练模型最新的 backbone 权重同步到已存在的推理 backbone。
        仅对支持纯权重更新的压缩器（例如 compile）启用。
        """
        if not self._supports_weight_sync:
            return
        if self.current_infer_model is None:
            return

        snapshot = self._snapshot_train_backbone()
        if snapshot is None:
            return

        # 先更新本地推理模型，避免下一次广播仍旧是旧权重
        self._load_state_into_infer(snapshot)

        workers = self.algo.workers.remote_workers()
        if not workers:
            return

        def _update(worker):
            def inner(policy, pid):
                if hasattr(policy.model, "update_compiled_backbone_weights"):
                    try:
                        policy.model.update_compiled_backbone_weights(snapshot)
                    except Exception as exc:
                        print(f"[PolicyManager] ⚠️ Weight push failed on worker, skipping: {exc}")
                return 1

            worker.foreach_policy(inner)
            return 1

        ray.get([w.apply.remote(_update) for w in workers])

    # ------------------------------------------------------------------
    # 同步/异步触发压缩
    # ------------------------------------------------------------------
    def maybe_trigger(self, epoch: int) -> Optional[Dict[str, Any]]:
        if self.mode == CompileMode.NONE:
            return None
        # 同步模式 —— 立即执行
        if self.mode == CompileMode.SYNC:
            outputs, meta = self.controller.run_sync(self.train_model, epoch)
            if outputs is None:
                return None

            infer_model = self._select_infer_model(outputs)
            if infer_model is None:
                return None

            self.current_infer_model = infer_model
            self.last_meta = meta

            update_only = self._should_update_only(meta)
            self._broadcast_inference_model(infer_model, warmup=False, update_only=update_only)
            print("[SyncCompile] ✅ Compiled & swapped immediately.")
            return meta

        # 异步模式 —— 触发后台线程
        elif self.mode == CompileMode.ASYNC:
            self.controller.trigger_async(self.train_model, epoch)
            return None

        return None

    # ------------------------------------------------------------------
    # 获取最近压缩信息
    # ------------------------------------------------------------------
    def get_last_meta(self):
        return self.last_meta

    # ------------------------------------------------------------------
    # 供 Trainer 访问的辅助
    # ------------------------------------------------------------------
    def _select_infer_model(self, outputs):
        if not outputs:
            return None
        if self.infer_output_index >= len(outputs):
            return None
        return outputs[self.infer_output_index]

    def get_infer_compressor_name(self) -> str:
        return self.infer_compressor_name

    def _should_update_only(self, meta: Optional[Dict[str, Any]]) -> bool:
        if not meta:
            return False
        name = self.get_infer_compressor_name()
        info = meta.get(name)
        if not info:
            return False
        return bool(info.get("reused"))

    def _snapshot_train_backbone(self):
        bb = getattr(self.train_model, "backbone", None)
        if bb is None:
            return None
        return {
            k: v.detach().cpu().clone()
            for k, v in bb.state_dict().items()
        }

    def _load_state_into_infer(self, snapshot: Dict[str, Any]):
        if self.current_infer_model is None:
            return
        if torch is None:
            return
        target = getattr(self.current_infer_model, "_orig_mod", self.current_infer_model)
        try:
            device = next(target.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        converted = {}
        for k, v in snapshot.items():
            if torch.is_tensor(v):
                converted[k] = v.to(device)
            else:
                converted[k] = v
        try:
            target.load_state_dict(converted, strict=False)
        except Exception as exc:
            print(f"[PolicyManager] ⚠️ Failed to update local inference model: {exc}")

    # ------------------------------------------------------------------
    # 可选：编译本地训练 backbone 加速前向
    # ------------------------------------------------------------------
    def _compile_training_backbone_once(self):
        if self._training_backbone_compiled:
            return
        if not hasattr(self.train_model, "backbone"):
            return
        if torch is None:
            return

        backend = "inductor"
        primary = self.compressors[0]
        if hasattr(primary, "backend"):
            backend = getattr(primary, "backend") or backend

        if hasattr(self.train_model, "to"):
            self.train_model.to(self.device)

        self.train_model.backbone = torch.compile(self.train_model.backbone, backend=backend)
        self._training_backbone_compiled = True
        print(f"[PolicyManager] 🧠 Local training backbone compiled via torch.compile backend={backend}.")

    def _resolve_device(self, device: str):
        if torch is None:
            return "cpu"
        try:
            resolved = torch.device(device)
            if resolved.type.startswith("cuda") and not torch.cuda.is_available():
                print(f"[PolicyManager] ⚠️ Device {device} unavailable, fallback to CPU.")
                return torch.device("cpu")
            return resolved
        except (RuntimeError, TypeError):
            print(f"[PolicyManager] ⚠️ Invalid device {device}, fallback to CPU.")
            return torch.device("cpu")
