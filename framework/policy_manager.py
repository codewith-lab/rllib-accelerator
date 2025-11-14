# path: framework/policy_manager.py

import threading
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
                 compile_training_backbone: bool = False):

        self.algo = algo
        self.mode = mode

        if not compressors:
            raise ValueError("PolicyManager requires at least one compressor.")
        if infer_output_index < 0 or infer_output_index >= len(compressors):
            raise ValueError("infer_output_index 超出了 compressors 范围。")
        self._compile_only_mode = (
            len(compressors) == 1
            and compressors[0].__class__.__name__ == "CompileCompressor"
        )
        self._compiled_once = False

        self.compressors = compressors

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

        # 当前 sampler 正在使用的“推理 backbone”
        self.current_infer_model: Optional[Any] = None

        # 记录最近一次压缩 metadata（latency 等）
        self.last_meta: Optional[Dict[str, Any]] = None

        self._compile_training_backbone_flag = compile_training_backbone
        self._training_backbone_compiled = False
        if self._compile_training_backbone_flag and self.mode != CompileMode.NONE:
            self._compile_training_backbone_once()

    # ------------------------------------------------------------------
    # 广播 compiled_backbone 到 rollout workers
    # ------------------------------------------------------------------
    def _broadcast_inference_model(self, model):
        """
        将给定 inference backbone 设置到所有 rollout worker 的 policy.model 中。
        你的 CustomPolicyNet 需要实现 set_compiled_backbone()。
        """
        workers = self.algo.workers.remote_workers()

        def _set(worker):
            def inner(policy, pid):
                if hasattr(policy.model, "set_compiled_backbone"):
                    policy.model.set_compiled_backbone(model)
                return 1
            worker.foreach_policy(inner)
            return 1

        if workers:
            ray.get([w.apply.remote(_set) for w in workers])

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

        self._broadcast_inference_model(infer_model)
        if self._compile_only_mode:
            self._compiled_once = True

        print("[AsyncCompile] 🔁 Swapped inference model.")
        return meta

    # ------------------------------------------------------------------
    # 同步/异步触发压缩
    # ------------------------------------------------------------------
    def maybe_trigger(self, epoch: int) -> Optional[Dict[str, Any]]:
        if self.mode == CompileMode.NONE:
            return None
        if self._compile_only_mode and self._compiled_once:
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

            self._broadcast_inference_model(infer_model)
            if self._compile_only_mode:
                self._compiled_once = True

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

        self.train_model.backbone = torch.compile(self.train_model.backbone, backend=backend)
        self._training_backbone_compiled = True
        print(f"[PolicyManager] 🧠 Local training backbone compiled via torch.compile backend={backend}.")
