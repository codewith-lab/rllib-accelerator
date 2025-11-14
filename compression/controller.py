# path: compression/controller.py

import threading
from typing import Any, Dict, List, Optional


class CompressionController:
    """控制 CompressionPipeline 的同步/异步执行。

    使用方式：
    - SYNC：直接调用 run_sync() → 立即压缩并返回结果
    - ASYNC：调用 trigger_async() → 在后台压缩
              下一次 epoch 用 try_swap() 检查是否有可用结果

    pending_outputs/pending_meta 会被锁保护。
    """

    def __init__(self, pipeline, mode, model_lock=None):
        """
        参数：
            pipeline: CompressionPipeline 实例
            mode: CompileMode 枚举（NONE / SYNC / ASYNC）
            model_lock: threading.Lock，用于保护训练模型的 snapshot
        """
        self.pipeline = pipeline
        self.mode = mode
        self.model_lock = model_lock

        # 保护 snapshot/pipeline 状态，避免并发压缩互相覆盖
        self.pipeline_lock = threading.Lock()

        # 异步 pending（等待 swap）
        self.pending_lock = threading.Lock()
        self.pending_outputs: Optional[List[Any]] = None
        self.pending_meta: Optional[Dict[str, Any]] = None

    def _take_snapshot(self, train_model: Any):
        if self.model_lock is not None:
            with self.model_lock:
                return self.pipeline.take_snapshot(train_model)
        return self.pipeline.take_snapshot(train_model)

    def _run_pipeline(self, snapshot: Any, epoch: int):
        with self.pipeline_lock:
            return self.pipeline.maybe_compress_with_snapshot(snapshot, epoch)

    # ============================================================
    # 同步执行
    # ============================================================
    def run_sync(self, train_model: Any, epoch: int):
        """直接阻塞执行 pipeline.maybe_compress。

        返回：
            outputs, meta
        """
        snapshot = self._take_snapshot(train_model)
        outputs, meta = self._run_pipeline(snapshot, epoch)
        return outputs, meta

    # ============================================================
    # 异步执行
    # ============================================================
    def trigger_async(self, train_model: Any, epoch: int):
        """在后台线程异步执行 pipeline.maybe_compress。

        snapshot / diff 检测 / compress 都在后台执行。
        结果写入 pending_outputs, pending_meta。
        """
        snapshot = self._take_snapshot(train_model)

        def worker(local_snapshot):
            try:
                outputs, meta = self._run_pipeline(local_snapshot, epoch)

                # 如果 pipeline 判定不需要压缩 → outputs 为 None
                if outputs is None:
                    return

                with self.pending_lock:
                    self.pending_outputs = outputs
                    self.pending_meta = meta

            except Exception as e:
                print(f"[AsyncCompression] ❌ Error: {e}")

        threading.Thread(target=worker, args=(snapshot,), daemon=True).start()

    # ============================================================
    # 异步 swap（只有 ASYNC 模式需要调用）
    # ============================================================
    def try_swap(self):
        """检查异步压缩是否完成。

        返回：
            outputs, meta
            - 如果没有 pending → (None, None)
            - 如果 pending 存在 → 返回并清除 pending
        """
        with self.pending_lock:
            if self.pending_outputs is None:
                return None, None

            outs = self.pending_outputs
            meta = self.pending_meta

            # 清空 pending（只 swap 一次）
            self.pending_outputs = None
            self.pending_meta = None

        return outs, meta
