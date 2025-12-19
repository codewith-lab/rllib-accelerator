# path: compression/controller.py

import threading
from typing import Any, Dict, List, Optional


class CompressionController:
    """Control synchronous/asynchronous execution of the CompressionPipeline.

    Usage:
    - SYNC: call run_sync() to compress immediately and return results
    - ASYNC: call trigger_async() to compress in background, then call try_swap()
             on the next epoch to retrieve results

    pending_outputs/pending_meta are protected by locks.
    """

    def __init__(self, pipeline, mode, model_lock=None):
        """
        Args:
            pipeline: CompressionPipeline instance
            mode: CompileMode enum (NONE / SYNC / ASYNC)
            model_lock: threading.Lock to guard training model snapshots
        """
        self.pipeline = pipeline
        self.mode = mode
        self.model_lock = model_lock

        # Protect snapshot/pipeline state to avoid concurrent overwrites
        self.pipeline_lock = threading.Lock()

        # Async pending state (waiting for swap)
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
    # Synchronous execution
    # ============================================================
    def run_sync(self, train_model: Any, epoch: int):
        """Run pipeline.maybe_compress in a blocking manner.

        Returns:
            outputs, meta
        """
        snapshot = self._take_snapshot(train_model)
        outputs, meta = self._run_pipeline(snapshot, epoch)
        return outputs, meta

    # ============================================================
    # Asynchronous execution
    # ============================================================
    def trigger_async(self, train_model: Any, epoch: int):
        """Run pipeline.maybe_compress asynchronously in a background thread.

        Snapshot/diff checking/compress all execute in the background.
        Results are written to pending_outputs and pending_meta.
        """
        snapshot = self._take_snapshot(train_model)

        def worker(local_snapshot):
            try:
                outputs, meta = self._run_pipeline(local_snapshot, epoch)

                # If pipeline decides no compression is needed → outputs is None
                if outputs is None:
                    return

                with self.pending_lock:
                    self.pending_outputs = outputs
                    self.pending_meta = meta

            except Exception as e:
                print(f"[AsyncCompression] ❌ Error: {e}")

        threading.Thread(target=worker, args=(snapshot,), daemon=True).start()

    # ============================================================
    # Async swap (only used in ASYNC mode)
    # ============================================================
    def try_swap(self):
        """Check whether asynchronous compression has completed.

        Returns:
            outputs, meta
            - If no pending result → (None, None)
            - If pending exists → return it and clear pending
        """
        with self.pending_lock:
            if self.pending_outputs is None:
                return None, None

            outs = self.pending_outputs
            meta = self.pending_meta

            # Clear pending (swap only once)
            self.pending_outputs = None
            self.pending_meta = None

        return outs, meta
