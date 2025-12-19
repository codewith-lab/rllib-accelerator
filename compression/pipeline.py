# path: compression/pipeline.py

from typing import Any, Dict, List, Optional, Tuple

from compression.base import BaseCompressor
from compression.policy import CompressionPolicy


class CompressionPipeline:
    """
    Pipeline that orchestrates the compression flow.

    Responsibilities:
    - Use the first compressor to take a snapshot
    - Decide whether to compress via CompressionPolicy
    - Sequentially run multiple compressors (compile/quant/prune/distill)
    - Store last_snapshot / last_outputs for future diff checks
    """

    def __init__(self,
                 compressors: List[BaseCompressor],
                 policy: CompressionPolicy):
        if not compressors:
            raise ValueError("CompressionPipeline requires at least one compressor.")

        self.compressors = compressors
        self.policy = policy

        self._last_snapshot: Optional[Any] = None
        self._last_outputs: Optional[List[Any]] = None

    # ------------------------------------------------------------
    # Read the latest snapshot and compression results
    # ------------------------------------------------------------
    @property
    def last_snapshot(self) -> Optional[Any]:
        return self._last_snapshot

    @property
    def last_outputs(self) -> Optional[List[Any]]:
        return self._last_outputs

    # ------------------------------------------------------------
    # Expose snapshot-taking to let callers control locks
    # ------------------------------------------------------------
    def take_snapshot(self, train_model: Any):
        return self.compressors[0].snapshot(train_model)

    # ------------------------------------------------------------
    # Core interface: snapshot → trigger_policy → compressors
    # ------------------------------------------------------------
    def maybe_compress(self,
                       train_model: Any,
                       epoch: int) -> Tuple[Optional[List[Any]], Dict[str, Any]]:
        snap = self.take_snapshot(train_model)
        return self.maybe_compress_with_snapshot(snap, epoch)

    def maybe_compress_with_snapshot(
        self,
        snapshot: Any,
        epoch: int
    ) -> Tuple[Optional[List[Any]], Dict[str, Any]]:
        """
        Args:
            snapshot: Snapshot produced by take_snapshot()
            epoch:    Current training epoch
        """
        do_fixed = self.policy.should_trigger_fixed(epoch)
        do_diff = self.policy.should_trigger_diff(
            self.compressors, snapshot, self._last_snapshot, epoch
        )
        need_recompress = (do_fixed or do_diff)

        if not need_recompress:
            return None, {
                "skipped": True,
                "reason": "no-change-and-not-fixed-period"
            }

        outputs, meta = self._run_compressors(snapshot)
        return outputs, meta

    def _run_compressors(self, snapshot: Any):
        outputs: List[Any] = []
        meta: Dict[str, Any] = {"skipped": False}

        # Chain execution: each compressor receives the previous output
        current_input = snapshot
        
        for idx, compressor in enumerate(self.compressors):
            # First compressor consumes the snapshot
            # Later compressors consume the prior output
            if idx == 0:
                out, info = compressor.compress(snapshot)
            else:
                # If the previous output is a model, hand it directly to this compressor
                out, info = compressor.compress(current_input)
            
            outputs.append(out)
            meta[compressor.__class__.__name__] = info
            current_input = out  # Pass to the next compressor

        self._last_snapshot = snapshot
        self._last_outputs = outputs

        return outputs, meta
