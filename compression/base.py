# path: compression/base.py

import abc
import time
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 1. Abstract compressor interface (compile/quant/prune/distill inherit from this)
# ============================================================
class BaseCompressor(abc.ABC):
    """Generic compressor interface.

    Each concrete compression implementation must provide:
    - snapshot(): how to extract the required state from train_model
    - should_recompress(): decide whether to recompress
    - compress(): perform the actual compression
    """

    # Marks whether this compressor's output can be synced by weight-only state_dict updates
    # (e.g., torch.compile'd models).
    supports_weight_sync: bool = False

    @abc.abstractmethod
    def snapshot(self, train_model: Any) -> Any:
        """Extract a snapshot from the training model (typically a copy of state_dict).

        Returns:
            Any snapshot object used later by should_recompress and compress.
        """
        raise NotImplementedError

    def should_recompress(self, new_snapshot: Any, last_snapshot: Optional[Any]) -> bool:
        """Determine whether recompression is needed based on the snapshot. Can be overridden.

        Default strategy: compress if last_snapshot is None; otherwise always recompress.
        """
        if last_snapshot is None:
            return True
        return True

    @abc.abstractmethod
    def compress(self, snapshot: Any) -> Tuple[Any, Dict[str, Any]]:
        """Perform compression.

        Returns:
            compressed_model: compressed inference model (e.g., compiled_backbone)
            meta: dict capturing latency, compression type, and other info
        """
        raise NotImplementedError
