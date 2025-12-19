# path: compression/policy.py

from typing import Any, Optional, List

from compression.base import BaseCompressor


class CompressionPolicy:
    """
    Strategy module that controls when compression is triggered.

    Two trigger modes (OR relationship):
    1) Fixed epoch interval (trigger_every)
    2) Difference checking based on weight changes
    """

    def __init__(self,
                 trigger_every: int = 0,
                 enable_diff_check: bool = True,
                 min_epoch_before_compress: int = 0):
        """
        Args:
            trigger_every: Compress every N epochs; 0 disables fixed trigger.
            enable_diff_check: Enable snapshot diff-based trigger logic.
            min_epoch_before_compress: Minimum epoch before any compression.
        """
        self.trigger_every = trigger_every
        self.enable_diff_check = enable_diff_check
        self.min_epoch_before_compress = min_epoch_before_compress

    # ------------------------------------------------------------
    # Fixed-interval trigger
    # ------------------------------------------------------------
    def should_trigger_fixed(self, epoch: int) -> bool:
        """Trigger compression on a fixed interval."""
        if self.trigger_every <= 0:
            return False
        # Check minimum epoch requirement
        if epoch < self.min_epoch_before_compress:
            return False
        return (epoch % self.trigger_every) == 0

    # ------------------------------------------------------------
    # Diff-based trigger
    # ------------------------------------------------------------
    def should_trigger_diff(self,
                            compressors: List[BaseCompressor],
                            new_snapshot: Any,
                            last_snapshot: Optional[Any],
                            epoch: int = 0) -> bool:
        """
        Use each compressor's diff logic to decide.

        Return True when compression should run.
        """
        # Check minimum epoch requirement
        if epoch < self.min_epoch_before_compress:
            return False
            
        if last_snapshot is None:
            return True  # Always compress the first time
        if not self.enable_diff_check:
            return False

        # Compress if any compressor requests recompression
        return any(
            c.should_recompress(new_snapshot, last_snapshot)
            for c in compressors
        )
