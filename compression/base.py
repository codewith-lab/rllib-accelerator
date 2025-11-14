# path: compression/base.py

import abc
import time
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 1. 抽象压缩器接口（compile/quant/prune/distill 继承自这个类）
# ============================================================
class BaseCompressor(abc.ABC):
    """压缩器通用接口。

    每种具体压缩方式必须实现：
    - snapshot(): 如何从 train_model 抽取压缩所需的状态
    - should_recompress(): 判断是否需要重新压缩
    - compress(): 执行实际的压缩行为
    """

    @abc.abstractmethod
    def snapshot(self, train_model: Any) -> Any:
        """从训练模型抽取 snapshot（通常是 state_dict 的某种复制）。

        返回：
            任意用于后续 should_recompress 和 compress 的快照对象。
        """
        raise NotImplementedError

    def should_recompress(self, new_snapshot: Any, last_snapshot: Optional[Any]) -> bool:
        """基于快照判断是否需要重新压缩。可被 override。

        默认策略：如果没有 last_snapshot → 压缩，否则永远压缩。
        """
        if last_snapshot is None:
            return True
        return True

    @abc.abstractmethod
    def compress(self, snapshot: Any) -> Tuple[Any, Dict[str, Any]]:
        """执行压缩。

        返回：
            compressed_model: 压缩后的可推理模型（例如 compiled_backbone）
            meta: 字典，记录 latency、压缩类型等额外信息
        """
        raise NotImplementedError

