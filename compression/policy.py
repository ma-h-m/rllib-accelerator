# path: compression/policy.py

from typing import Any, Optional, List

from compression.base import BaseCompressor


class CompressionPolicy:
    """
    控制压缩触发方式的策略模块。

    两种触发方式（OR 关系）：
    1) 固定 epoch 触发（trigger_every）
    2) 基于权重变化检测触发（difference checking）
    """

    def __init__(self,
                 trigger_every: int = 0,
                 enable_diff_check: bool = True,
                 min_epoch_before_compress: int = 0):
        """
        参数：
            trigger_every      : 每隔多少个 epoch 必定触发一次压缩。0 表示关闭。
            enable_diff_check  : 是否启用快照参数差分触发逻辑。
            min_epoch_before_compress : 最小 epoch 数，在此之前不触发压缩。
        """
        self.trigger_every = trigger_every
        self.enable_diff_check = enable_diff_check
        self.min_epoch_before_compress = min_epoch_before_compress

    # ------------------------------------------------------------
    # 固定 epoch 触发
    # ------------------------------------------------------------
    def should_trigger_fixed(self, epoch: int) -> bool:
        """固定周期触发压缩"""
        if self.trigger_every <= 0:
            return False
        # 检查是否达到最小 epoch 要求
        if epoch < self.min_epoch_before_compress:
            return False
        return (epoch % self.trigger_every) == 0

    # ------------------------------------------------------------
    # 基于权重变化 diff 触发
    # ------------------------------------------------------------
    def should_trigger_diff(self,
                            compressors: List[BaseCompressor],
                            new_snapshot: Any,
                            last_snapshot: Optional[Any],
                            epoch: int = 0) -> bool:
        """
        使用各 compressor 的 diff 判断逻辑。

        返回 True 表示需要压缩。
        """
        # 检查是否达到最小 epoch 要求
        if epoch < self.min_epoch_before_compress:
            return False
            
        if last_snapshot is None:
            return True  # 第一次一定要压缩
        if not self.enable_diff_check:
            return False

        # 如果任意一个 compressor 判断需要重新压缩 → 执行压缩
        return any(
            c.should_recompress(new_snapshot, last_snapshot)
            for c in compressors
        )
