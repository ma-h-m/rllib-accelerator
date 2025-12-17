# path: compression/pipeline.py

from typing import Any, Dict, List, Optional, Tuple

from compression.base import BaseCompressor
from compression.policy import CompressionPolicy


class CompressionPipeline:
    """
    统一执行压缩流程的 Pipeline。

    功能：
    - 使用第一个 compressor 抽取 snapshot
    - 根据 CompressionPolicy 判断是否触发压缩
    - 顺序执行多个 compressor（compile/quant/prune/distill）
    - 存储 last_snapshot / last_outputs 用于下一次 diff 检测
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
    # 读取最近一次快照和压缩结果
    # ------------------------------------------------------------
    @property
    def last_snapshot(self) -> Optional[Any]:
        return self._last_snapshot

    @property
    def last_outputs(self) -> Optional[List[Any]]:
        return self._last_outputs

    # ------------------------------------------------------------
    # 单独暴露 snapshot，方便上层控制锁
    # ------------------------------------------------------------
    def take_snapshot(self, train_model: Any):
        return self.compressors[0].snapshot(train_model)

    # ------------------------------------------------------------
    # 核心接口：触发 snapshot → trigger_policy → compressors
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
        参数：
            snapshot: 由 take_snapshot() 产生的快照
            epoch:    当前训练 epoch
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

        # 链式执行：每个 compressor 接收前一个的输出
        current_input = snapshot
        
        for idx, compressor in enumerate(self.compressors):
            # 第一个 compressor 接收 snapshot
            # 后续 compressor 接收前一个的输出
            if idx == 0:
                out, info = compressor.compress(snapshot)
            else:
                # 如果前一个输出是模型，让这个 compressor 直接处理
                out, info = compressor.compress(current_input)
            
            outputs.append(out)
            meta[compressor.__class__.__name__] = info
            current_input = out  # 传递给下一个

        self._last_snapshot = snapshot
        self._last_outputs = outputs

        return outputs, meta
