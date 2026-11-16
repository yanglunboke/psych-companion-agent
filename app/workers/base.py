"""Worker 基类。

四个 Worker 的共同点：**职责固定、输入输出固定、不跨职责说话**。
每个 Worker 就是一个带固定系统提示的 AgentScope ``Agent``，输入由编排器
显式拼好（本轮原话 + 12 轮原话 + 侧写摘要 + 上游 JSON + 检索到的卡），
输出固定结构的 JSON 或一段学生可见的文本。
"""

from __future__ import annotations

import logging

from agentscope.agent import Agent, ReActConfig
from agentscope.model import ChatModelBase
from agentscope.tool import Toolkit

logger = logging.getLogger(__name__)


class BaseWorker:
    """Worker 基类。"""

    name: str = "worker"
    #: 最大推理-行动迭代次数。Worker 不需要长链，默认给到 6 轮足够。
    max_iters: int = 6

    def __init__(self, model: ChatModelBase, *, max_iters: int | None = None) -> None:
        self.model = model
        if max_iters is not None:
            self.max_iters = max_iters

    # ------------------------------------------------------------------
    def build_agent(
        self,
        system_prompt: str,
        *,
        toolkit: Toolkit | None = None,
        name: str | None = None,
    ) -> Agent:
        """构造一个"每轮新建"的 Agent。

        不复用 Agent 实例：本轮的上下文全部由编排器显式拼装注入，
        Agent 内部状态不应该跨轮累积，否则和 Redis 的 12 轮窗口口径冲突。
        """
        return Agent(
            name=name or self.name,
            system_prompt=system_prompt,
            model=self.model,
            toolkit=toolkit,
            react_config=ReActConfig(max_iters=self.max_iters),
        )


__all__ = ["BaseWorker"]
