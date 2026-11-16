"""中学积极心理陪伴智能体。

包结构::

    app/
      config.py          全局配置（pydantic-settings）
      schemas.py         领域模型与 API 契约
      models/            LLM 适配层（AgentScope ChatModelBase 实现）
      workers/           4 个职责固定的 Worker
      orchestrator/      编排器（ReAct / Plan-execute 路由 + 安全门 + 上报）
      rag/               策略知识库检索（召回 → RRF → 重排 → 截断）
      memory/            短期记忆 + 情绪轨迹 + 心理侧写
      observability/     trace 日志 + 反思模块
      api/               HTTP 路由
"""

__version__ = "0.1.0"
