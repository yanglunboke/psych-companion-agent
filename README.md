# 中学积极心理陪伴智能体

> 4 WORKERS 编排 · 情绪轨迹 · 按需资源建议 · 高危固定上报流程

基于 [AgentScope](https://github.com/agentscope-ai/agentscope) 2.x 实现的陪伴系统，
包含同源聊天前端与 FastAPI 后端。一个编排器调度多个**职责固定**的 Worker；
每轮走 ReAct（按需调用）还是 Plan-execute（固定计划）**由代码决定**，而不是交给模型自由发挥。

---

## 目录

- [一、架构总览](#一架构总览)
- [二、运行方式](#二运行方式)
- [三、目录结构](#三目录结构)
- [四、配置项](#四配置项)
- [五、阶段进度](#五阶段进度)
- [六、设计口径与假设](#六设计口径与假设)

---

## 一、架构总览

```
01 学生输入与安全分流
   学生发起对话(文本/学生ID/会话ID) → 会话上下文(最近12轮 + 学生侧写)
        → 情绪识别[WORKER 0] → 安全路由(关键词命中 ∨ 风险分≥阈值)
                                              │
                    常规对话 ────────────────┤
                    ReAct 按需调用           │ 高危
        需求澄清[WORKER 1] → 策略检索工具 →   │
        积极引导[WORKER 2]（流式）           ↓
        资源建议[仅 intervention_fit=需要练习] 03 高危处理 Plan-execute 固定计划
                                     共情回复安全模式 → escalate_risk → 通知老师/家长

04 持续记录与支撑能力
   短期记忆[Redis] · 情绪轨迹[Postgres] · 学生侧写[Postgres]
   策略知识库[RAG] · 日志与复盘[TRACE]
```

**风险门控规则（两路，缺一不可）**

| 路径 | 机制 | 阈值 |
| --- | --- | --- |
| 第一路 | 关键词词表匹配（自伤/自杀、伤人、未成年人受侵害） | 命中即 `lexicon_hit=true` |
| 第二路 | 情绪识别 Worker 内置风险分（**小模型，temperature=0**） | `risk_score ≥ 0.72` |

两路结果交回同一个函数，入参 `lexicon_hit` + `risk_score`，由代码决定启用哪条流程。

---

## 二、运行方式

### 1. 安装依赖

```bash
# Windows（Git Bash）
python3 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# macOS / Linux
python3 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt   # macOS: .venv/bin/python
```

> 国内网络建议加镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`

### 2. 配置（可选）

```bash
cp .env.example .env      # 不改也能跑，默认走离线规则模型
```

### 3. 启动

```bash
./.venv/Scripts/python.exe run.py            # http://127.0.0.1:8000
./.venv/Scripts/python.exe run.py --reload   # 开发热重载
```

Windows 的 cmd / PowerShell 用户，进入项目目录后用：

```powershell
.\.venv\Scripts\python.exe run.py
```

终端出现 `Uvicorn running on http://127.0.0.1:8000` 后保持该窗口打开，
**在浏览器访问 [http://127.0.0.1:8000/](http://127.0.0.1:8000/)** 即可使用聊天页面；
API 调试页在 `/docs`。启动服务会占用终端，属于正常现象；用 Ctrl+C 停止。
浏览器不要用 `file://` 打开 `index.html`，页面需要同源 `/api/chat` 接口。

页面提供流式对话、按需展示练习卡、风险上报状态、移动端布局。
“开始新对话”会创建新的随机会话与学生标识，并清空**页面**消息；
它**不会删除**服务端已经保存的原话、轨迹和侧写。页面不会在浏览器持久保存聊天记录；
刷新后不能恢复历史消息。当前项目未实现用户认证，仅适合可信的本机开发环境；
不要直接暴露到公网或用真实未成年人身份信息测试。

### 4. 验证

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/meta

./.venv/Scripts/python.exe -m pytest -q
```

### 4.1 语义向量检索（可选，PGVector + bge-m3）

默认 `EMBEDDING_BACKEND=hashing` + `VECTOR_BACKEND=memory`，用本地哈希向量 +
进程内索引把「向量召回」这一路跑通（**无语义**，与 BM25 高度重合）。要升级为
真实语义向量库，需要三步：

**① 起 Postgres + pgvector**（需已安装 Docker）：

```bash
docker compose up -d   # 暴露在 localhost:5433
```

**② 装语义向量依赖**：

```bash
./.venv/Scripts/python.exe -m pip install pgvector sentence-transformers
```

**③ 下载 bge-m3 权重**（约 2.3GB，二选一）：

```bash
# 方式 A：ModelScope（国内推荐，需 pip install modelscope）
./.venv/Scripts/python.exe -c "from modelscope import snapshot_download; print(snapshot_download('BAAI/bge-m3'))"
#   → C:\Users\<你>\.cache\modelscope\models\BAAI--bge-m3\snapshots\master

# 方式 B：HuggingFace 镜像
#   set HF_ENDPOINT=https://hf-mirror.com
```

**④ 改 `.env`**（`EMBEDDING_MODEL` 指向第③步下载的本地路径即可，无需再联网）：

```dotenv
EMBEDDING_BACKEND=bge
EMBEDDING_MODEL=C:/Users/<你>/.cache/modelscope/models/BAAI--bge-m3/snapshots/master
VECTOR_BACKEND=pgvector
PG_DSN=postgresql+psycopg://psych:psych@localhost:5433/psych
```

启动后访问 `/api/meta`，`runtime.vector` 会如实显示 `embedder=bge-m3`、
`store=pgvector`；连不上 PGVector 时自动降级回进程内哈希索引（服务照常能跑）。

**PGVector 是向量的权威源**：启动时装配阶段会先从库里**装载**已持久化的向量，
只对新增/缺失的卡编码，并自动清理库里已删除的卡。因此重启**不会**重复编码
（bge-m3 编码几十张卡要几十秒），`runtime.vector` 里能看到明细：

```json
{ "loaded_from_pgvector": 46, "encoded": 0, "pruned": 3, "cards": 46 }
```

> **Windows 提示**：`run.py` 已在 Windows 下强制使用 `SelectorEventLoop`
> （psycopg3 异步驱动不兼容默认的 `ProactorEventLoop`），无需手动干预。

### 5. 对话接口

```bash
# 同步（脚本 / 联调）
curl -X POST http://127.0.0.1:8000/api/chat/sync \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","student_id":"stu1","text":"这次月考又考砸了，我真的太笨了"}'

# SSE 流式（学生端）
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","student_id":"stu1","text":"这次月考又考砸了"}'

# 按 trace id 查这一轮的工具调用顺序
curl http://127.0.0.1:8000/api/trace/<trace_id>
```

SSE 事件类型：`meta`（trace_id 等）→ `delta`（学生可见增量文本）→
`card`（练习卡）→ `escalate`（高危上报）→ `decision`（路由决策）→
`blocked`（高危轮未过闸门，回复不发给学生）→ `done`（本轮全量结果）。

### 5. 切换到真实大模型

把 `.env` 改成：

```dotenv
MODEL_PROVIDER=openai
MODEL_NAME=MiniMax-M2.7             # 或 qwen-plus / deepseek-chat / gpt-4o-mini ...
MODEL_BASE_URL=https://api.minimaxi.com/v1
MODEL_API_KEY=sk-xxxxxxxx
```

任何 OpenAI 兼容服务都可以直接用（OpenAI / DeepSeek / Moonshot / 通义 / MiniMax /
无问芯穹 / 智谱 / Ollama / vLLM / LM Studio）。**业务代码零改动**，只换后端。

接之前先自检连通性（会同时试 OpenAI 兼容与 Anthropic 兼容两种协议，并打印
服务端原始报错）：

```bash
# 用 .env 里的配置自检
./.venv/Scripts/python.exe scripts/probe_model.py

# 拿一把 key 自动扫一遍常见端点，找到能用的那个并写回 .env
./.venv/Scripts/python.exe scripts/probe_model.py --api-key sk-xxx --scan --save

# 指定地址自检
./.venv/Scripts/python.exe scripts/probe_model.py --base-url https://xxx/v1 --model yyy
```

> 未配置密钥或密钥不可用时，服务会**自动降级**到本地规则引擎（按文档的业务语义
> 做真实推断，不是固定假字符串），全链路照常跑通，失败原因写在 `/api/meta` 里。
> 降级后还会**熔断**（`MODEL_CIRCUIT_BREAKER=true`）：真实模型失败一次就不再重试，
> 避免每轮白跑一个注定失败的请求、白白拖慢首字延迟。

**数据库写入策略**：只要本轮任一 Worker 实际调用了本地规则引擎（包括直接设置
`MODEL_PROVIDER=mock`、真实模型调用失败后自动降级，以及 Worker 内部规则回退），
整轮就不会向长期 SQL 存储写入学生/助手原话、情绪轨迹、学生侧写或工具调用日志。
只有整轮没有调用规则引擎时，才按原逻辑写入这些表。短期会话上下文仍会保留，
规则轮次照常返回回复；高危上报是独立流程，仍会写入配置的 outbox 或发送 webhook，
**并不受“不写数据库”策略限制**。进程内 trace 仍可在服务运行期间查询。
这项改动仅作用于新对话，**不会清理旧数据库记录**。如启用了 Redis 短期记忆，
短期上下文仍可能存放在 Redis 中；“不写数据库”不等于完全不留存。

---

## 三、目录结构

```
app/
├── config.py            全局配置（pydantic-settings）
├── schemas.py           领域模型与 API 契约（唯一口径来源）
├── container.py         依赖装配（配置 → 一整套可用组件）
├── logging_setup.py     日志
├── main.py              FastAPI 装配与同源页面路由
├── static/              前端页面（index.html / style.css / app.js）
├── models/              LLM 适配层
│   ├── base.py          模型工厂（main / risk 双用途）
│   ├── mock_model.py    离线规则模型（AgentScope ChatModelBase 实现）
│   ├── openai_model.py  OpenAI 兼容后端
│   └── rule_engine.py   本地规则引擎（纯函数，可单测）
├── workers/             4 个职责固定的 Worker
│   ├── emotion.py       Worker 0 情绪识别 + 风险分小模型
│   ├── needs.py         Worker 1 需求澄清
│   ├── guidance.py      Worker 2 积极引导（流式）+ 安全回应
│   └── exercise.py      资源建议（仅需要练习时）
├── orchestrator/
│   ├── safety.py        安全门（词表 + 风险分阈值）
│   ├── router.py        Tool Router（风险 / fit / 说话人）
│   ├── escalate.py      escalate_risk + 老师家长通知（上报体白名单）
│   └── orchestrator.py  主流程（ReAct / Plan-execute）
├── rag/
│   ├── tokenizer.py     中英文混合分词（向量与 BM25 共用）
│   ├── embedding.py     向量化（本地哈希向量器，可换真实模型）
│   ├── bm25.py          BM25 索引
│   ├── index.py         卡片内存索引（按 collection 隔离）
│   ├── query.py         结构化查询构造
│   └── retriever.py     四步检索：召回 → RRF → 重排 → 截断
├── memory/
│   ├── store.py         存储接口协议
│   ├── memory_impl.py   进程内实现（12 轮窗口 + 情绪轨迹 + 侧写）
│   ├── redis_impl.py    Redis 短期记忆（RPUSH/LTRIM/LRANGE，窗口按轮算）
│   ├── sql_impl.py      SQL 长期存储（Postgres / SQLite，四张表）
│   ├── factory.py       后端装配与降级
│   └── profile.py       侧写聚合（EMA、证据门槛、优势/年级抽取）
├── observability/
│   ├── trace.py         trace id + 有序 tool_call_log + 落库 sink
│   └── reflection.py    反思模块（常规异步 / 高危同步闸门）
├── prompts/             各 Worker 的系统提示 + 上下文拼装
├── api/                 chat（SSE / sync / trace）+ health
└── data/
    ├── lexicon/high_risk_lexicon.json   高危关键词词表
    └── cards/                            回应策略卡 / 禁止说法卡 / 练习条目
scripts/                 probe_model.py / gen_eval_data.py / eval_retrieval.py
                         eval_profile_recall.py / eval_all.py / expand_knowledge_base.py
eval/                    生成的评测集与报告（retrieval_eval.jsonl / profile_eval.jsonl / report*.json）
tests/                   单元测试
```

### 数据库表（`DB_BACKEND=sql`）

| 表 | 内容 | 对应文档 |
| --- | --- | --- |
| `emotion_trajectory` | 时间、情绪类型、强度、诱因域 | 情绪轨迹（按时间排的流水） |
| `student_profile` | 年级、优势、稳定偏差、需求、风险等级、上一次诱因、证据计数 | 心理侧写 |
| `student_utterance` | 学生与助手原话 + trace_id | 学生对话原话 |
| `tool_call_log` | 工具名、入参、是否成功、耗时 | 工具调用可追溯（跨进程） |

---

## 四、配置项

完整清单见 `.env.example`，几个关键项：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MODEL_PROVIDER` | `mock` | `mock` / `openai` |
| `RISK_THRESHOLD` | `0.72` | 风险分阈值（宁误报勿漏报） |
| `MEMORY_BACKEND` | `memory` | `memory` / `redis` |
| `DB_BACKEND` | `memory` | `memory` / `sql` |
| `SQL_DSN` | `sqlite+aiosqlite:///./data/psych_companion.db` | 生产改成 `postgresql+psycopg://…` |
| `POSTGRES_DSN` | 空 | 填了就优先用它（覆盖 `SQL_DSN`） |
| `STORAGE_FALLBACK` | `true` | 后端连不上时是否降级到进程内 |
| `EMBEDDING_BACKEND` | `hashing` | `hashing`（本地哈希，离线默认）/ `bge`（真实语义模型） |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 语义向量模型（`EMBEDDING_BACKEND=bge` 时生效） |
| `VECTOR_BACKEND` | `memory` | `memory`（进程内）/ `pgvector`（Postgres + pgvector 持久化） |
| `PG_DSN` | 空 | PGVector 连接串（`VECTOR_BACKEND=pgvector` 时生效） |
| `EMBEDDING_DIM` | `1024` | 向量维度（bge-m3 = 1024） |
| `MODEL_FALLBACK_TO_MOCK` | `true` | 模型不可用时是否降级到本地规则引擎 |
| `MODEL_CIRCUIT_BREAKER` | `true` | 真实模型失败后是否熔断（不再每轮重试） |
| `SHORT_TERM_TURNS` | `12` | 短期记忆窗口（最近 12 轮） |
| `INTENSITY_EMA_ALPHA` | `0.3` | 情绪强度指数平滑：`0.7*旧值 + 0.3*本轮` |
| `PROFILE_EVIDENCE_THRESHOLD` | `3` | 认知偏差进入稳定项的证据次数 |
| `RECALL_TOP_K` | `20` | 向量 / BM25 各召回 20 |
| `VECTOR_MIN_SCORE` | `0.35` | 向量相似度下限 |
| `RRF_K` | `60` | RRF 常数 |
| `RERANK_MIN_SCORE` | `0.45` | 重排分数下限 |
| `DYNAMIC_K_BY_FIT` | 见 `.env` | 各 `intervention_fit` 的动态 K（轻量引导 = 2） |
| `SCORE_GAP_THRESHOLD` | `0.15` | 断崖截断：与第 2 名分差超阈值即舍弃后续 |

---

## 五、阶段进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 一 | 工程框架：目录 / 配置 / 领域模型 / 模型抽象 / trace / 健康检查 / 测试脚手架 | ✅ 完成 |
| 二 | 核心模块：安全门、4 个 Worker、编排器 ReAct + Plan-execute、RAG 四步检索、SSE 流式 | ✅ 完成 |
| 三 | 记忆与支撑：Redis 短期记忆、SQL（Postgres/SQLite）情绪轨迹与侧写、trace 落库、反思调优、通知通道、模型容错 | ✅ 完成 |
| 四 | 细节完善：知识库扩充（46 张卡）、检索与侧写召回评测、阈值标定、运行手册 | ✅ 完成 |

---

## 三点五、存储与模型的"永远能跑"设计

阶段三把三处外部依赖都做成了**可降级**：

| 依赖 | 首选 | 降级 | 状态暴露 |
| --- | --- | --- | --- |
| 大模型 | `MODEL_PROVIDER=openai`（任意 OpenAI 兼容服务） | 本地规则引擎 | `GET /api/meta` → `runtime.model[].fallback_reason` |
| 短期记忆 | `MEMORY_BACKEND=redis` | 进程内环形缓冲 | `runtime.storage.short_term.detail` |
| 长期存储 | `DB_BACKEND=sql`（Postgres / SQLite） | 进程内实现 | `runtime.storage.profile.detail` |

降级**不静默**：每次降级都打 WARNING 日志，并在 `/api/meta` 里写明原因。
设置 `STORAGE_FALLBACK=false` / `MODEL_FALLBACK_TO_MOCK=false` 可改为"宁可启动失败也不降级"。

---

## 六、评测

### 6.1 怎么跑

```bash
# 一次跑完：生成评测集 → 两项评测 → 汇总报告（eval/report.json）
./.venv/Scripts/python.exe scripts/eval_all.py

# 或分开跑
./.venv/Scripts/python.exe scripts/gen_eval_data.py
./.venv/Scripts/python.exe scripts/eval_retrieval.py --json eval/report_retrieval.json
./.venv/Scripts/python.exe scripts/eval_profile_recall.py --json eval/report_profile.json
```

### 6.2 检索 Top5 命中率（200 条查询，每条标一张金标主卡）

| 配置 | 命中率 | 相对只走向量 |
| --- | --- | --- |
| A 只走向量 | **90.5%** (181/200) | — |
| B 向量 + BM25（RRF 融合） | 94.0% (188/200) | +3.5 pp |
| C + 重排（读卡片正文） | **100.0%** (200/200) | +9.5 pp |
| D + 断崖/动态 K 截断（完整管线） | 100.0% (200/200) | +9.5 pp |

按样本类型：`exact`（查询带精确术语）向量 91.8%、完整管线 100%；`paraphrase`（同义改写）向量 88.9%、完整管线 100%。

### 6.3 侧写事实召回（50 段对话 × 8 轮，每段问 3 件早期事实）

| 配置 | 年级/升学目标 | 优势 | 上一次诱因 | 合计 |
| --- | --- | --- | --- | --- |
| A1 只留最近 12 条原话 | 100.0% | **0.0%** | 100.0% | **66.7%** |
| A2 只留最近 12 轮原话 | 100.0% | 100.0% | 100.0% | 100.0% |
| B1 12 条原话 + 侧写摘要 | 100.0% | 100.0% | 100.0% | **100.0%** |
| B2 12 轮原话 + 侧写摘要 | 100.0% | 100.0% | 100.0% | 100.0% |

复现了文档描述的现象：**早期优势掉出原话窗口后答不出来（0%），把侧写摘要固定进
Prompt 后捞得回来（100%）**，12 条原话口径下整体 +33.3 个百分点。

### 6.4 两点必须说清楚的差异

1. **评测数据是合成的**。文档给了口径和指标但没给原始数据，`scripts/gen_eval_data.py`
   生成的是合成集。绝对值**不等价于**原系统（62% → 82%、60% → 81%），
   可参考的是各配置之间的**相对关系**。
2. **本地向量器是稀疏哈希占位实现**，余弦分布整体偏低。文档的
   `VECTOR_MIN_SCORE=0.35` 是按**稠密 embedding 模型**标定的，直接套用会砍掉
   近九成召回（阈值敏感度表里 0.35 → 命中率 13%）。因此占位实现自带标定阈值
   0.15，接真实 embedding 模型后自动回到文档口径的 0.35：

   ```
   阈值 0.05 → 91.5%      阈值 0.20 → 80.5%
   阈值 0.10 → 91.5%      阈值 0.25 → 67.0%
   阈值 0.15 → 90.5% ←本地默认生效值
                          阈值 0.30 → 39.5%
                          阈值 0.35 → 13.0%（文档口径）
   ```

   这条标定差异是阶段四评测暴露出来的**真实实现问题**，已修掉。

---

## 七、运行手册

### 7.1 启动

```bash
./.venv/Scripts/python.exe run.py                 # 127.0.0.1:8000
./.venv/Scripts/python.exe run.py --port 8080
./.venv/Scripts/python.exe run.py --reload        # 热重载
```

### 7.2 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 存活探针 + 当前生效的模型后端 |
| GET | `/api/meta` | 全部关键参数 + **运行时状态**（模型降级原因、存储后端） |
| POST | `/api/chat` | SSE 流式对话 |
| POST | `/api/chat/sync` | 同步对话（脚本 / 联调） |
| GET | `/api/trace/{trace_id}` | 查这一轮的工具调用顺序 |
| GET | `/docs` | OpenAPI 交互文档 |

### 7.3 生产部署

```bash
# 1) 真实模型
MODEL_PROVIDER=openai
MODEL_BASE_URL=<你的 OpenAI 兼容地址>
MODEL_API_KEY=<你的密钥>
MODEL_NAME=<模型名>

# 2) Redis（短期记忆最近 12 轮）
MEMORY_BACKEND=redis
REDIS_URL=redis://127.0.0.1:6379/0

# 3) Postgres（情绪轨迹 / 心理侧写 / 对话原话 / 工具调用日志）
DB_BACKEND=sql
POSTGRES_DSN=postgresql+psycopg://user:pw@host:5432/psych

# 4) 高危上报与通知
ESCALATION_WEBHOOK_URL=https://内部网关/escalate
TEACHER_NOTIFY_CHANNEL=https://内部网关/teacher
PARENT_NOTIFY_CHANNEL=outbox          # 家长侧待接入真实通道
```

换 Postgres **不需要改代码**：SQLAlchemy 异步，DSN 换掉即可。

### 7.4 排障

| 现象 | 查哪里 |
| --- | --- |
| 模型没生效，一直在用本地规则引擎 | `GET /api/meta` → `runtime.model[].fallback_reason` |
| 存储连不上，已降级 | `GET /api/meta` → `runtime.storage.*.detail` |
| 想知道这轮调了哪些工具 | `GET /api/trace/{trace_id}`，或查 `tool_call_log` 表 |
| 高危上报有没有送达 | `data/outbox/escalations.jsonl`（未配 webhook 时） |
| 检索召回不对 | 跑 `scripts/eval_retrieval.py` 看阈值敏感度表 |

### 7.5 常用维护命令

```bash
# 扩充策略知识库
./.venv/Scripts/python.exe scripts/expand_knowledge_base.py

# 重新生成评测集并跑评测
./.venv/Scripts/python.exe scripts/eval_all.py

# 模型连通性自检
./.venv/Scripts/python.exe scripts/probe_model.py

# 全量测试
./.venv/Scripts/python.exe -m pytest -q
```

---

## 八、设计口径与假设

实现过程中对文档与架构图做的判断，均记录在此，避免"凭空编造"：

1. **Worker 数量**：架构图标题为「4 WORKERS」，但「资源建议」框被标成 `WORKER 0`
   （与「情绪识别」重号），判定为**图中标注笔误**。按文档口径实现为 4 个 Worker：
   情绪识别、需求澄清、积极引导、资源建议。
2. **风险判定位置**：文档第 4 节的编排流程与第 8 节的数据流描述存在两种顺序。
   实现为：**安全门在编排器入口**（词表命中可短路），词表未命中时由情绪识别产出的
   `risk_score` 与阈值共同决定走 ReAct 还是 Plan-execute。
3. **`intervention_fit` 四档**：文档明确为「只想说说 / 情绪疏导 / 轻量引导 / 需要练习」；
   仅「轻量引导 → K=2」有明确取值，其余档位按 token 预算做了合理推断。
4. **家长通知渠道**：文档只说「通知老师/家长」，未定义家长侧通道。实现为可配置
   `PARENT_NOTIFY_CHANNEL`，默认落地到本地 outbox，**待补充真实通道**。
5. **向量库与 Cross-Encoder**：文档未指定具体实现。检索层做成**可插拔接口**，
   默认提供本地 numpy 余弦 + 纯 Python BM25 + 可配置重排器；生产可替换为
   Milvus / pgvector + 本地 cross-encoder 模型。
6. **评测数据集**：文档给了评测口径（200 条查询、50 段对话）但未给数据。
   阶段四已生成**合成评测集**（`scripts/gen_eval_data.py`），指标为近似参考，
   **不等价于原始评测**。
7. **老师说话时怎么办**：文档只说 Tool Router 要看"说话的是学生还是老师"，
   未定义老师侧口径。实现为：老师输入仍走 ReAct，但**不推学生练习卡**，
   回复只陈述系统实际记录到的事实（情绪/诱因/干预档位）。
8. **高危轮是否跑情绪识别**：图片脚注写"情绪识别每轮运行"，而文档第 8 问写
   词表命中直接触发上报。实现为**情绪识别每轮都跑**（轨迹需要它），
   但词表命中时跳过需求澄清 / 检索 / 资源建议，直接走 Plan-execute。
9. **重排分数分布**：文档给的 0.86 / 0.81 / 0.52 是原系统的分布。
   本地重排器（字段匹配 + IDF 词项覆盖）的分值口径不同，
   断崖阈值 `SCORE_GAP_THRESHOLD` 需要按实际分布标定，默认 0.15。
10. **向量召回阈值**：文档的 `0.35` 是按稠密 embedding 模型标定的。本地占位向量器
    是稀疏哈希实现，余弦分布整体偏低，照搬会把召回砍到 13%（见 6.4 敏感度表）。
    因此占位实现自带标定值 0.15；接入真实 embedding 模型后自动回到 0.35。
11. **「最近 12 轮原话」的口径**：文档同时出现"12 轮"与"12 轮原话"。评测里
    「原话」按**消息条数**理解（12 条 ≈ 6 轮），这样 8 轮对话的第 1 轮会掉出窗口，
    才能复现文档描述的"早期优势不在窗口里"。本项目的 Redis 窗口按**轮**实现
    （12 轮 = 24 条），两种口径在评测里都给了对照。
