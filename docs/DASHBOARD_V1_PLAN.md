# Memento Dashboard V1：评估与实施计划

评估日期：2026-09-23。依据当前工作区源码；本文是实施设计，不代表功能已实现或验收通过。

## 结论与边界

方案可行。保留 AgentHook → 独立 SQLite → aiohttp → React 的主线；Memory、Knowledge、Cron 继续调用当前运行实例。无需引入另一套 Agent 后端或观测平台。

V1 是本机运行检查台：Overview 判断运行情况，Runs 定位执行步骤，Memory 检查记忆及召回，Knowledge 主动执行检索诊断，Tasks 控制已有任务。后三页不是通用 tracing 平台通常能直接提供的业务能力。

下列内容是建议修订，实施时应作为最终设计使用，尤其是 Memory GET 的语义、Run 边界和同步 SQLite 的调用位置。

## 源码核对与必要修订

| 位置 | 真实行为 | 实施决定 |
| --- | --- | --- |
| `nanobot/agent/runner.py` | 只有工具轮调用 `before_execute_tools`；最终回复轮只调用 `after_iteration` | 有工具时在 before_execute_tools 写 model event；无工具但有 response 时在 after_iteration 写 model event，每轮恰好一次 |
| `nanobot/agent/runner.py` | `AgentRunResult.usage` 每轮被覆盖，最后返回末轮值 | Trace 逐轮累加 `context.usage`；finish_run 不再累加 result.usage，不改变现有 usage 的业务语义 |
| `nanobot/agent/loop.py` | `_run_agent_loop` 返回三元组，丢失 result.error；维护 `_last_stop_reason` | 将内部调用链改为传递 AgentRunResult，并同步修改调用点和相关测试；trace 和当前 response metadata 均取局部 result，不能临时读全局字段 |
| `nanobot/agent/loop.py` | 普通消息与 system 消息各有一套 prepare/run/save 路径 | 两条路径都包围 Run 生命周期；slash command 不建 Run；不深入 SubagentManager 内部建立父子 trace |
| `nanobot/agent/loop.py` | Memory 后还有 fit_request、同步 consolidation、ContextBuilder、session.save | Run 耗时覆盖这些前台操作；memory event 只衡量 prepare_context；事件耗时之和不必等于总耗时 |
| `tests/agent/test_task_cancel.py`、API timeout | 存在真实任务取消 | 单独捕获 CancelledError，尽力落 cancelled 终态后重新抛出；普通 except Exception 不够 |
| `nanobot/agent/memory_service.py` | sync_markdown 可能导入 Markdown、embedding、写入 DB/文件；search_dynamic 也可能调用 embedding | 推荐 GET memory 只读已提交 snapshot；POST memory/search 在用户点击后执行 sync + core + search，并提示 embedding / Markdown 同步副作用 |
| `nanobot/agent/memory_service.py` | search_dynamic 是召回结果，prepare_context 还会裁剪 token budget，后续 fit_request 还会调整 | Retrieval Debug 明确叫“召回候选”，不得标为某次 Run 最终注入上下文 |
| `nanobot/agent/knowledge_retrieval.py` | 返回最终 parents/children、assessment 和 online 摘要；还存在 online_error 状态 | 直接序列化 to_dict；覆盖五种 status；不伪造初次评估、全部候选、每步时延 |
| `nanobot/agent/tools/knowledge.py`、runner.py | KB 业务错误可以是 JSON status；runner 工具错误依据异常或 Error 前缀 | Overview Tool Errors 按 runner.tool_events 的 error 计数；KB 的 retrieval_error 等另外展示为业务状态，insufficient 不计为工具执行失败 |
| `nanobot/cli/commands.py` | gateway 没有现成 aiohttp listener；OpenAI API 是独立命令/实例 | Dashboard 在 gateway 新建 AppRunner/TCPSite，使用该 gateway 的 agent/cron；本阶段不合并独立 `/v1` server |
| `nanobot/config/schema.py` | gateway.host 默认 0.0.0.0 | 新增 dashboard 独立配置，默认 127.0.0.1；不直接继承 gateway.host |
| `nanobot/cron/service.py` | task 变化可能来自 Dashboard、cron tool、调度完成、一次性任务处理 | task.changed 在 CronService 成功持久化变更后通过可选回调发布；不能只在 PATCH/DELETE handler 发 |
| Dockerfile / pyproject.toml | 只构建 bridge；wheel/sdist 没有 dashboard 静态资源 | 补 dashboard npm 构建和 Python 包资源包含；验证脱离源码目录安装后仍能访问页面 |

### “不影响 Agent”的可实现保证

- Store 初始化、写入、序列化、广播、Dashboard 绑定端口失败都不能改变 Agent 的回复或异常语义。采集失败记录日志；API 对不可用存储返回明确 503，不伪装为“零条运行”。
- sqlite3 本身同步没有问题；但 `busy_timeout=3000` 若在事件循环线程等待锁，会阻塞同进程 Agent / Cron / Channel。保留同步 Store 方法，在异步接入边界使用 `asyncio.to_thread` 执行，连接在执行线程内创建/关闭，不跨线程共享连接。每个 run 的写操作顺序 await；不额外实现消息队列、批处理器或重试框架。
- 这仍然有少量记录延迟，不宣称“完全零开销”。写锁冲突测试必须验证事件循环仍能推进。
- WS subscriber 用有界 Queue，publish 不等待客户端；满队列可关闭该慢客户端，重连后 REST 补状态。不能让客户端阻塞 Agent。
- Python 异常、/stop、API timeout 和有序 shutdown 可以尽力结束 Run；进程被 SIGKILL、断电或 DB 不可写时无法保证终态成功落库。V1 不为此增加 lease / owner / heartbeat 恢复协议，也不能启动时把共享 DB 全部 running 行改为 error。

## 外部实现比较

| 参考 | 可复用的工作方式 | 本项目如何采用 | 不采用什么 |
| --- | --- | --- | --- |
| [Langfuse data model](https://langfuse.com/docs/observability/data-model) 与 [trace best practices](https://langfuse.com/docs/observability/best-practices) | 单次工作对应 trace；按每次模型调用和工具动作构建树；session 关联多次工作 | Run 对应一次 Agent 执行，iteration 下并列 model / tools；使用稳定事件名 | 完整 trace 平台、提示词管理、评测体系 |
| [Phoenix tracing](https://arize.com/docs/phoenix/learn/tracing) | 用执行步骤类型、输入输出和状态定位问题 | 将 Memory / Model / Tools 分开，点击节点看有界输入输出与错误 | OTel / OpenInference 采集接入和独立后端 |
| [Langfuse architecture](https://langfuse.com/handbook/product-engineering/architecture) | 平台级采集与分析分工 | 只借鉴采集故障隔离原则 | ClickHouse、Redis、对象存储、平台 worker |
| [aiohttp lifecycle](https://docs.aiohttp.org/en/stable/web_advanced.html) | AppRunner、异步生命周期和资源清理 | 同一 gateway event loop 启停 REST/WS/static | 第二个 asyncio.run 或 web.run_app |

这是交互和契约层面的复用，不建议复制这些项目的后端代码或前端整页。Memory / Knowledge / Tasks 仍需当前项目自己的薄适配层。

SQLite 的 WAL 允许读写并行，但仍只有一个写入者，也存在 SQLITE_BUSY；仅支持同机可靠本地文件系统的共享 workspace。[SQLite WAL](https://www.sqlite.org/wal.html)

工具链按 Node 20 的约束执行，但明确最低 Node 20.19，并锁定经过验证的 Vite 和依赖版本；当前 Vite 文档的要求不是任意 Node 20。[Vite requirements](https://vite.dev/guide/)

## 页面与视觉设计

设计方向：紧凑、偏开发者工具的检查台。中性底色、细分隔线、单一交互强调色；绿色成功、红色错误、橙色限制/未满足、灰色取消，所有状态同时带文字。数字使用等宽数字，JSON 单独用等宽字体。

- 桌面左侧约 200px 导航；主区留白 24px。Runs 桌面列表约 280px，剩余区域放 Trace，事件详情在树节点下展开；足够宽时可切换右侧详情。不要在普通笔记本上同时塞四列。
- Overview：六个指标紧凑排列；Runs/hour 与 Latency/hour 分成两图、共用时间范围，不用双 Y 轴。下面 Recent 10 runs 可直达详情。无数据显示 `—`，不能伪装 0ms 或 100% 成功率。
- Runs：run list + summary + nested tree。Memory 是根下独立节点，Iteration 是分组，Model / Tools 是同级子节点。工具 call 只显示状态及预览，不显示未采集的单工具耗时。只给 tools batch 标耗时。
- Memory：revision、六类计数、记录列表为主；Retrieval Debug 放独立区域，Core 与 Dynamic 分开。rank 空值显示 `—`，source 为小标签，RRF / similarity 精确展示，不能画成“置信度百分比”。
- Knowledge：查询与执行按钮 → 结果结论和缺口 → Selected Evidence → Online supplementation。显示 `online_attempted`、fetched/ingested/error；不做看似实时但实际没有后端事件的进度步骤条。
- Tasks：列表直接比较状态/计划/下次/上次运行；详情展开已有 history。enable/disable 等成功后更新；delete 用具体任务名确认。无创建、编辑或立即执行按钮。
- 页面状态：loading、empty、error、disabled service、无效 run id、WS disconnected。WS 断开保留已加载页面，显示手动刷新；恢复连接后重新读 REST。
- 窄屏：导航折行为页签、列表与详情上下排列；表格必要时横向滚动。键盘可展开树和确认操作；颜色不是唯一信息载体。

## 数据与 API 契约

### Run

- 一次真实 Agent 执行分配一个 UUID；在 command dispatch 后、prepare_context 前创建。
- UUID 在本地生成后再交给异步数据库调用；取消时仍能关联已开始的写入。to_thread 中已提交的工作不会因外层取消而自动停止，必须等待该 run 在途写入收尾后再写终态，避免 start/finish 乱序。
- 系统消息也采集；不追踪只做 command dispatch 的操作、不采集子 agent 内部细节。
- 独立 API 的空回答重试会调用 process_direct 两次，因此产生两个 run，不能承诺每个 HTTP 请求恰好一个 run。
- 前台 session 保存完成后结束 run；不等待后台 consolidation / knowledge ingestion 或 channel 实际投递。
- 数据库时间统一 UTC ISO8601；duration 使用 perf_counter；UI 转用户时区。
- status：running / completed / error / stopped / cancelled。completed 要求当前结果正常结束且前台保存完成；max_iterations/context_limit 归 stopped，保留原 stop_reason；业务异常和工具致命错误归 error。
- 前台异常覆盖 completed，error 有简短说明；CancelledError 落 cancelled 并继续传播。缺失 usage 只累计 provider 报告的数值，详情标未报告；不是全系统调用总用量，不包含 KB 内部 assessment / embedding / background consolidation。

### Event

沿用 runs/events 两表，补 `events(run_id, event_id)` 索引，查询以 event_id 稳定排序。iteration 存 runner 原值（0 起），UI 显示 +1。

- memory：core_count、retrieved_count、计时、错误时错误摘要。
- model：本轮 token usage、是否提供 usage、stop_reason、工具调用数量、有限的最终文本预览；不保存完整 messages、system prompt、reasoning 内容。
- tools：call_id、name、裁剪后的 arguments、runner status、result_preview、truncated 标识。按 tool_calls 的索引与 results/events 配对，同名多次调用也不合并。
- error：用于 provider 抛异常、上下文构建失败等无法完成普通事件的场景。context budget 在 LLM 调用前失败时，不能创建虚假的 model success。
- 工具结果每条最多 4KiB UTF-8 文本预览；arguments、query 和 error 也设置明确上限与 truncated 标识。已知 credential 字段脱敏；不记录环境变量、provider 配置、HTTP 认证头、完整二进制/base64。任意正文仍可能含私人内容，UI 不宣称完全脱敏。
- 工具批次只有 batch latency。嵌套在 kb_search 内部的调用仅通过结果摘要可见，不另造 model/tool event。
- model-step 是 Hook 边界间的耗时，可能包含请求预算处理、provider retry 和流式回调，并非精确模型服务端耗时；命名和 UI 说明保持这个边界。

### Overview 的统一口径

统计过去 24h 内开始的 run：Runs 包含 running；Success Rate = 正常 completed 的终态数 / 全部终态数，排除 running，包含取消和 stopped 在分母；completed 必须具有 stop_reason=completed。无终态返回 null。

Avg/P95 对有 duration 的终态计算；P95 用 nearest-rank：排序后第 ceil(0.95×n) 项。小时桶使用 UTC，前端显示用户时区；无样本的 latency 为 null，run count 为 0。Tool Errors 是同一 run 集合中 tools.calls.status=error 的调用数，不能只统计出错 batch。Enabled Tasks 从真实 CronService 读取当前值。

### HTTP

保留 `/api/dashboard/*` 和 `/dashboard/`。除 GET memory 改为纯 snapshot 读取外，其余路径沿用原方案。

| API | 契约 |
| --- | --- |
| GET overview | 指标、口径所用窗口、24 个小时桶、recent 10 runs；enabled_tasks 从 CronService 合入 |
| GET runs | limit 默认 50，上限 100；用 started_at + run_id 稳定游标，返回 items/next_cursor；列表不返回完整 events |
| GET runs/{run_id} | run + events，未找到 404 |
| GET memory | 已提交 snapshot、revision、六类 counts；不自动触发 embedding / Markdown 导入 |
| POST memory/search | query、limit 校验后 sync_markdown → read_core_memories → search_dynamic；说明有效 limit 还会被 service 的 dynamic_top_k 截断 |
| POST knowledge/search | 调真实 retrieve(query, doc_limit=…, evidence_limit=…)；返回 to_dict；不自动重试，不因 WS 或刷新而重复调用 |
| GET tasks | list_jobs(include_disabled=True)，序列化完整 state/run_history |
| PATCH tasks/{id} | 仅接受严格布尔 enabled；调用 enable_job；未知任务 404 |
| DELETE tasks/{id} | remove_job；成功 204；不存在 404 |
| WS events | 仅 type + run_id/job_id 等失效标识；不传个人内容 |

请求错误返回 400；Memory 同步冲突/无效 Markdown 返回明确 409；上游失败 502，配置/服务不可用 503。KnowledgeRetriever 已返回的 retrieval_error/assessment_error/online_error 是有结构的诊断结果：HTTP 200 原样展示 status，不能丢弃 evidence 后只返回一句网络错误；未处理异常另走 502/500。

对 JSON body 大小、query 长度和 limit 设置合理上限。生产静态页面和 API 同源，不开放任意 CORS；检查 Host，浏览器 mutation/WS 检查 Origin；不添加用户账号或 RBAC。

Memory 原提案的 GET 先同步与“访问无副作用”不同时成立。本计划选择只读 GET；调试搜索会同步并可能写入，因此按钮旁也需说明。若必须自动导入 Markdown，应明确接受该语义后再调整，不能藏在后台自动刷新中。

## 改动地图

| 文件或目录 | 计划工作 |
| --- | --- |
| `nanobot/observability/store.py` | SQLite schema、读写、指标、预览限制 |
| `nanobot/observability/hook.py` | 每次 run 独立 hook、计时与 usage；不持有全局当前 run |
| `nanobot/observability/events.py` | 同进程有界广播，无 MessageBus 依赖 |
| `nanobot/agent/loop.py` | 普通/system turn 生命周期、局部 AgentRunResult、错误/取消、存储故障隔离 |
| `nanobot/api/dashboard.py` | API factory、薄 service 适配、static 和 WS、错误映射 |
| `nanobot/config/schema.py` | Dashboard 配置；默认本机绑定 |
| `nanobot/cli/commands.py` | gateway 注入真实实例、listener startup/shutdown；CLI/API 采集接入 |
| `nanobot/cron/service.py` | 可选变更通知回调，仅用于 invalidation，不改调度语义或持久格式 |
| `dashboard/` | React/TS/Vite、HashRouter、Tailwind/shadcn、Recharts、fetch |
| `pyproject.toml`、Dockerfile、构建相关脚本 | aiohttp 基础依赖、静态资源打包、Node 构建 |
| `tests/dashboard/` | store/hook/loop/API/WS/gateway focused tests |
| 文档 | 启动、构建、数据语义、检索副作用、采集范围、故障排查 |

既有测试参考：`tests/agent/test_runner.py`、`test_hook_composite.py`、`test_task_cancel.py`、`test_loop_save_turn.py`、`test_loop_consolidation_tokens.py`；`tests/test_openai_api.py` 已有 aiohttp TestClient/TestServer；Memory 复用 `nanobot/tests/p1`、`p2`、`p3` 和 `memory_test_utils.py`；Knowledge 复用 `nanobot/tests/knowledge/test_knowledge_retrieval.py`、`test_knowledge_online.py`；Cron 复用 `tests/cron/test_cron_service.py`。

需要变更配置、内部函数返回契约和打包资源。只增加 observability DB 的初始 schema，不迁移 memory/knowledge/cron 数据，不新增通用 migration 框架。

## 实际开发顺序

顺序：P0 契约 → P1 真实采集 → P2 gateway REST → P3 Overview/Runs → P4 Memory → P5 Knowledge → P6 Tasks → P7 realtime/packaging。

P3 完成后已形成第一个可交付闭环。P4–P6 每阶段都完成自己的 API 与页面，避免先堆好所有页面再接数据。

### P0 — 确定契约与运行边界

**解决问题 / 完成状态：** 消除采集含义、失败状态和统计口径歧义；后续任务可直接依据稳定契约实现。

**包含功能与顺序任务：**

1. P0.1 在设计文档固定 Run 生命周期、status/stop_reason 映射、取消和不可恢复故障边界。
2. P0.2 固定 Event data_json 字段、4KiB preview、参数/错误长度和脱敏规则。
3. P0.3 固定 API envelope、分页、错误响应、UTC 时间、Overview 口径。
4. P0.4 固定 GET memory 只读、POST search 同步并可能付费、Knowledge 仅手动执行的语义。
5. P0.5 选定 UI token 和宽窄屏布局；以示例数据核对五页展示字段全部有后端来源。
6. P0.6 确定 DashboardConfig(enabled/host/port)，默认 enabled、127.0.0.1:18790；Dashboard 端口绑定失败只关闭 Dashboard，不让 gateway 退出。

**验收：** 一条两轮工具链路、一条 Memory error、一条 /stop、一个 KB online_error 都能在契约中无歧义表示；所有可见数值说明来源；无单工具时延或虚假 KB 中间状态。

**暂不做：** 运行代码、数据库接入、真实外部检索、设计系统平台。

### P1 — Observability Core 与真实 Agent 接入

**解决问题 / 完成状态：** 真实 turn 自动写入独立 DB，可由 Python 查询，Agent 回复保持原样。

**包含功能与顺序任务：**

1. P1.1 创建 observability 包、Run/Event 数据契约和 Store；建立 runs/events、索引、WAL、busy_timeout、foreign_keys。
2. P1.2 实现 start_run、finish_run、add_event、add_usage、list_runs、get_run；写入使用参数化 SQL、短事务。
3. P1.3 实现 get_overview：时间窗、终态分母、nearest-rank P95、工具调用错误计数、小时桶。
4. P1.4 实现有界文本预览、已知 credential 字段脱敏、truncated 标记；保留 call_id 和 tools 同名调用顺序。
5. P1.5 实现 ObservabilityHook：工具轮在 before_execute_tools 结束 model；最终回复轮在 after_iteration 结束 model；tools batch 结束后写结果；逐轮累加 usage。
6. P1.6 将 `_run_agent_loop` 内部返回契约改为 AgentRunResult，更新两个消息分支、调用点和已有测试；不要新增兼容包装层。
7. P1.7 在 command 后 / Memory 前建立 run；普通/system 两条链路分别记录 memory，前台 session 保存后结束。
8. P1.8 增加异常、context_limit、CancelledError、无内容和 message tool 已发送的收尾路径；异常保持原传播行为。
9. P1.9 Store 方法在异步边界 to_thread；初始化/采集失败仅 log。CLI、gateway、独立 API 使用相同 workspace 路径，独立进程可分别持有 Store。
10. P1.10 编写 test_store.py、test_hook.py、test_loop_trace.py，复用假 provider/tool/memory fixtures；更新采集语义文档。

**验收：**

- Fake provider 两轮产生 1 run、1 memory、2 model、1 tools；末轮文本与不开采集一致。
- 第一轮 usage 10/2，第二轮 20/3，run 为 30/5；只报告最后一轮的 result.usage 不被再加。
- completed、model error、Memory error、max_iterations、context_limit、cancelled 按契约落终态；无 LLM 调用时不产生 model success。
- 同时执行两 session，一成功一失败，run/result/error/usage 不串；不依赖 `_last_stop_reason`。
- Store 不可写/抛异常时 Agent 回复仍正确；持锁线程使 DB 写等待时，事件循环另一个小任务能继续。
- 独立连接可读取已提交结果；长工具输出被截断；command 不建 run；原相关 focused tests 通过。

**暂不做：** HTTP、React、WS、子 agent tracing、工具实现改造、崩溃恢复协议、retention。

### P2 — Dashboard REST 与 gateway 生命周期

**解决问题 / 完成状态：** 同一个 gateway 已能通过 REST 查询 trace；不需要第二个 Agent。

**包含功能与顺序任务：**

1. P2.1 将 aiohttp 提升为基础依赖，更新安装说明与过时的 optional dependency 错误提示；不新增 FastAPI。
2. P2.2 新增 DashboardConfig，并与 gateway 的旧 host 字段分离。
3. P2.3 创建 create_dashboard_app(agent_loop, cron_service, observability_store)，注入原实例；定义统一 JSON 错误输出和输入验证。
4. P2.4 实现 GET overview、runs、runs/{id}；列表有界分页，详情 404；enabled_tasks 读取真实 CronService。
5. P2.5 增加 Host/Origin 校验、请求体大小限制；默认同源，不开放宽泛 CORS。
6. P2.6 在 gateway 的 async run 中 setup AppRunner、启动 TCPSite；监听失败记录日志、显示 Dashboard 不可用，继续原 agent/channel/cron。
7. P2.7 在 finally 中关闭 Dashboard 资源、结束/等待当前 Agent task 的取消收尾、再关闭 Store；保证部分 startup 失败也 cleanup。
8. P2.8 增加 test_api_runs.py、test_gateway_dashboard.py；更新本机启动和 curl 示例。

**验收：** 三个 GET 返回真实 Store 数据；未知 run 404；非法分页 400；相同 agent/cron 对象身份得到验证；端口占用不导致 Agent 停止；shutdown 后 listener 关闭、无悬挂 dashboard task。独立 `/v1` tests 保持通过。

**暂不做：** `/v1` 合并、Memory/Knowledge/Tasks 新接口、WS、认证/RBAC、远程部署。

### P3 — Frontend Shell、Overview、Runs

**解决问题 / 完成状态：** 第一个完整纵向闭环 Agent → DB → REST → Browser 可演示。

**包含功能与顺序任务：**

1. P3.1 创建 dashboard/package.json、lockfile、tsconfig、vite.config.ts；约束 Node ≥20.19 且使用约定 Node 20；build 包含类型检查和 Vite 构建。
2. P3.2 安装并锁定 React、HashRouter、Tailwind、需要的 shadcn 组件与 Recharts；只添加实际使用的组件。
3. P3.3 创建 DashboardLayout、导航、颜色/排版/间距 token、StatusBadge、ErrorState、EmptyState。
4. P3.4 创建 API 类型和 native fetch 客户端，处理非 2xx、AbortController 和组件卸载；不引入额外缓存库。
5. P3.5 实现 Overview 六项指标、两张小时图、recent 10 runs 跳转；空图/空分母正确显示。
6. P3.6 实现 Runs 列表、加载更多、run summary、URL 中的 run_id；切换运行时避免旧请求覆盖新详情。
7. P3.7 实现 TraceTree：Memory → Iteration → Model/Tools → Call；原生展开/折叠和键盘操作；展示 token/arguments/preview/error/truncation。
8. P3.8 Vite 设置 base=/dashboard/，开发代理 `/api/dashboard`；aiohttp 托管构建输出并处理 `/dashboard` 到 `/dashboard/`。
9. P3.9 用 fixture server 和 P1 fake turn 做页面验收，更新本地开发/构建说明；未实现页面不显示可点击假入口。

**验收：** npm ci 与 npm run build 通过；从 gateway 打开 `/dashboard/#/runs/{id}` 可直达真实 run；1+2+1 的事件树正确；选中工具能看到有限预览；404、无数据、长 JSON、窄屏和键盘展开均可用。此阶段用手动刷新即可。

**暂不做：** 三个业务页面、WS、复杂筛选系统、React Flow、DAG、单工具 waterfall、费用分析。

### P4 — Memory 状态与召回诊断

**解决问题 / 完成状态：** 能看到 Agent 当前持久记忆，主动检查真实混合召回；无需复制任何 retrieval 算法。

**包含功能与顺序任务：**

1. P4.1 实现 GET memory，调用 database.read_snapshot，返回 revision、六类 counts 和 records。
2. P4.2 实现 POST memory/search，验证 query/limit，调用 sync_markdown、read_core_memories、search_dynamic。
3. P4.3 序列化 DynamicMemoryHit 的 record、sources、rrf_score、fts_rank、vector_rank、vector_similarity；透传实际 service 排序，不重新打分。
4. P4.4 区分 Memory conflict/Markdown validation（409）、embedding 上游故障（502）、不可用配置（503）；不静默降级为另一套 FTS-only workflow。
5. P4.5 增加 Memory.tsx：revision、六类选择、记录列表、Core/Dynamic 搜索结果；搜索按钮说明同步和 embedding 调用；不自动搜索。
6. P4.6 使用临时真实 MemoryDatabase 与 fake embedder 编写 test_api_memory.py，比较 API 与 service 字段/顺序；更新 Memory 页语义说明。

**验收：** 固定记忆 fixture 的 revision/count/记录正确；相同 query 的 scores/ranks/sources 与 service 一致；超过 dynamic_top_k 时有效结果遵循 service；GET 不调用 embedding、不写 snapshot；POST 的 sync 冲突与上游错误在页面明确显示。

**暂不做：** 编辑、删除、revision diff、embedding 可视化、重实现 hybrid retrieval、宣称 debug hits 就是某次 run 最终上下文。

### P5 — Knowledge 实际 workflow 诊断

**解决问题 / 完成状态：** 用户可主动运行当前 KB 流程并查看证据/缺口/联网补充结果。

**包含功能与顺序任务：**

1. P5.1 实现 POST knowledge/search 参数验证，将 snake_case 参数传给现有 retrieve；retriever 未启用时返回 503。
2. P5.2 原样返回 KnowledgeResult.to_dict，保留 parents/children 关联和五种业务 status。
3. P5.3 为无法处理的上游异常定义 API 错误映射；不增加检索重试或新 ingestion 路径。
4. P5.4 增加 Knowledge.tsx 查询、doc_limit/evidence_limit 和 Run retrieval；按钮旁明确可能调用模型/embedding/rerank/online API 并写入 KB。
5. P5.5 实现结果结论、missing_points、Selected Evidence、Online supplementation；fetched_urls 表达流程记录的抓取尝试，不等同于成功抓取。
6. P5.6 请求中禁用重复提交；路由切换、WS invalidation、普通刷新不能重放 POST；无后端步骤事件时仅显示“执行中”。
7. P5.7 复用 knowledge fixtures 编写 test_api_knowledge.py；mock retrieve 验证接入，复用现有真实 retriever + fake dependencies 验证在线路径；更新按钮副作用说明。

**验收：** sufficient local、insufficient 后 online、retrieval_error、assessment_error、online_error 均显示正确；保留 available evidence 和 online_errors；只调用传入的 retriever 一次；未点击时零查询调用；页面能展示 sufficient=null。自动化测试不使用真实付费 API。

**暂不做：** document manager、全部 rejected candidates、实时阶段时延、历史评估对比、强制联网开关、自动后台检索。

### P6 — Tasks 复用调度服务

**解决问题 / 完成状态：** Dashboard 能查看和控制 gateway 真正执行的任务。

**包含功能与顺序任务：**

1. P6.1 实现 GET tasks，序列化 CronJob、schedule、state 和现有 run_history；不复制 jobs store。
2. P6.2 实现 PATCH tasks/{id}，仅接受 enabled 布尔值，调用 enable_job；返回当前 job。
3. P6.3 实现 DELETE tasks/{id}，调用 remove_job，定义 204/404；不增加运行时执行按钮。
4. P6.4 增加 Tasks.tsx 表格；按 at/every/cron 和 timezone 展示 schedule；统一时间单位和空值。
5. P6.5 实现 history 展开、enable/disable loading/error、任务名删除确认；成功后 REST 刷新。
6. P6.6 用临时 jobs.json 和真实 CronService 编写 test_api_tasks.py；更新任务行为说明：禁用阻止后续调度，不承诺取消已经在执行的任务。

**验收：** GET 可见 disabled job；PATCH false 后禁用；PATCH true 后 next_run 重算；DELETE 后消失；重建 CronService 可读回同样状态；jobs.json schema 不变；API 与 CronTool/调度器看到同一个实例的修改。

**暂不做：** 创建、编辑、立即运行、强制终止正在运行任务、cron 引擎重写、task-run 与 agent-run 新外键。

### P7 — Realtime、打包与最终闭环

**解决问题 / 完成状态：** 同进程运行自动刷新，Docker 和安装后的 Python 包可直接提供五页 Dashboard。

**包含功能与顺序任务：**

1. P7.1 新增 events.py：subscribe/unsubscribe、每客户端有界 Queue、非阻塞 publish、慢客户端处理。
2. P7.2 Store 成功提交后由采集接入点广播 run.started/updated/finished；不把广播失败传播到业务。
3. P7.3 给 CronService 增加可选变更回调，在修改已持久化后发送 task.changed；覆盖 Dashboard mutation、CronTool 变更、调度完成和一次性任务状态变化，保持 jobs.json 不变。
4. P7.4 实现 WS handler、subscriber cleanup、断开检测和 gateway shutdown 关闭连接；两客户端都收到独立事件。
5. P7.5 React 创建一个 WS 连接；按类型合并短时间 invalidation，只 refetch 相关 GET；禁止自动触发 Memory/Knowledge POST。
6. P7.6 WS 恢复连接时全量刷新当前 REST 视图；保留手动刷新。WS 只涵盖 gateway 同进程事件；另一个 CLI/API 进程写入 DB 后，通过手动刷新读取，不承诺跨进程即时广播。
7. P7.7 Docker COPY dashboard package/lock，npm ci/build；把 dist 放入约定的 nanobot/api/dashboard_static；安装 Python 包前完成资源生成或显式复制到运行位置。
8. P7.8 修改 wheel/sdist 资源规则，新增构建步骤说明；不提交 node_modules；验证离开源码目录运行安装包能返回 index、JS/CSS。
9. P7.9 编写 test_events.py/test_api_events.py：双订阅者、慢客户端、断开、重连 REST、CronService 通知来源；完成浏览器真实页面 smoke。
10. P7.10 执行原 Python tests 和 dashboard focused tests、npm 构建、Docker/安装包 smoke；更新 README/CONFIGURATION 中端口、开启/关闭、构建、断线、数据位置与采集限制。

**验收：** 浏览器 Runs 中看到 running→completed，无整页刷新；双浏览器均更新；WS 断开不影响 REST/Agent；重新连接补读；agent 通过 cron tool 修改任务也使 Tasks 更新；构建出的镜像/包可访问五页及静态资源；自动化全程无真实付费查询。

Docker 默认本机绑定与容器网络要分别处理：仅容器内访问时保持 127.0.0.1；需要宿主机浏览器访问时在容器内显式绑定 0.0.0.0，并使用宿主 `127.0.0.1:18790:18790` 端口映射。不能为了容器访问把普通本机默认也改为全网监听。

**暂不做：** 跨进程事件总线、Redis、远程生产部署、RBAC、retention、全量 OTel、回放、成本估计、故障告警。

## 最终验收门槛

1. 用 fake provider + 真实 AgentLoop/临时 DB/无害测试工具完成两轮链路；用户响应和原行为一致。
2. 相同 run_id 可经 SQLite → REST → Runs Tree 追踪到 Memory、Model、Tools 和正确累计用量。
3. 取消/错误路径与跨 session 并发通过；采集不可用时业务仍工作。
4. Memory/Knowledge/Cron API 调用注入的真实服务，结果转换不重写业务算法。
5. npm build、原 Python 测试和新增聚焦测试通过；gateway cleanup 与安装后资源访问通过。
6. 浏览器确认五页、长内容、空态、错误态、键盘和断线状态可用。构建成功不单独等于 UI 验收通过。

本轮仅静态代码核对、公开文档研究和交互示意；未执行 Agent、检索、embedding、数据库业务写入或付费 API，也未声称任何实现测试已通过。
