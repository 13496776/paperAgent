# 论文研究工作台

前端使用 Vue 3、TypeScript 和 Vite，后端使用 FastAPI。模型和工具调用沿用项目自定义接口，OpenAI 和 DeepSeek 使用兼容的聊天接口。

在项目根目录执行 `pip install -e '.[web,papers]'`。配置 Redis 的 `REDIS_URL`，设置 `CORECODER_JWT_KEY`、`CORECODER_JWT_ISSUER`、`CORECODER_JWT_AUDIENCE`，并由现有身份服务签发包含 `sub`、`iss`、`aud`、`iat` 和 `exp` 的 HS256 JWT。摘要数据库由 `CORECODER_SQLITE_PATH` 指定。设置 `OPENAI_API_KEY` 或 `DEEPSEEK_API_KEY`，可通过 `OPENAI_MODEL`、`DEEPSEEK_MODEL`、`OPENAI_BASE_URL`、`DEEPSEEK_BASE_URL` 覆盖默认设置。

登录后可在“论文语料库”上传 PDF，系统提取文本并分块入库。也可在研究问题中要求在线检索，或要求导入 arXiv 论文。用户导入的论文按 JWT 用户 ID 隔离，论文元数据、页码和分块正文存于 `CORECODER_SQLITE_PATH`；向量持久化在 Chroma。也可按照 [论文语料说明](PAPER_QA.md) 创建共享 JSON 索引，并将 `PAPER_CORPUS_PATH` 指向该文件。首次使用会初始化嵌入模型和交叉编码器，可用 `PAPER_EMBED_MODEL`、`PAPER_RERANK_MODEL` 指向本地模型。

上传时可以选择多个 PDF，页面会逐个上传并显示导入、重复和失败状态；失败的文件可以再次尝试。批量上传从每个 PDF 读取元数据，手工补充的元数据只用于单篇上传。

运行 `uvicorn corecoder.web:app --host 127.0.0.1 --port 8000`。在 `web` 目录运行 `npm ci` 和 `npm run dev`。页面输入 JWT 和会话 ID，生成计划并确认后执行。可用同一会话 ID 加载任务状态，已完成后可以提交修改要求。

Supervisor 根据问题生成依赖有序的任务，五类 worker 分别对应数据接入、推理分析、文献检索、生成交付和测评运维。文献检索 worker 支持本地论文清单与计数、语义检索及在线检索，数据接入 worker 支持从 arXiv 导入 PDF，测评 worker 支持标注指标计算。其他角色由模型进行文本分析，尚未接入图表视觉识别、代码执行、文件导出或运维部署工具。任务状态和工具调用结果存储在 Redis，最后一次读取或写入 24 小时后过期，会话较长时将旧对话摘要保存到 SQLite。已完成的研究结果、子任务输出和执行追踪按用户及会话归档到 `CORECODER_SQLITE_PATH` 指定的 SQLite 数据库。Redis 会话过期后，仍可用原会话 ID 加载最近一次归档的研究结果。未完成的任务和对话上下文随 Redis 会话过期，SQLite 中的结果及摘要会保留。归档功能上线前已完成的结果仍按原 Redis 会话的有效期保留。计划生成后等待用户确认；确认后页面显示提交中，并从后端同步工作流及子任务状态，连接失败时显示同步错误。会话 ID 在当前浏览器标签页中保留，刷新后可继续加载。执行过程中可请求暂停或取消，系统在当前模型或工具调用完成后处理请求。已保存的工具回执会在恢复时使用，结果不明的工具调用不会自动重复执行。

执行中的暂停与取消为协作式操作，不能强制中断正在进行的模型请求或工具调用。子任务使用受限工具列表，但尚未提供操作系统级沙箱或独立容器。模型输出及引用仍需核验。前后端和 Redis 的访问控制、传输加密和部署配置需要根据实际运行环境设置。

PDF 提取使用 pypdf，只支持含文本层的 PDF，扫描件需要先 OCR。标题、作者和 DOI 尝试从 PDF 元数据及文本提取，缺失或错误时需核对。在线搜索与下载需要访问 `export.arxiv.org` 和 `arxiv.org`；模型需要从 Hugging Face 下载或配置本地路径。导入时保留提取的正文分块及元数据，不保存原始 PDF。Python 的 SQLite 需要启用 FTS5。个人语料通过 SQLite FTS5 进行关键词召回、通过 Chroma 进行向量召回，再融合重排；已有向量可在服务重启后复用。

上传 PDF 时可手工填写标题、作者、DOI、期刊和发表年月；手工值优先。未提供时标题和作者取 PDF 文档元数据，标题仍缺失则使用文件名；DOI 仅从 PDF 元数据和首页尝试提取，可能缺失或有误，应人工核对。页数由 PDF 实际页面数获取。系统不将 PDF 创建时间视为发表时间。arXiv 导入保存其 API 提供的标题、作者、DOI 及首次公开年月（`date_source=arxiv_published`），该日期不一定是正式期刊或会议发表日期；手填日期标记为 `user`。无法确定的日期、DOI 等保持空值，历史数据不会自动补全；相同 PDF 再次导入会去重，不更新已有元数据，需删除后重新导入。更新后 SQLite 自动添加月份和日期来源列，建议升级前停止服务并备份 SQLite 和 Chroma。

## Windows CMD 一体化启动

在项目根目录创建并激活 Python 虚拟环境，然后执行一次依赖安装。需要事先安装 Python 3.10 以上及 Node.js/npm。

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python run_paperagent.py --setup
```

执行 `python run_paperagent.py`。首次运行会创建 `paperagent.config.json` 并提示编辑该文件。填写 `deepseek_api_key` 或 `openai_api_key`，以及 `embedding_model`、`reranker_model` 的本地路径或 Hugging Face 名称。JSON 中 Windows 路径使用正斜杠（例如 `D:/models/bge-m3`）或双反斜杠。配置文件包含密钥，已加入 `.gitignore`，请勿上传或分享。

再次执行 `python run_paperagent.py`。脚本读取配置，生成本机 JWT，启动后端和前端，打开一次性本地登录链接并自动填入令牌。若浏览器未自动打开，可复制终端中的一次性链接；链接有效期为 10 分钟且只能使用一次。也可将终端中的备用令牌手动粘贴到“访问令牌”。默认优先使用已配置的 DeepSeek；仅配置 OpenAI 时默认选择 OpenAI，也可在网页手动切换。Ctrl+C 停止脚本启动的服务。令牌有效期为 24 小时；启动时生成新令牌，并将默认签名密钥保存在项目目录的 .paperagent-jwt-key 文件中供后续重启使用。

这里的 24 小时从 JWT 签发时间开始计算，使用令牌不会自动延长有效期；10 分钟只限制一次性登录链接，并不限制换取后的 JWT。自行运行后端时由外部身份服务签发的 JWT，其有效期由签发方的 `exp` 决定，项目不会修改它。保管好 .paperagent-jwt-key 并随数据库备份，删除或更换密钥会使此前签发的令牌失效。设置 CORECODER_JWT_KEY 时以该环境变量为准。升级前未保存的旧签名密钥无法恢复，因此旧令牌需要重新获取。

脚本使用已有 Redis。如果默认地址的 Redis 未运行且 Docker 可用，会启动一个本地 Redis 容器并在退出时停止，数据保存在 Docker 卷 `paperagent-redis-data`。使用其他 Redis 地址时需要自行启动该服务。执行 `python run_paperagent.py --check` 可运行 Python 测试和前端构建。Linux 或 macOS 也可使用此脚本，虚拟环境使用 `source .venv/bin/activate` 激活。

`chunk_size`、`chunk_overlap`、`rerank_threshold`、`retrieval_limit` 和 `retrieval_candidates` 可在配置中调整。环境变量优先于 JSON；更改后需重启服务。新参数作用于新导入的论文，已有分块不会自动重建。原始 PDF 未存储，相同文件重复上传会被去重；如要重新分块，可删除该论文后重新导入，或使用新 SQLite 文件。JSON 文件建库需重新执行 `corecoder --index-papers`。

任务执行时，页面显示当前模型调用、索引构建、关键词检索、查询向量编码和重排等步骤及耗时。模型调用超时默认每次 60 秒，可通过 JSON 中的 `llm_timeout_seconds` 调整，程序可能对暂时性错误重试。统计本地论文使用 `list_corpus_papers` 工具，按用户论文记录及共享索引中的论文 ID 计数，不依赖检索返回的片段数。

## 论文数据与迁移

更新代码后执行 `python run_paperagent.py --setup` 安装 Chroma 等依赖，然后重启。`sqlite_path` 保存论文元数据、分块正文、索引状态和会话摘要；`chroma_path` 指定向量目录。旧配置没有 `chroma_path` 时，默认目录为 SQLite 文件去掉扩展名后加 `-chroma`。两处数据都应备份，并在服务停止时备份以保证一致。

旧 `paper_corpus` JSON 记录在首次访问时复制到新表，原表保留；旧论文会显示为 `indexing`，首次检索时使用当前 embedding 模型补建向量。迁移不保存或补充原 PDF，历史记录缺失的页数、页码和 CCF 信息会保持未知。新上传可手工填写标题、作者、期刊、年份、DOI、CCF 等级和来源；CCF 信息不会自动核验。PDF 按页面保留页码并在页面内分块，跨页段落可能被拆开。当前 PDF 文本提取不会可靠地识别章节标题层级；Markdown 格式标题可以保留层级。

上传时先记录 `indexing` 状态，再将向量幂等写入 Chroma，完成后标记为 `ready`。删除时先标记 `deleting` 并从检索结果隐藏，再删除对应用户的向量和 SQLite 记录。中途中断的索引在下次检索时重试，未完成删除在下次访问论文列表时重试。历史 Chroma 目录不可用时删除会保留待处理状态，需要恢复该目录后重试。

更换 embedding 模型名称或 Chroma 目录后，现有 chunk 会在下次检索时重新向量化，旧模型和旧目录的索引会保留到论文被删除。若在原路径覆盖模型权重，系统无法识别权重已更改，应使用新路径。`PAPER_CORPUS_PATH` 指向的旧共享 JSON 保持文件管理方式，其向量也在 Chroma 中按内容版本持久化；修改共享 JSON 会产生新的索引集合，旧集合不会自动清除。内嵌 Chroma 面向本机单进程部署，多进程或多机部署需要单独验证和配置。

## 慢查询排查

论文数量查询可使用 `list_corpus_papers` 直接读取元数据；普通语义检索需要模型将查询编码、执行 SQLite FTS5 与 Chroma 召回，并对候选分块重排。计划生成、Worker 决策和最终回答还会产生模型请求。页面“查看步骤耗时”可区分等待模型、索引重建、关键词检索、向量检索及重排。首次导入或旧数据迁移需要加载模型并计算向量，后续检索复用已持久化的向量；新增论文只需索引新增分块。FTS5 使用项目现有的中文二字词项，检索质量需要在实际标注集上评估。

每个工作流最多执行 100 次工具调用，任一同名工具最多 50 次，失败调用也计数；计数跨子任务和暂停、恢复保留，达到上限的请求不会执行。工具的内部 HTTP 重试不作为新的工具调用计数。每个子任务最多进行 101 次模型决策，以便在 100 次工具调用后仍能总结。前端显示任务停止原因及失败工具的错误。arXiv 只读请求对 429、指定的 5xx、网络和超时错误最多尝试 3 次，采用指数退避加随机抖动；`Retry-After` 超过 60 秒时停止重试并返回错误。该重试机制不提供跨进程限速，批量导入仍应控制请求频率。原始通用单 Agent CLI 入口（包括 demo 和持久会话入口）已停用；Web 多 Agent 和独立论文问答 CLI 保留。

达到检索预算时系统会尝试让模型基于已有证据给出部分结果，并标明工具预算耗尽；如果模型无法遵守无工具调用要求或模型请求失败，工作流仍会失败。正常模式可使用 `search_arxiv` 在线检索（可指定 `start` 和 `max_results`，上限 50），arXiv 请求在同一机器、同一操作系统用户且共享临时目录的进程之间串行；任一请求结束至少 3 秒后才开始下一次请求；search_arxiv 搜索调用另外默认在前一次搜索结束后等待至少 10 秒，重试也遵守这两项间隔。搜索等待期间其他 arXiv 请求仍可在满足 3 秒限制后执行。搜索间隔可在配置文件设置 arxiv_search_interval_seconds，或使用环境变量 PAPER_ARXIV_SEARCH_INTERVAL，数值须为不小于 3 的有限秒数。多机、不同用户或不共享临时目录的容器需要额外协调。测评模式仍禁用在线搜索和导入，以保持本地语料测试可比较。

Web 会话最后一次访问或写入后在 Redis 保留 24 小时；完成一轮后若消息数超过 12，则把旧消息与已有摘要压缩为 SQLite 摘要，并保留最近 8 条消息（约 4 轮），失败时保留原消息。Supervisor 规划时读取摘要与最近 10 条消息（包括当前提问）；新 Worker 会接收摘要和最多 8 条之前的消息及依赖任务证据。当前没有给 Web 工作流设置统一的上下文 token 上限或自动 token 预算，模型服务仍可能拒绝过长上下文；任务状态、调用记录和工作流是结构化保存，但没有独立的结构化业务实体记忆库。

工作流的每次状态更新也会保存到 SQLite 数据库所在目录的 `workflow_results/`，每个工作流一个 JSON 文件。文件包含用户及会话 ID、问题、任务状态、模型消息、检索查询参数、工具返回片段、耗时、错误和最终结果。已有 SQLite 完成记录在会话存储初始化时补导出，未归档的历史任务无法补导。文件含研究内容，请妥善保管。直接被拒绝且未创建工作流的请求，其 HTTP 错误详情仍需从浏览器网络响应或日志查看。

按年份列举或统计本地论文时，检索 worker 可调用 list_corpus_papers 并传入 year，SQLite 按记录的年份筛选，无需加载语义检索模型。返回的 unknown_year_count 表示年份未知的论文数；arXiv 导入的年份是首次公开年份，并非必然等同正式发表年份。模型负责选择工具，具体执行仍取决于其工具调用。共享 JSON 索引的年份需要保存在每个论文分块的 year 字段中，缺失时计为未知。

`retrieve_chunks` 的独立等待时限默认为 180 秒，可在配置文件设置 `retrieval_timeout_seconds` 或设置环境变量 `PAPER_RETRIEVAL_TIMEOUT`，必须为正且有限的秒数。超时会记录工具失败，并向模型说明结果未知。正在运行的模型推理或索引恢复可能继续在后台执行，同一服务进程内同一用户的后续检索会被拒绝，直到该调用结束；多进程之间没有此保护。超时不是强制中断，后台索引恢复仍可能写入数据。

其他工具方面，arXiv 元数据请求的连接和读取等网络超时默认为 20 秒，可用配置 arxiv_api_timeout_seconds 或环境变量 PAPER_ARXIV_API_TIMEOUT 调整，PDF 下载的网络超时为 60 秒，并可能重试和等待限速；导入还包含无独立总时限的 PDF 解析及索引写入。论文清单使用 SQLite，数据库锁等待为 30 秒，清理待删除条目可能访问向量库。指标计算没有独立时限。模型调用的 `llm_timeout_seconds` 与检索超时分别配置。

工作流执行期间会定期续期 Redis 会话锁，以支持较长的任务；Redis 不可用或锁丢失时，后续保存会失败。

## Paperlist 在线搜索

在研究问题中提出在线搜索，确认计划后工作流使用 search_online：默认先搜索 Paperlist，发生服务错误或没有匹配结果时转用 arXiv，并记录来源与切换原因。明确要求 arXiv 时直接使用 arXiv；Paperlist 的分页游标只用于继续查询 Paperlist。Paperlist 覆盖其收录的 AI 会议论文，结果范围与 arXiv 不同。可在环境变量 PAPERLIST_API_KEY 或本地 paperagent.config.json 的 paperlist_api_key 中配置密钥，修改后重启。密钥应通过环境设置或被忽略的本地配置填写，切勿写入版本库。Paperlist ID 与 arXiv ID 不同；明确请求导入时，arXiv 导入工具使用 arXiv ID。测评的本地模式不会启用在线搜索工具。意图识别依赖模型生成计划，请在执行前核对计划。

明确要求保存 Paperlist 结果时，工作流可用 import_paperlist 按原始 Paperlist ID 使用已保存的搜索结果入库，无需再次联网；用户明确指定而会话没有对应结果的 ID 会通过 Paperlist 查询；只能导入本轮或本会话最近一次在线搜索返回的 ID，或用户明确写出的 ID。跨轮保存时可通过 list_paperlist_search_results 分页读取上一轮已完成搜索的 ID。摘要存在时建立摘要分块和向量索引，内容标记为 abstract；缺摘要时保存题录并标记为 metadata_only，不启动向量索引。这两种情况都没有导入 PDF 全文，页数保持未知；如需全文可上传 PDF。相同用户重复导入同一 Paperlist ID 不会新增记录，后续摘要可用于升级仅题录记录。Paperlist 条目与另行上传的 PDF 暂不跨来源自动去重。模型或索引加载失败时，题录已保存，可以按相同 ID 重试摘要索引。
