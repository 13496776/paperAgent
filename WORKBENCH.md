# 论文研究工作台

前端使用 Vue 3、TypeScript 和 Vite，后端使用 FastAPI。模型和工具调用沿用项目自定义接口，OpenAI 和 DeepSeek 使用兼容的聊天接口。

在项目根目录执行 `pip install -e '.[web,papers]'`。配置 Redis 的 `REDIS_URL`，设置 `CORECODER_JWT_KEY`、`CORECODER_JWT_ISSUER`、`CORECODER_JWT_AUDIENCE`，并由现有身份服务签发包含 `sub`、`iss`、`aud`、`iat` 和 `exp` 的 HS256 JWT。摘要数据库由 `CORECODER_SQLITE_PATH` 指定。设置 `OPENAI_API_KEY` 或 `DEEPSEEK_API_KEY`，可通过 `OPENAI_MODEL`、`DEEPSEEK_MODEL`、`OPENAI_BASE_URL`、`DEEPSEEK_BASE_URL` 覆盖默认设置。

登录后可在“论文语料库”上传 PDF，系统提取文本并分块入库。也可搜索 arXiv，选择论文导入，或在研究任务中要求检索和导入 arXiv 论文。用户导入的论文按 JWT 用户 ID 隔离，存于 `CORECODER_SQLITE_PATH`。也可按照 [论文语料说明](PAPER_QA.md) 创建共享 JSON 索引，并将 `PAPER_CORPUS_PATH` 指向该文件。首次使用会初始化嵌入模型和交叉编码器，可用 `PAPER_EMBED_MODEL`、`PAPER_RERANK_MODEL` 指向本地模型。

运行 `uvicorn corecoder.web:app --host 127.0.0.1 --port 8000`。在 `web` 目录运行 `npm ci` 和 `npm run dev`。页面输入 JWT 和会话 ID，生成计划并确认后执行。可用同一会话 ID 加载任务状态，已完成后可以提交修改要求。

Supervisor 根据问题生成依赖有序的任务，五类 worker 分别对应数据接入、推理分析、文献检索、生成交付和测评运维。文献检索 worker 支持本地检索及 arXiv 在线检索，数据接入 worker 支持从 arXiv 导入 PDF，测评 worker 支持标注指标计算。其他角色由模型进行文本分析，尚未接入图表视觉识别、代码执行、文件导出或运维部署工具。任务状态和工具调用结果存储在 Redis，最后一次读取或写入 24 小时后过期，会话较长时将旧对话摘要保存到 SQLite。Redis 会话过期后 SQLite 摘要不会自动删除，且不能通过已过期的会话自动恢复。工具调用执行结果不明时，当前系统只能标记中断并重新执行该任务，当前注册工具仅执行读取和计算。

当前暂停点位于计划确认阶段。现有执行循环为同步请求，执行中的主动暂停尚未接入。子任务使用受限工具列表，但尚未提供操作系统级沙箱或独立容器。模型输出及引用仍需核验。前后端和 Redis 的访问控制、传输加密和部署配置需要根据实际运行环境设置。

PDF 提取使用 pypdf，只支持含文本层的 PDF，扫描件需要先 OCR。标题、作者和 DOI 尝试从 PDF 元数据及文本提取，缺失或错误时需核对。在线搜索与下载需要访问 `export.arxiv.org` 和 `arxiv.org`；模型需要从 Hugging Face 下载或配置本地路径。导入时保留提取的正文分块及元数据，不保存原始 PDF。检索时每次重新计算语料向量，语料较大时速度较慢。
