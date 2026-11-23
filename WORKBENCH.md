# 论文研究工作台

前端使用 Vue 3、TypeScript 和 Vite，后端使用 FastAPI。模型和工具调用沿用项目自定义接口，OpenAI 和 DeepSeek 使用兼容的聊天接口。

在项目根目录执行 `pip install -e '.[web,papers]'`。配置 Redis 的 `REDIS_URL`，设置 `CORECODER_JWT_KEY`、`CORECODER_JWT_ISSUER`、`CORECODER_JWT_AUDIENCE`，并由现有身份服务签发包含 `sub`、`iss`、`aud`、`iat` 和 `exp` 的 HS256 JWT。摘要数据库由 `CORECODER_SQLITE_PATH` 指定。设置 `OPENAI_API_KEY` 或 `DEEPSEEK_API_KEY`，可通过 `OPENAI_MODEL`、`DEEPSEEK_MODEL`、`OPENAI_BASE_URL`、`DEEPSEEK_BASE_URL` 覆盖默认设置。

登录后可在“论文语料库”上传 PDF，系统提取文本并分块入库。也可搜索 arXiv，选择论文导入，或在研究任务中要求检索和导入 arXiv 论文。用户导入的论文按 JWT 用户 ID 隔离，存于 `CORECODER_SQLITE_PATH`。也可按照 [论文语料说明](PAPER_QA.md) 创建共享 JSON 索引，并将 `PAPER_CORPUS_PATH` 指向该文件。首次使用会初始化嵌入模型和交叉编码器，可用 `PAPER_EMBED_MODEL`、`PAPER_RERANK_MODEL` 指向本地模型。

运行 `uvicorn corecoder.web:app --host 127.0.0.1 --port 8000`。在 `web` 目录运行 `npm ci` 和 `npm run dev`。页面输入 JWT 和会话 ID，生成计划并确认后执行。可用同一会话 ID 加载任务状态，已完成后可以提交修改要求。

Supervisor 根据问题生成依赖有序的任务，五类 worker 分别对应数据接入、推理分析、文献检索、生成交付和测评运维。文献检索 worker 支持本地检索及 arXiv 在线检索，数据接入 worker 支持从 arXiv 导入 PDF，测评 worker 支持标注指标计算。其他角色由模型进行文本分析，尚未接入图表视觉识别、代码执行、文件导出或运维部署工具。任务状态和工具调用结果存储在 Redis，最后一次读取或写入 24 小时后过期，会话较长时将旧对话摘要保存到 SQLite。Redis 会话过期后 SQLite 摘要不会自动删除，且不能通过已过期的会话自动恢复。计划生成后等待用户确认；确认后页面显示提交中，并从后端同步工作流及子任务状态，连接失败时显示同步错误。会话 ID 在当前浏览器标签页中保留，刷新后可继续加载。执行过程中可请求暂停或取消，系统在当前模型或工具调用完成后处理请求。已保存的工具回执会在恢复时使用，结果不明的工具调用不会自动重复执行。

执行中的暂停与取消为协作式操作，不能强制中断正在进行的模型请求或工具调用。子任务使用受限工具列表，但尚未提供操作系统级沙箱或独立容器。模型输出及引用仍需核验。前后端和 Redis 的访问控制、传输加密和部署配置需要根据实际运行环境设置。

PDF 提取使用 pypdf，只支持含文本层的 PDF，扫描件需要先 OCR。标题、作者和 DOI 尝试从 PDF 元数据及文本提取，缺失或错误时需核对。在线搜索与下载需要访问 `export.arxiv.org` 和 `arxiv.org`；模型需要从 Hugging Face 下载或配置本地路径。导入时保留提取的正文分块及元数据，不保存原始 PDF。检索时每次重新计算语料向量，语料较大时速度较慢。

## Windows CMD 一体化启动

在项目根目录创建并激活 Python 虚拟环境，然后执行一次依赖安装。需要事先安装 Python 3.10 以上及 Node.js/npm。

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
python run_paperagent.py --setup
```

执行 `python run_paperagent.py`。首次运行会创建 `paperagent.config.json` 并提示编辑该文件。填写 `deepseek_api_key` 或 `openai_api_key`，以及 `embedding_model`、`reranker_model` 的本地路径或 Hugging Face 名称。JSON 中 Windows 路径使用正斜杠（例如 `D:/models/bge-m3`）或双反斜杠。配置文件包含密钥，已加入 `.gitignore`，请勿上传或分享。

再次执行 `python run_paperagent.py`。脚本读取配置，生成本机 JWT，启动后端和前端，打开一次性本地登录链接并自动填入令牌。若浏览器未自动打开，可复制终端中的一次性链接；链接有效期为 10 分钟且只能使用一次。也可将终端中的备用令牌手动粘贴到“访问令牌”。模型提供商可在网页选择。Ctrl+C 停止脚本启动的服务。令牌有效期为 24 小时；每次启动默认生成新的签名密钥和令牌。

脚本使用已有 Redis。如果默认地址的 Redis 未运行且 Docker 可用，会启动一个本地 Redis 容器并在退出时停止，数据保存在 Docker 卷 `paperagent-redis-data`。使用其他 Redis 地址时需要自行启动该服务。执行 `python run_paperagent.py --check` 可运行 Python 测试和前端构建。Linux 或 macOS 也可使用此脚本，虚拟环境使用 `source .venv/bin/activate` 激活。

`chunk_size`、`chunk_overlap`、`rerank_threshold`、`retrieval_limit` 和 `retrieval_candidates` 可在配置中调整。环境变量优先于 JSON；更改后需重启服务。新参数作用于新导入的论文，已有分块不会自动重建。原始 PDF 未存储，相同文件重复上传会被去重；如要重新分块，可使用新 SQLite 文件并重新导入论文。JSON 文件建库需重新执行 `corecoder --index-papers`。
