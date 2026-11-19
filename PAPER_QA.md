# 本地论文问答

安装 `pip install -e '.[papers]'`。准备 UTF-8 JSON 数组，每项包含 `title`、`authors`、`source`、`doi` 和 `text`。`text` 支持 Markdown 标题作为章节边界，正文会按自然段和句子分割并保留窗口重叠。可选 `id` 用于标识论文。请自行提供有权使用的论文正文。

```json
[{"title":"论文标题","authors":["作者"],"source":"期刊或会议","doi":"10.example/example","text":"# 摘要\n论文正文"}]
```

执行 `corecoder --index-papers papers.json paper-index.json` 生成本地分块语料。配置现有项目支持的模型凭据后，执行 `corecoder --paper-corpus paper-index.json`，或使用 `-p '问题'` 单次提问。首次使用会加载 sentence-transformers 的嵌入和交叉编码器模型。可通过 `PAPER_EMBED_MODEL` 和 `PAPER_RERANK_MODEL` 指定本地模型目录或兼容的模型名称。

交互模式中 `/trace` 显示每轮各阶段输入、输出、耗时、状态和检索调用，`/memory` 显示压缩摘要与最近对话，`/reset` 清空当前会话。对话和追踪目前保存在进程内。检索通过 BM25 和向量相似度分别排序，使用 RRF 融合、去重，再由交叉编码器重排。生成回答时使用检索结果中的 `[P1]` 等标记，并附上元数据。分块和向量均在本地处理，问答内容和证据会发送到配置的模型服务。

`corecoder.papers.evaluation.evaluate` 接收标注结果列表，每项包含 `intent`、`expected_intent`、`query`、`expected_query`、`relevant_ids` 和 `retrieved_ids`，计算意图准确率、改写严格匹配率和有相关文档标注样本上的平均召回率。改写严格匹配仅衡量文本完全一致，需结合人工判断语义正确性。相关 ID 应与检索结果的分块 `id` 一致。
