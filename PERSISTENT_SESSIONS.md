# 持久化会话

安装 `pip install -e '.[persistent]'`，启动可访问的 Redis，并设置 `REDIS_URL`、`CORECODER_JWT_KEY`、`CORECODER_JWT_ISSUER`、`CORECODER_JWT_AUDIENCE` 和 `CORECODER_JWT`，同时按原项目说明配置模型。JWT 由外部身份系统签发，当前入口验证 HS256 签名及 `sub`、`exp`、`iat`、`iss`、`aud`。`sub` 是用户 ID。使用不同用户 ID 的相同会话名会得到独立会话。`CORECODER_SQLITE_PATH` 可指定摘要数据库，默认是当前目录的 `corecoder-memory.sqlite3`。

`corecoder --persistent-session SESSION_ID -p '问题'` 开始或继续会话。`corecoder --persistent-session SESSION_ID --resume-pending` 恢复已有会话。此入口运行原有工具调用循环，Redis 记录活动消息和状态。工具结果作为 tool 消息保存。每次模型输出和工具回执后建立检查点，完成后超出窗口的旧消息压缩为 SQLite 摘要，下一轮将摘要加入模型上下文。

如果程序在工具执行后、回执保存前中断，恢复时该工具的实际结果不可知。系统补写未知状态回执，交由模型和用户判断，避免重复执行。摘要 SQLite 和 Redis 数据需要同时保留。Redis 会话在最后一次读取或写入后的 24 小时过期，读取会续期。过期后无法通过该会话恢复摘要，SQLite 中的摘要记录仍保留。早先创建的无 TTL 会话会在下次访问或写入时获得 TTL。部署方仍需配置 Redis 持久化及备份。Redis、SQLite 不具备跨数据库原子事务，SQLite 中可能存在未引用的摘要快照。

此模式目前仅支持单次 CLI 调用，使用现有工具权限规则。完成检查可通过 `PersistentAgent` 的 `completion_check` 回调提供，回调返回反馈时继续循环。已有子代理隔离各自的消息上下文，文件系统和命令执行仍使用当前进程环境。论文检索模式尚未接入此持久化循环。
