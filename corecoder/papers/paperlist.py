"""Search the Paperlist MCP service for conference paper metadata."""
import json
import os
import time
from urllib.parse import urlsplit

import httpx

PAPERLIST_MCP_URL = "https://paperlist.ai/mcp"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024


def _rpc(client, headers, method, params, request_id=None):
    message = {"jsonrpc": "2.0", "method": method, "params": params}
    if request_id is not None:
        message["id"] = request_id
    with client.stream("POST", PAPERLIST_MCP_URL, json=message, headers=headers, timeout=30, follow_redirects=False) as response:
        if response.status_code in (401, 403):
            raise RuntimeError("Paperlist 认证或访问被拒绝，请检查 API Key 和网络权限")
        response.raise_for_status()
        session = response.headers.get("Mcp-Session-Id")
        if session:
            headers["Mcp-Session-Id"] = session
        if request_id is None:
            return {}
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        def parse(payload):
            try:
                reply = json.loads(payload)
            except ValueError as exc:
                raise RuntimeError("Paperlist MCP 返回了无效 JSON") from exc
            if not isinstance(reply, dict) or reply.get("id") != request_id:
                return None
            if "error" in reply:
                raise RuntimeError("Paperlist MCP 请求失败，请检查认证或服务状态")
            if reply.get("jsonrpc") != "2.0" or not isinstance(reply.get("result"), dict):
                raise RuntimeError("Paperlist MCP 响应无效")
            return reply["result"]

        size = 0
        deadline = time.monotonic() + 30
        if content_type == "text/event-stream":
            data = []
            for line in response.iter_lines():
                size += len(line.encode("utf-8")) + 1
                if size > MAX_RESPONSE_BYTES or time.monotonic() > deadline:
                    raise RuntimeError("Paperlist MCP 响应超时或过大")
                if line.startswith("data:"):
                    data.append(line[5:].removeprefix(" "))
                elif not line and data:
                    parsed = parse("\n".join(data))
                    if parsed is not None:
                        return parsed
                    data.clear()
        elif content_type == "application/json":
            chunks = []
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES or time.monotonic() > deadline:
                    raise RuntimeError("Paperlist MCP 响应超时或过大")
                chunks.append(chunk)
            parsed = parse(b"".join(chunks))
            if parsed is not None:
                return parsed
        else:
            raise RuntimeError("Paperlist MCP 响应格式不受支持")
    raise RuntimeError("Paperlist MCP 未返回匹配的响应")


def _call_tool(name, args, client=None):
    key = os.environ.get("PAPERLIST_API_KEY", "").strip()
    if not key:
        raise RuntimeError("请配置 PAPERLIST_API_KEY 后使用 Paperlist")
    headers = {"Authorization": "Bearer " + key, "Accept": "application/json, text/event-stream"}
    owned = client is None
    client = client or httpx.Client()
    try:
        init = _rpc(client, headers, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "paperagent", "version": "1.0"}}, 1)
        version = init.get("protocolVersion")
        if version not in ("2025-06-18", "2025-03-26"):
            raise RuntimeError("Paperlist MCP 协议版本不受支持")
        headers["MCP-Protocol-Version"] = version
        _rpc(client, headers, "notifications/initialized", {})
        result = _rpc(client, headers, "tools/call", {"name": name, "arguments": args}, 2)
        if result.get("isError"):
            raise RuntimeError("Paperlist 请求失败，请检查 API Key 和服务状态")
        data = result.get("structuredContent")
        if not isinstance(data, dict):
            content = result.get("content", [])
            text = next((item.get("text") for item in content if isinstance(item, dict) and item.get("type") == "text"), "")
            try:
                data = json.loads(text)
            except (ValueError, TypeError) as exc:
                raise RuntimeError("Paperlist 返回结果格式无效") from exc
        if not isinstance(data, dict):
            raise RuntimeError("Paperlist 返回结果格式无效")  # noqa: TRY004
        return data
    finally:
        if headers.get("Mcp-Session-Id"):
            try:
                client.delete(PAPERLIST_MCP_URL, headers=headers, timeout=5, follow_redirects=False)
            except httpx.HTTPError:
                pass
        if owned:
            client.close()


def _normalize_paper(paper):
    if not isinstance(paper, dict) or not isinstance(paper.get("id"), str) or not isinstance(paper.get("title"), str) or not paper["title"].strip():
        raise RuntimeError("Paperlist 论文记录格式无效")
    links = paper.get("links") if isinstance(paper.get("links"), dict) else {}
    url = links.get("paperlist") or paper.get("url") or ""
    try:
        parsed = urlsplit(url)
        valid_url = parsed.scheme == "https" and parsed.hostname in ("paperlist.ai", "www.paperlist.ai") and not parsed.username
    except (ValueError, TypeError):
        valid_url = False
    authors = paper.get("authors")
    doi = links.get("doi") or paper.get("doi")
    return {"id": paper["id"], "title": paper["title"],
            "authors": [a for a in authors if isinstance(a, str)] if isinstance(authors, list) else [],
            "abstract": paper.get("abstract") if isinstance(paper.get("abstract"), str) else "",
            "url": url if valid_url else "", "doi": doi if isinstance(doi, str) else "",
            "published_at": paper.get("published_at"), "conference": paper.get("conference")}


def search_paperlist(query, *, cursor=None, client=None):
    if not isinstance(query, str) or not query.strip() or len(query) > 300:
        raise ValueError("query must contain 1–300 characters")
    if cursor is not None and (not isinstance(cursor, str) or not cursor or len(cursor) > 2000):
        raise ValueError("invalid Paperlist cursor")
    args = {"query": query.strip(), "fields": "full", "limit": 10}
    if cursor:
        args["cursor"] = cursor
    data = _call_tool("search_papers", args, client)
    if not isinstance(data.get("papers"), list):
        raise RuntimeError("Paperlist 搜索结果缺少论文列表")  # noqa: TRY004
    next_cursor = data.get("next_cursor")
    note = data.get("note")
    return {"papers": [_normalize_paper(paper) for paper in data["papers"]],
            "next_cursor": next_cursor if isinstance(next_cursor, str) and next_cursor else None,
            "note": note if isinstance(note, str) else ""}


def get_paperlist(ident, *, client=None):
    if not isinstance(ident, str) or not ident.strip() or len(ident) > 200:
        raise ValueError("invalid Paperlist ID")
    data = _call_tool("get_paper", {"id": ident}, client)
    paper = _normalize_paper(data.get("paper", data))
    if paper["id"] != ident:
        raise RuntimeError("Paperlist 返回的论文 ID 与请求不一致")
    return paper
