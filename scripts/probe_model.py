"""模型连通性自检 / 自动发现可用端点。

用法::

    # 1) 用 .env 里的配置自检（会打印服务端原始报错）
    ./.venv/Scripts/python.exe scripts/probe_model.py

    # 2) 拿一把 key 自动扫一遍常见的 Coding Plan / OpenAI 兼容地址，
    #    找到能用的那个，并把它写回 .env
    ./.venv/Scripts/python.exe scripts/probe_model.py --key sk-xxx --scan --save

    # 3) 指定地址自检
    ./.venv/Scripts/python.exe scripts/probe_model.py --base-url https://api.minimax.io/v1 --model MiniMax-M2.7

它会分别试 OpenAI 兼容与 Anthropic 兼容两种协议，并打印服务端返回的**原始错误信息**，
方便判断是"地址不对"还是"密钥不对"。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402

#: 常见的 Coding Plan / OpenAI 兼容端点（按平台排列）
CANDIDATE_ENDPOINTS: list[tuple[str, str, str, str]] = [
    # (显示名, base_url, 协议, 模型名)
    ("MiniMax 国内 · OpenAI", "https://api.minimaxi.com/v1", "openai", "MiniMax-M2.7"),
    ("MiniMax 国内 · Anthropic", "https://api.minimaxi.com/anthropic", "anthropic", "MiniMax-M2.7"),
    ("MiniMax 国际 · OpenAI", "https://api.minimax.io/v1", "openai", "MiniMax-M2.7"),
    ("MiniMax 国际 · Anthropic", "https://api.minimax.io/anthropic", "anthropic", "MiniMax-M2.7"),
    ("无问芯穹 · Coding Plan", "https://cloud.infini-ai.com/maas/coding/v1", "openai", "glm-5"),
    ("无问芯穹 · 通用 MaaS", "https://cloud.infini-ai.com/maas/v1", "openai", "glm-5"),
    ("阿里云百炼 · 通用", "https://dashscope.aliyuncs.com/compatible-mode/v1", "openai", "qwen-plus"),
    ("阿里云百炼 · Coding Plan", "https://coding-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1", "openai", "qwen-plus"),
    ("火山方舟 · 通用", "https://ark.cn-beijing.volces.com/api/v3", "openai", "doubao-seed-1.6"),
    ("DeepSeek", "https://api.deepseek.com/v1", "openai", "deepseek-chat"),
    ("OpenRouter", "https://openrouter.ai/api/v1", "openai", "openai/gpt-4o-mini"),
]


async def _probe_openai(base_url: str, api_key: str, model: str) -> tuple[int, str]:
    import httpx  # noqa: PLC0415

    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "回复两个字：你好"}],
        "max_tokens": 16,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(url, json=body, headers=headers)
        return resp.status_code, resp.text[:300]
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


async def _probe_anthropic(base_url: str, api_key: str, model: str) -> tuple[int, str]:
    import httpx  # noqa: PLC0415

    url = base_url.rstrip("/") + "/v1/messages"
    body = {
        "model": model,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "回复两个字：你好"}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            resp = await client.post(url, json=body, headers=headers)
        return resp.status_code, resp.text[:300]
    except Exception as exc:  # noqa: BLE001
        return 0, f"{type(exc).__name__}: {exc}"


async def _list_models(base_url: str, api_key: str, protocol: str) -> list[str]:
    import httpx  # noqa: PLC0415

    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    if protocol == "anthropic":
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    url = base_url.rstrip("/") + ("/v1/models" if protocol == "anthropic" else "/models")
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code >= 300:
            return []
        data = resp.json()
        items = data.get("data") or data.get("models") or []
        return [str(i.get("id")) for i in items if isinstance(i, dict)]
    except Exception:  # noqa: BLE001
        return []


def _write_env(base_url: str, api_key: str, model: str) -> None:
    """把可用配置写回 .env。"""
    env = ROOT / ".env"
    text = env.read_text(encoding="utf-8") if env.exists() else (ROOT / ".env.example").read_text(encoding="utf-8")

    def set_kv(t: str, k: str, v: str) -> str:
        pat = re.compile(rf"^{re.escape(k)}=.*$", re.M)
        return pat.sub(f"{k}={v}", t) if pat.search(t) else t + f"\n{k}={v}\n"

    text = set_kv(text, "MODEL_PROVIDER", "openai")
    text = set_kv(text, "MODEL_NAME", model)
    text = set_kv(text, "MODEL_BASE_URL", base_url)
    text = set_kv(text, "MODEL_API_KEY", api_key)
    text = set_kv(text, "RISK_MODEL_NAME", model)
    text = set_kv(text, "RISK_MODEL_BASE_URL", base_url)
    text = set_kv(text, "RISK_MODEL_API_KEY", api_key)
    env.write_text(text, encoding="utf-8")
    print(f"\n已写入 {env.name}：MODEL_BASE_URL={base_url}  MODEL_NAME={model}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="模型连通性自检 / 端点自动发现")
    parser.add_argument("--base-url", default="", help="指定 OpenAI 兼容 base url")
    parser.add_argument("--anthropic-base-url", default="", help="指定 Anthropic 兼容 base url")
    parser.add_argument("--api-key", default="", help="指定 API Key")
    parser.add_argument("--model", default="", help="指定模型名")
    parser.add_argument("--scan", action="store_true", help="自动扫一遍常见端点")
    parser.add_argument("--save", action="store_true", help="找到可用端点后写回 .env")
    parser.add_argument("--list-models", action="store_true", help="列出该端点可用模型")
    args = parser.parse_args()

    settings = get_settings()
    api_key = args.api_key or settings.model_api_key
    if not api_key:
        print("没有 API Key：请设置 MODEL_API_KEY，或用 --api-key 传入")
        return 2

    print(f"key = {api_key[:8]}...{api_key[-4:]} (len={len(api_key)})")
    print("=" * 78)

    # ---------------- 指定地址自检 ----------------
    if args.base_url or args.anthropic_base_url:
        base = args.base_url or settings.model_base_url
        model = args.model or settings.model_name
        code, text = await _probe_openai(base, api_key, model)
        print(f"[OpenAI 兼容] {code}  {text}")
        if code == 0 or code >= 400:
            code2, text2 = await _probe_anthropic(
                args.anthropic_base_url or base, api_key, model
            )
            print(f"[Anthropic 兼容] {code2}  {text2}")
            if code2 == 0 or code2 >= 400:
                print("\n两种协议都失败。最常见的三种原因：")
                print("  1) Base URL 用错了（Coding Plan 专用地址通常含 /coding/ 路径，与通用地址不互通）")
                print("  2) 密钥已过期 / 订阅未生效 / 用错了账号的密钥")
                print("  3) 模型名与平台文档不一致（区分大小写）")
                return 1
        print("\n连通性 OK")
        if args.list_models:
            models = await _list_models(base, api_key, "openai")
            if models:
                print("\n可用模型：")
                for m in models[:40]:
                    print("   ", m)
        return 0

    # ---------------- 自动扫描 ----------------
    if args.scan:
        print("扫描常见端点……\n")
        found = None
        for label, base, protocol, model in CANDIDATE_ENDPOINTS:
            if protocol == "openai":
                code, text = await _probe_openai(base, api_key, model)
            else:
                code, text = await _probe_anthropic(base, api_key, model)
            ok = 200 <= code < 300
            flag = "✅ 可用" if ok else "✗"
            print(f"  {flag} {label:26s} {base}")
            print(f"      {code}  {text[:150]}")
            if ok and found is None:
                found = (label, base, model)
        print("=" * 78)
        if found is None:
            print("没有端点接受这把密钥。最可能是：订阅未生效 / 密钥已过期或被撤销 /")
            print("这把是别的账号的密钥。请到平台控制台确认 Coding Plan 是否已激活，")
            print("并重新生成一把密钥。")
            return 1
        label, base, model = found
        print(f"找到可用端点：{label}  {base}  模型 {model}")
        if args.save:
            _write_env(base, api_key, model)
        return 0

    # 默认：用 .env 里的配置自检
    base = settings.model_base_url
    model = settings.model_name
    print(f"base_url = {base}")
    print(f"model    = {model}")
    print("-" * 78)
    code, text = await _probe_openai(base, api_key, model)
    print(f"[OpenAI 兼容] {code}  {text}")
    if code == 0 or code >= 400:
        code2, text2 = await _probe_anthropic(base, api_key, model)
        print(f"[Anthropic 兼容] {code2}  {text2}")
        if code2 == 0 or code2 >= 400:
            print("-" * 78)
            print("两种协议都失败。跑一次自动扫描看看别的地址：")
            print("  ./.venv/Scripts/python.exe scripts/probe_model.py --scan --save")
            return 1
    print("-" * 78)
    print("连通性 OK")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
