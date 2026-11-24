"""模型输出净化（剥离推理内容）。

部分 OpenAI 兼容服务（MiniMax、DeepSeek-R1、Qwen3 等）**不返回**独立的
``reasoning_content`` 字段，而是把推理过程**内联**进 ``content``，用
``<think>...</think>`` 包起来。AgentScope 的 OpenAI 适配器拿不到独立通道，
这段文字就以普通文本块的增量事件（``TextBlockDeltaEvent``）到达 Worker。

若原样透传，学生读到的第一段内容会是模型的内心独白（英文），真正的回复
排在后面；高危轮更糟——安全回应本身很短，推理部分可能占掉大半篇幅。

本模块提供两把刷子：

* :class:`ThinkFilter` —— **流式**过滤器。推理标签可能跨增量块断开
  （``<thi`` + ``nk>``），逐块做正则替换一定会漏，所以按状态机处理；
* :func:`strip_think` —— **一次性**过滤，用于已经拿到完整文本的场景
  （高危安全回应走的是 ``agent.reply()``，拿到的是整段文本）。

两种过滤都是**保守**的：只认下面列出的标签，不会误伤正文里普通的尖括号。
"""

from __future__ import annotations

#: 推理块的开始标签。
OPEN_TAGS: tuple[str, ...] = ("<think>", "<thinking>", "<reasoning>")

#: 推理块的结束标签。
CLOSE_TAGS: tuple[str, ...] = ("</think>", "</thinking>", "</reasoning>")

#: 全部需要被剥离的标签。
ALL_TAGS: tuple[str, ...] = OPEN_TAGS + CLOSE_TAGS


def _find_next_tag(text: str, tags: tuple[str, ...]) -> tuple[int, str] | None:
    """找出 ``text`` 里最早出现的标签，返回 ``(位置, 标签)``。"""
    best: tuple[int, str] | None = None
    for tag in tags:
        idx = text.find(tag)
        if idx >= 0 and (best is None or idx < best[0]):
            best = (idx, tag)
    return best


def _hold_back_len(text: str, tags: tuple[str, ...]) -> int:
    """返回需要暂扣的尾部长度。

    暂扣是为了识别**跨块断开**的标签：如果本块结尾是 ``<thi``，下一块可能
    是 ``nk>``，那就是一个完整的 ``<think>``，必须当作标签处理而不是正文。
    """
    best = 0
    for tag in tags:
        for size in range(min(len(tag) - 1, len(text)), 0, -1):
            if size > best and text.endswith(tag[:size]):
                best = size
    return best


class ThinkFilter:
    """流式推理内容过滤器。

    用法::

        filt = ThinkFilter()
        for chunk in stream:
            clean = filt.feed(chunk)
            if clean:
                emit(clean)
        tail = filt.flush()
        if tail:
            emit(tail)
    """

    def __init__(self) -> None:
        self._in_think = False
        self._pending = ""

    # ------------------------------------------------------------------
    @property
    def in_think(self) -> bool:
        """当前是否处在推理块内部（未闭合）。"""
        return self._in_think

    # ------------------------------------------------------------------
    def feed(self, chunk: str) -> str:
        """喂入一个增量块，返回可以安全展示给学生的文本。"""
        if not chunk:
            return ""
        data = self._pending + chunk
        self._pending = ""
        out: list[str] = []

        while data:
            if self._in_think:
                # 推理块内部：只找结束标签，其余一律丢弃
                found = _find_next_tag(data, CLOSE_TAGS)
                if found is None:
                    keep = _hold_back_len(data, CLOSE_TAGS)
                    self._pending = data[len(data) - keep :] if keep else ""
                    data = ""
                    break
                idx, _tag = found
                data = data[idx + len(_tag) :]
                self._in_think = False
                continue

            found = _find_next_tag(data, ALL_TAGS)
            if found is None:
                # 整段安全：暂扣可能构成标签前缀的尾巴，其余放行
                keep = _hold_back_len(data, ALL_TAGS)
                if keep:
                    self._pending = data[len(data) - keep :]
                    out.append(data[: len(data) - keep])
                else:
                    out.append(data)
                data = ""
                break

            idx, tag = found
            out.append(data[:idx])
            data = data[idx + len(tag) :]
            if tag in OPEN_TAGS:
                self._in_think = True
            # 落单的结束标签（如正文里孤立的 </think>）直接丢弃

        return "".join(out)

    # ------------------------------------------------------------------
    def flush(self) -> str:
        """流结束时收尾：吐出暂扣的尾巴（若仍在推理块内则丢弃）。"""
        tail = self._pending
        self._pending = ""
        if self._in_think:
            # 推理块没有闭合 → 剩下的内容整体属于推理，丢弃
            return ""
        return tail


def strip_think(text: str) -> str:
    """一次性剥离文本里的推理内容。

    用于已经拿到完整字符串的场景（例如 ``agent.reply()`` 返回的消息体）。
    未闭合的推理块会把它后面的内容一并丢弃——宁可少给，也不能把推理
    当成回复发出去。
    """
    if not text:
        return ""
    out: list[str] = []
    rest = text
    while True:
        found = _find_next_tag(rest, ALL_TAGS)
        if found is None:
            out.append(rest)
            break
        idx, tag = found
        out.append(rest[:idx])
        rest = rest[idx + len(tag) :]
        if tag in OPEN_TAGS:
            close = _find_next_tag(rest, CLOSE_TAGS)
            if close is None:
                # 未闭合：剩余内容全是推理
                rest = ""
                break
            c_idx, c_tag = close
            rest = rest[c_idx + len(c_tag) :]
    return "".join(out)


__all__ = [
    "ALL_TAGS",
    "CLOSE_TAGS",
    "OPEN_TAGS",
    "ThinkFilter",
    "strip_think",
]
