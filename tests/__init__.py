"""测试包。

这里做一件容易被忽略但代价很高的事：**让评估脚本在 Windows GBK 控制台下不会因为
打印 emoji 而崩**。这些脚本的 emoji 汇总打印位于"评估全部跑完"之后、写报告之前，
一旦抛 UnicodeEncodeError，整轮评估（含 LLM 调用）作废且报告不落盘。
把不可编码字符替换成 "?" 即可，不影响任何指标。
"""

import sys


def _make_console_encoding_safe() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001 - 流不支持 reconfigure 时忽略
            pass


_make_console_encoding_safe()
