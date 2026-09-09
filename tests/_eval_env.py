"""
评估环境引导与前置资产预检。

背景: settings 里模型/索引路径默认是相对路径(如 ./models/best)。
评估脚本若从其他工作目录启动, Agent 会静默降级(规则兜底/空检索)
却照常输出"报告已保存", 导致数字不可信。

本模块提供:
- ensure_project_root(): 把进程工作目录切到项目根目录, 使相对路径默认值生效
- require_assets(): 关键资产缺失时抛错退出, 避免在降级链路上产出误导性报告
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_project_root() -> str:
    """切换到项目根目录, 并把项目根加入 sys.path, 返回项目根路径。"""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)
    return PROJECT_ROOT


def require_assets(*paths: str) -> None:
    """检查关键资产; 缺失时抛 RuntimeError, 评估不应继续。"""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            "评估前置资产缺失, 拒绝继续(避免在降级链路上产出误导性报告):\n  "
            + "\n  ".join(missing)
            + "\n请先运行对应构建脚本(scripts/build_*.py)或修正路径后再执行评估。"
        )


def _extract_usage(result, model_name: str) -> dict:
    """从 LangChain AIMessage 提取 token 用量(兼容 usage_metadata 与 response_metadata)。"""
    prompt_tokens = completion_tokens = 0

    um = getattr(result, "usage_metadata", None)
    if isinstance(um, dict):
        prompt_tokens = int(um.get("input_tokens", 0))
        completion_tokens = int(um.get("output_tokens", 0))

    if prompt_tokens == 0:
        rm = getattr(result, "response_metadata", None) or {}
        tu = rm.get("token_usage") or rm.get("tokenUsage") or {}
        if isinstance(tu, dict):
            prompt_tokens = int(tu.get("prompt_tokens", tu.get("input_tokens", 0)))
            completion_tokens = int(tu.get("completion_tokens", tu.get("output_tokens", 0)))

    return {
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


class TokenRecordingLLM:
    """包装 LLM: 每次 invoke 把 token 用量记入全局 obs。

    评估脚本用它包住生产链路(Agent/编排器)的 LLM, 让报告里的 token_usage
    包含真实链路调用——此前只有 Judge 侧记录, 链路调用全部漏计。
    """

    def __init__(self, llm):
        self._llm = llm

    def invoke(self, input, config=None, **kwargs):
        result = self._llm.invoke(input, **kwargs)
        try:
            usage = _extract_usage(
                result,
                getattr(self._llm, "model_name", "")
                or getattr(self._llm, "model", "")
                or "unknown",
            )
            if usage["prompt_tokens"] or usage["completion_tokens"]:
                from orchestration.observability import obs

                obs.record_tokens(
                    model=usage["model"],
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                )
        except Exception:
            pass
        return result

    def __call__(self, input, **kwargs):
        """让 LCEL 组合(prompt | llm)可用: coerce_to_runnable 会把 callable 包成 RunnableLambda。"""
        return self.invoke(input, **kwargs)

    def __getattr__(self, name):
        return getattr(self._llm, name)
