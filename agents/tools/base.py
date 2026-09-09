"""
共享工具基类
所有子 Agent 和可复用工具继承此类，统一接口规范
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgentTool(ABC):
    """
    所有工具和子 Agent 的基类

    设计理念：
    - 每个 Agent / Tool 都有 name 和 description
    - 必须实现 run() 同步执行 和 arun() 异步执行
    - get_input_schema() 返回工具描述供 LM 路由使用
    """

    name: str = ""
    description: str = ""

    def __init__(self, name: str = "", description: str = ""):
        if name:
            self.name = name
        if description:
            self.description = description

    @abstractmethod
    def run(self, **kwargs) -> str:
        """同步执行入口"""
        ...

    @abstractmethod
    async def arun(self, **kwargs) -> str:
        """异步执行入口"""
        ...

    def get_input_schema(self) -> Dict[str, Any]:
        """返回工具的输入 schema，供 Router/LLM 参考"""
        return {
            "name": self.name,
            "description": self.description,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
