"""电商 NER 实体抽取 Agent。

从用户查询中抽取实体：
    HPPX=品牌, HCCX=商品, XH=款式, MISC=规格/其他

用法:
    agent = NerAgent(model_path="models/ner/best_model")
    agent.extract("红色韩版连衣裙") -> {"品牌": [], "商品": ["连衣裙"], "款式": ["红色", "韩版"], "规格": []}
"""

from __future__ import annotations

import os
import re
from typing import Dict, List

import torch

from agents.tools.base import BaseAgentTool

ENTITY_NAMES = {"HPPX": "品牌", "HCCX": "商品", "XH": "款式", "MISC": "规格"}


class NerAgent(BaseAgentTool):
    """基于 BERT 的电商查询实体抽取（懒加载 + 进程内缓存复用）。"""

    name: str = "ner_agent"
    description: str = "从商品查询中抽取品牌、商品、款式、规格等实体"

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self._tokenizer = None
        self._model = None
        self._labels = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_loaded(self):
        if self._model is not None:
            return
        if not os.path.isdir(self.model_path):
            print(f"[NerAgent] 模型路径不存在: {self.model_path}，实体抽取不可用")
            return
        # 统一走 models.persistence 的进程内缓存（线程安全 + 避免多实例重复加载）
        from models.persistence import load_token_classification_model, validate_model_dir

        missing = validate_model_dir(self.model_path)
        if missing:
            print(f"[NerAgent] 模型目录不完整，缺失: {missing}")
            return
        print(f"[NerAgent] 加载 NER 模型: {self.model_path}（缓存复用）")
        self._tokenizer, self._model, self._labels, self._device = (
            load_token_classification_model(self.model_path)
        )
        print(f"[NerAgent] NER 模型加载完成，{len(self._labels or [])} 个标签")

    def extract(self, query: str) -> Dict[str, List[str]]:
        """抽取实体 -> {"品牌": [...], "商品": [...], "款式": [...], "规格": [...]}"""
        result = {name: [] for name in ENTITY_NAMES.values()}
        query = (query or "").strip()
        if not query or len(query) > 200:
            return result
        # 无意义输入守卫：无中文且无字母数字（纯 emoji/符号）
        if not re.search(r"[\u4e00-\u9fa5A-Za-z0-9]", query):
            return result
        self._ensure_loaded()
        if self._model is None:
            return result

        encoding = self._tokenizer(
            query,
            max_length=128,
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = encoding.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self._device) for k, v in encoding.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        pred_ids = logits[0].argmax(-1).tolist()

        # 用 offset_mapping 还原实体 span（兼容 BIO / BIOES 两种标签方案）
        text = query
        spans: List[tuple] = []  # (type_key, start, end)

        def flush(cur_type, cur_start, cur_end):
            if cur_type is not None:
                spans.append((ENTITY_NAMES.get(cur_type, "规格"), cur_start, cur_end))

        cur_type, cur_start = None, None
        for (start, end), pid in zip(offsets, pred_ids):
            if start == end:
                continue
            label = self._labels[pid] if pid < len(self._labels) else "O"
            if label == "O":
                flush(cur_type, cur_start, start)
                cur_type, cur_start = None, None
                continue

            if label.startswith("B-"):
                flush(cur_type, cur_start, start)
                cur_type, cur_start = label[2:], start
            elif label.startswith("S-"):
                # 单 token 实体
                flush(cur_type, cur_start, start)
                spans.append((ENTITY_NAMES.get(label[2:], "规格"), start, end))
                cur_type, cur_start = None, None
            elif label.startswith("I-"):
                if cur_type == label[2:]:
                    continue  # 续接当前实体
                # 孤儿 I-（无 B- 起始）：视为新实体起点，避免漏抽
                flush(cur_type, cur_start, start)
                cur_type, cur_start = label[2:], start
            elif label.startswith("E-"):
                if cur_type == label[2:]:
                    spans.append((ENTITY_NAMES.get(cur_type, "规格"), cur_start, end))
                    cur_type, cur_start = None, None
                else:
                    # 无起点的 E-：按单 token 实体处理
                    spans.append((ENTITY_NAMES.get(label[2:], "规格"), start, end))
        flush(cur_type, cur_start, len(text))

        # 合并相邻同类实体：模型常把同一实体拆成连续多个 B- 片段（如「蓝牙」「耳机」），
        # 片段间无间隔时合并回完整实体
        merged: List[tuple] = []
        for span in spans:
            if merged and span[0] == merged[-1][0] and span[1] == merged[-1][2]:
                merged[-1] = (span[0], merged[-1][1], span[2])
            else:
                merged.append(span)

        # 去空格伪影、去重保序
        for k in result:
            seen = set()
            cleaned = []
            for type_key, s, e in merged:
                if type_key != k:
                    continue
                x = text[s:e].strip()
                if x and x not in seen:
                    seen.add(x)
                    cleaned.append(x)
            result[k] = cleaned
        return result

    def run(self, query: str, **kwargs) -> str:
        """对外接口：返回可读的实体列表字符串"""
        entities = self.extract(query)
        parts = [f"{k}: {'、'.join(v) if v else '无'}" for k, v in entities.items()]
        return "；".join(parts)

    async def arun(self, **kwargs) -> str:
        return self.run(**kwargs)
