"""模型持久化与加载管理。

职责：
1. 校验模型目录完整性（权重/分词器/标签/训练配置是否齐全）
2. 描述模型元信息（类别数、标签、训练指标、文件占用）
3. 进程内缓存加载器：模型只加载一次，跨请求/跨 Agent 实例持久复用
4. 训练后固化模型目录（含校验）
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 模型目录必需文件（缺失任一视为不完整）
MODEL_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "labels.txt",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "vocab.txt",
    "training_config.json",
)

MODEL_OPTIONAL_FILES = ("training_args.bin", "pytorch_model.bin", "model.bin")


def validate_model_dir(model_path: str) -> List[str]:
    """校验模型目录，返回缺失的必需文件列表；返回空列表表示完整。"""
    missing = []
    for fname in MODEL_REQUIRED_FILES:
        if not os.path.isfile(os.path.join(model_path, fname)):
            missing.append(fname)
    return missing


def describe_model(model_path: str) -> Dict:
    """读取模型元信息，返回可序列化的字典。"""
    info: Dict = {"model_path": model_path}
    if not os.path.isdir(model_path):
        info["error"] = "目录不存在"
        return info

    info["missing_files"] = validate_model_dir(model_path)
    info["total_bytes"] = sum(
        os.path.getsize(os.path.join(model_path, f))
        for f in os.listdir(model_path)
        if os.path.isfile(os.path.join(model_path, f))
    )

    labels_file = os.path.join(model_path, "labels.txt")
    if os.path.isfile(labels_file):
        with open(labels_file, encoding="utf-8") as f:
            info["labels"] = [line.strip() for line in f if line.strip()]
        info["num_labels"] = len(info["labels"])

    config_file = os.path.join(model_path, "config.json")
    if os.path.isfile(config_file):
        try:
            with open(config_file, encoding="utf-8") as f:
                cfg = json.load(f)
            info["architectures"] = cfg.get("architectures", [])
            info["model_type"] = cfg.get("model_type")
            info["hidden_size"] = cfg.get("hidden_size")
            info["num_hidden_layers"] = cfg.get("num_hidden_layers")
        except Exception as exc:  # noqa: BLE001
            info["config_error"] = str(exc)

    train_config_file = os.path.join(model_path, "training_config.json")
    if os.path.isfile(train_config_file):
        try:
            with open(train_config_file, encoding="utf-8") as f:
                tc = json.load(f)
            info["training"] = {
                "train_samples": tc.get("train_samples"),
                "val_samples": tc.get("val_samples"),
                "epochs": tc.get("epochs"),
                "batch_size": tc.get("batch_size"),
                "learning_rate": tc.get("learning_rate"),
                "max_length": tc.get("max_length"),
                "eval_result": tc.get("eval_result"),
            }
        except Exception as exc:  # noqa: BLE001
            info["training_config_error"] = str(exc)
    return info


class ModelCache:
    """线程安全的进程内模型缓存：同一路径的模型只加载一次，之后持久复用。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[str, Tuple] = {}

    def load(
        self,
        model_path: str,
        labels_path: Optional[str] = None,
        task: str = "classification",
    ) -> Tuple:
        """加载（或返回已缓存的）(tokenizer, model, labels, device)。

        Args:
            model_path: 模型目录
            labels_path: 标签文件路径（默认取模型目录内 labels.txt）
            task: "classification" | "token_classification"
        """
        resolved = str(Path(model_path).resolve())
        key = f"{task}:{resolved}"
        with self._lock:
            if key not in self._cache:
                self._cache[key] = self._load_from_disk(resolved, labels_path, task)
            return self._cache[key]

    @staticmethod
    def _load_from_disk(
        model_path: str,
        labels_path: Optional[str],
        task: str,
    ) -> Tuple:
        import torch

        missing = validate_model_dir(model_path)
        if missing:
            raise FileNotFoundError(f"模型目录不完整，缺失文件: {missing}")

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if task == "token_classification":
            from transformers import AutoModelForTokenClassification

            model = AutoModelForTokenClassification.from_pretrained(model_path)
        else:
            from transformers import AutoModelForSequenceClassification

            model = AutoModelForSequenceClassification.from_pretrained(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = maybe_quantize_model(model, device)
        model.to(device)
        model.eval()

        labels_file = labels_path or os.path.join(model_path, "labels.txt")
        with open(labels_file, encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        return tokenizer, model, labels, device

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# 全局单例：API 服务与各 Agent 共用同一份缓存
MODEL_CACHE = ModelCache()


def load_classification_model(
    model_path: str, labels_path: Optional[str] = None
) -> Tuple:
    """加载分类模型（进程内持久复用）。"""
    return MODEL_CACHE.load(model_path, labels_path)


def load_token_classification_model(
    model_path: str, labels_path: Optional[str] = None
) -> Tuple:
    """加载序列标注（NER）模型（进程内持久复用 + 线程安全）。"""
    return MODEL_CACHE.load(model_path, labels_path, task="token_classification")


def maybe_quantize_model(model, device) -> object:
    """加载期 int8 动态量化（仅 CPU；量化失败时原样返回）。"""
    try:
        from config.settings import settings

        if not settings.model_quantize:
            return model
        if str(device) != "cpu":
            return model
        import torch
        import warnings

        # transformers 5.x 的 BertModel 支持动态量化 Linear 层；
        # 量化后不可再 .to(非cpu)，这里只对 CPU 生效。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[Persistence] int8 量化跳过（保持 fp32）: {exc}")
        return model


def quantize_sentence_transformer(embedder) -> object:
    """对 SentenceTransformer 底层的 transformers 模型做 int8 动态量化（仅 CPU）。"""
    try:
        from config.settings import settings

        if not settings.model_quantize:
            return embedder
        import torch
        import warnings

        container = None
        # 兼容 sentence-transformers 5.x（st[0].auto_model）与 4.x（st.model[0].auto_model）
        try:
            auto_model = embedder[0].auto_model
            container = "new"
        except (AttributeError, IndexError, TypeError):
            try:
                auto_model = embedder.model[0].auto_model
                container = "old"
            except (AttributeError, IndexError, TypeError):
                return embedder
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            quantized = torch.quantization.quantize_dynamic(
                auto_model, {torch.nn.Linear}, dtype=torch.qint8
            )
        if container == "new":
            embedder[0].auto_model = quantized
        else:
            embedder.model[0].auto_model = quantized
        return embedder
    except Exception as exc:  # noqa: BLE001
        print(f"[Persistence] 嵌入模型 int8 量化跳过（保持 fp32）: {exc}")
        return embedder


def load_ood_gate(model_path: str, ood_stats_path: Optional[str] = None) -> Optional[dict]:
    """加载分类模型域外拒识参考统计（逐类 means / precs / threshold）。

    返回 None 表示未生成（域外门控不生效，仅保留关键词拒识）。
    """
    import numpy as np

    path = ood_stats_path
    if path is None:
        from config.settings import settings

        path = settings.ood_stats_path
    if not os.path.isfile(path):
        # 兼容模型目录内放置的 ood_stats.npz
        alt = os.path.join(model_path, "ood_stats.npz")
        if os.path.isfile(alt):
            path = alt
        else:
            return None
    try:
        data = np.load(path, allow_pickle=False)
        if "thresholds" in data.files:
            threshold = data["thresholds"]  # 逐类阈值 (num_labels,)
        else:
            # 兼容旧版单阈值
            threshold = float(data["threshold"])
        return {
            "means": data["means"],  # (num_labels, dim)
            "precs": data["precs"],  # (num_labels, dim, dim)
            "thresholds": threshold,
            "labels": list(data["labels"]),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[Persistence] OOD 统计加载失败: {exc}")
        return None


def persist_model(
    src_dir: str,
    target_dir: str,
    overwrite: bool = False,
) -> Dict:
    """把训练好的模型目录固化到目标位置，并在固化后校验完整性。"""
    missing = validate_model_dir(src_dir)
    if missing:
        raise FileNotFoundError(f"源模型目录不完整，缺失文件: {missing}")

    target = Path(target_dir)
    if target.exists() and not overwrite:
        raise FileExistsError(f"目标目录已存在: {target}（如需覆盖请传 overwrite=True）")

    target.mkdir(parents=True, exist_ok=True)
    for fname in os.listdir(src_dir):
        src = os.path.join(src_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, target / fname)

    missing_after = validate_model_dir(str(target))
    if missing_after:
        raise RuntimeError(f"固化后校验失败，缺失文件: {missing_after}")
    return describe_model(str(target))
