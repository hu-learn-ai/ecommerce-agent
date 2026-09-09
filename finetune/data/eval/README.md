# finetune/data/eval

人工标注测试集目录（P3 阶段使用，唯一评估口径）：

- `test_labeled.jsonl`：150-200 条人工编写/复核的客服问答，覆盖 8 场景 + 多轮 + 情绪 + OOD
- 与训练/扩写数据完全隔离（相似度去重），生成后回填 `configs/eval.yaml`

在 P1 阶段可先从 `data/processed/test.jsonl` 抽选人工复核，再逐步扩充到 150-200 条。
