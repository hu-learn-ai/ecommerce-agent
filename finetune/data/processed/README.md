# finetune/data/processed

由 `finetune/scripts/build_dataset.py` 生成：

- `train.jsonl` / `val.jsonl` / `test.jsonl`：90/5/5 分层划分
- `dataset_report.json`：全流程统计（分布、丢弃原因、防泄漏）

该目录为生成产物（gitignore），复现方式：运行构建脚本。
