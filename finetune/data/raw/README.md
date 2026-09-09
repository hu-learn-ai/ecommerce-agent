# finetune/data/raw

- `policy_library.json`：8 类政策库，ground truth 实体库（人工维护，禁止机器改写）
- `seeds/*.jsonl`：人工编写的高质量种子数据（320 条），按场景分文件
- `expanded/`：DeepSeek 扩写缓存（脚本生成，gitignore；可断点续跑）
- `seed_data.jsonl`：`build_dataset.py` 生成的种子合并规范文件

种子与政策库是"可解释、可审计"的核心资产，请用人工 review 而非脚本覆盖。
