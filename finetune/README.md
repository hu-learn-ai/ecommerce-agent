# finetune —— 电商客服大模型 QLoRA 微调

对应 `docs/微调项目执行计划.md`，本目录承载"数据 → 训练 → 评估 → 部署"全链路。

## 当前进度（P1 数据与环境）

- [x] 8 类政策库：`data/raw/policy_library.json`（ground truth 实体库，评估可判定指标的依据）
- [x] 人工种子数据：`data/raw/seeds/*.jsonl`（约 320 条，覆盖 8 场景 + 多轮 + 情绪）
- [x] 清洗/构建流水线：`scripts/clean_dataset.py`、`scripts/build_dataset.py`
- [x] 种子数据划分：`data/processed/{train,val,test}.jsonl`（320 条种子 100% 通过清洗）
- [x] DeepSeek 扩写链路冒烟验证（8 次调用、27 条新增、估算 ¥0.07；缓存可续跑）
- [x] 训练/合并/量化/评估/人工抽检/部署脚本全部就绪（见下）
- [x] model_router 接入点完成（Tier1 本地 vLLM + 探活兜底，见附录 A）
- [ ] LLM 扩写至 15k+（消耗 API 额度，命令见下）
- [ ] AutoDL 4090 环境跑通训练 smoke（见 `docs/微调项目执行计划.md` 附录 B）

> 数据规模现状：当前 `data/processed` 为 320 条种子 + 27 条冒烟扩写，合计 346 条
> （train 310 / val 18 / test 17）。执行下方第 2 条命令后会自动在缓存基础上续跑至 15k。

## 脚本一览（全部代码就绪，按流水线顺序）

| 脚本 | 阶段 | 说明 |
|------|------|------|
| `scripts/build_dataset.py` | 数据 | 种子合并 → DeepSeek 扩写（缓存续跑）→ 清洗 → 90/5/5 划分 → 防泄漏 |
| `scripts/clean_dataset.py` | 数据 | 格式校验 / 去重 / 实体覆盖质量过滤（可单独运行） |
| `scripts/train_qlora.py` | 训练 | QLoRA 4bit + assistant-only loss mask + 断点续跑 + smoke 参数 |
| `scripts/merge_adapter.py` | 部署 | LoRA adapter 合并导出完整模型 |
| `scripts/quantize_awq.py` | 部署 | AWQ 4-bit 量化（对比档，T4/16G 卡） |
| `scripts/evaluate_llm.py` | 评估 | 三方对比 + 四维 Judge + 可判定指标 + win rate + badcase |
| `scripts/human_eval.py` | 评估 | 人工抽检评分表 + Cohen kappa 一致性 |
| `scripts/serve_vllm.sh` | 部署 | vLLM OpenAI 兼容 API 启动 |
| `docker/` | 部署 | vLLM 镜像 + compose（模型目录挂载） |

## 完整流程（AutoDL 上按序执行）

```bash
# 1. 数据（本地或服务器）
python finetune/scripts/build_dataset.py --expand --target 15000

# 2. 训练（smoke 先验证链路）
python finetune/scripts/train_qlora.py --config finetune/configs/qlora_7b.yaml \
    --max-steps 50 --data-subset 200
python finetune/scripts/train_qlora.py --config finetune/configs/qlora_7b.yaml

# 3. 合并 adapter → 完整模型（供 vLLM / 评估使用）
python finetune/scripts/merge_adapter.py \
    --base-model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --adapter finetune/checkpoints/qwen-cs-7b \
    --output /root/autodl-tmp/models/qwen-cs-7b-merged

# 4. 三方评估（base / 微调 / DeepSeek，Judge 走 DeepSeek API）
python finetune/scripts/evaluate_llm.py \
    --base-model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
    --finetuned-model /root/autodl-tmp/models/qwen-cs-7b-merged \
    --judge-model deepseek-chat

# 5. 人工抽检 + kappa
python finetune/scripts/human_eval.py --sample --count 50
python finetune/scripts/human_eval.py --kappa --human-csv finetune/reports/human_scores.csv

# 6. 量化对比档（可选）
python finetune/scripts/quantize_awq.py \
    --model /root/autodl-tmp/models/qwen-cs-7b-merged \
    --output /root/autodl-tmp/models/qwen-cs-7b-awq \
    --calib-data finetune/data/processed/train.jsonl

# 7. 部署 + 接入 model_router
./finetune/scripts/serve_vllm.sh   # 或 docker compose -f finetune/docker/docker-compose.vllm.yml up -d
# 根目录 .env 设 SELF_HOSTED_CS=true / CS_MODEL_BASE_URL / CS_MODEL_NAME 后重启应用
```

评估产出：`reports/finetune_eval_report.md`（方法、口径、结果表、badcase、结论），
对应计划 S3 验收项；部署与量化对比填入计划 5.2 表。

## 快速开始

```bash
# 1. 仅用种子数据构建（免费、无需 API）
python finetune/scripts/build_dataset.py --seeds-only

# 2. 全量扩写到目标规模（消耗 DeepSeek API 额度，约 ¥10-30/15k 条）
python finetune/scripts/build_dataset.py --expand --target 15000

# 3. 仅冒烟扩写（验证链路，约 20 条）
python finetune/scripts/build_dataset.py --expand --target 30 --smoke

# 4. 单独执行清洗
python finetune/scripts/clean_dataset.py --input data/raw/seed_data.jsonl
```

输出：

- `data/raw/seed_data.jsonl`：种子合并后的规范 JSONL
- `data/raw/expanded/`：DeepSeek 扩写缓存（按类别分批，可断点续跑，已 gitignore）
- `data/processed/train.jsonl / val.jsonl / test.jsonl`：90/5/5 分层划分，test 已做与 train 的相似度防泄漏
- `data/processed/dataset_report.json`：全流程统计（类别/场景/情绪分布、丢弃原因、防泄漏记录）

## 数据 Schema

```json
{
  "id": "cs_000123",
  "category": "售后_退换货",
  "scene": "多轮_追问",
  "emotion": "不满",
  "conversation": [
    {"role": "user", "content": "我昨天买的耳机有杂音，能退吗？"},
    {"role": "assistant", "content": "您好，商品出现质量问题支持7天无理由退换货……"}
  ],
  "policy_ref": ["faq_ret_001"],
  "answer_entities": {"policy": "7天无理由退换货", "duration": "7天"}
}
```

## 目录说明

```
finetune/
├── configs/           # 训练/评估/部署配置
├── data/raw/          # 政策库 + 种子数据（人工编写，入库）；expanded/ 为生成缓存（忽略）
├── data/processed/    # 清洗划分后的 JSONL（忽略）
├── data/eval/         # 人工标注测试集（P3 使用，唯一评估口径）
├── scripts/           # 数据/训练/评估/部署脚本
├── reports/           # 训练曲线、评估报告、成本账（忽略大文件）
└── docker/            # vLLM 服务（P4 使用）
```

> 与计划的差异说明：计划中 `data/raw/` 整体 gitignore；这里**保留人工种子与政策库入库**（小、可审计、是"可复现"的前提），仅忽略机器生成的扩写缓存与处理产物。如需严格按计划忽略 raw，删掉 `.gitignore` 中对应豁免即可。
