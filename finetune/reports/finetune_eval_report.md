# 微调评估报告

- 测试集：/root/autodl-tmp/ecommerce-agent/finetune/data/processed/test.jsonl（289 条）
- 评估时间：2026-09-10 22:59:53
- 模型：base=`/root/autodl-tmp/models/Qwen2.5-7B-Instruct`，finetuned=`/root/autodl-tmp/models/qwen-cs-7b-5k-merged`，deepseek=`deepseek-chat`
- 口径：LLM-as-Judge 四维评分（1-5）+ 规则可判定指标 + 人工抽检（见 human_eval）

## 1. 四维 Judge 均分

| 模型 | 正确性 | 完整性 | 语气 | 安全 | 总分 |
|------|--------|--------|------|------|------|
| base | 3.78 | 3.65 | 4.66 | 4.85 | 16.94 |
| finetuned | 4.44 | 3.92 | 4.83 | 4.99 | 18.18 |
| deepseek | 4.65 | 4.57 | 4.96 | 4.98 | 19.16 |

## 2. 可判定指标

| 指标 | base | 微调 | DeepSeek | 目标 |
|------|------|------|----------|------|
| policy_citation_accuracy | 28.3% | 52.3% | 44.3% | ≥90% |
| hallucination_rate | 11.4% | 19.0% | 34.6% | ≤5% |
| ood_refusal_rate | 36.4% | 72.7% | 48.5% | ≥90% |
| multi_turn_keep_rate | 85.1% | 98.2% | 100.0% | ≥85% |

## 3. Win Rate

| 对比 | win | tie | lose | win rate |
|------|-----|-----|------|----------|
| 微调 vs base | 153 | 79 | 57 | 52.9% |
| 微调 vs DeepSeek | 36 | 103 | 150 | 12.5% |

## 4. Badcase（微调明显落后）

### vs base
- `cs_售后退换货_00853` delta=-3.0（finetuned 16.5 vs other 19.5）
- `cs_售后退换货_00602` delta=-2.5（finetuned 16.0 vs other 18.5）
- `cs_000313` delta=-3.0（finetuned 15.0 vs other 18.0）
- `cs_投诉安抚_00082` delta=-2.0（finetuned 14.0 vs other 16.0）
- `cs_投诉安抚_00618` delta=-2.0（finetuned 17.5 vs other 19.5）
- `cs_投诉安抚_00830` delta=-2.0（finetuned 17.0 vs other 19.0）
- `cs_投诉安抚_01124` delta=-2.0（finetuned 16.0 vs other 18.0）
- `cs_投诉安抚_00487` delta=-2.0（finetuned 16.0 vs other 18.0）
- `cs_投诉安抚_00099` delta=-3.0（finetuned 15.0 vs other 18.0）
- `cs_投诉安抚_01200` delta=-2.5（finetuned 16.0 vs other 18.5）
- `cs_投诉安抚_00247` delta=-2.5（finetuned 14.5 vs other 17.0）
- `cs_物流发货_01470` delta=-2.0（finetuned 16.0 vs other 18.0）
- `cs_域外拒答_00206` delta=-2.5（finetuned 17.0 vs other 19.5）
- `cs_域外拒答_01062` delta=-4.5（finetuned 13.0 vs other 17.5）
- `cs_域外拒答_01288` delta=-2.0（finetuned 17.5 vs other 19.5）
- `cs_政策规则_00613` delta=-7.0（finetuned 13.0 vs other 20.0）
- `cs_政策规则_00372` delta=-2.0（finetuned 16.0 vs other 18.0）
- `cs_政策规则_00138` delta=-4.0（finetuned 14.0 vs other 18.0）
- `cs_售前咨询_01435` delta=-3.0（finetuned 15.0 vs other 18.0）
- `cs_售前咨询_02178` delta=-2.0（finetuned 13.0 vs other 15.0）
- `cs_售前咨询_02702` delta=-2.0（finetuned 16.0 vs other 18.0）
- `cs_000001` delta=-6.0（finetuned 13.0 vs other 19.0）
- `cs_售前咨询_01026` delta=-2.0（finetuned 16.0 vs other 18.0）
- `cs_售前咨询_02447` delta=-3.0（finetuned 14.5 vs other 17.5）
- `cs_售前咨询_02609` delta=-2.0（finetuned 18.0 vs other 20.0）
- `cs_售前咨询_02226` delta=-4.0（finetuned 14.0 vs other 18.0）
- `cs_售前咨询_02149` delta=-3.0（finetuned 13.0 vs other 16.0）
- `cs_售前咨询_01241` delta=-5.5（finetuned 14.0 vs other 19.5）
- `cs_退款发票_01369` delta=-3.5（finetuned 16.5 vs other 20.0）
- `cs_退款发票_00867` delta=-5.0（finetuned 13.0 vs other 18.0）
### vs DeepSeek
- `cs_售后退换货_02639` delta=-2.0（finetuned 18.0 vs other 20.0）
- `cs_售后退换货_02526` delta=-2.0（finetuned 18.0 vs other 20.0）
- `cs_售后退换货_01076` delta=-3.0（finetuned 17.0 vs other 20.0）
- `cs_售后退换货_02890` delta=-2.0（finetuned 18.0 vs other 20.0）
- `cs_售后退换货_00221` delta=-3.0（finetuned 17.0 vs other 20.0）
- `cs_售后退换货_02959` delta=-2.5（finetuned 17.0 vs other 19.5）
- `cs_售后退换货_02069` delta=-3.5（finetuned 16.5 vs other 20.0）
- `cs_售后退换货_00853` delta=-2.0（finetuned 16.5 vs other 18.5）
- `cs_售后退换货_00422` delta=-2.5（finetuned 16.5 vs other 19.0）
- `cs_售后退换货_00602` delta=-4.0（finetuned 16.0 vs other 20.0）
- `cs_售后退换货_00060` delta=-3.0（finetuned 16.0 vs other 19.0）
- `cs_售后退换货_01690` delta=-2.0（finetuned 18.0 vs other 20.0）
- `cs_售后退换货_02734` delta=-3.0（finetuned 17.0 vs other 20.0）
- `cs_000081` delta=-2.5（finetuned 16.5 vs other 19.0）
- `cs_售后退换货_01776` delta=-2.5（finetuned 17.0 vs other 19.5）
- `cs_售后退换货_02853` delta=-3.0（finetuned 16.0 vs other 19.0）
- `cs_售后退换货_01427` delta=-2.0（finetuned 14.0 vs other 16.0）
- `cs_000313` delta=-4.5（finetuned 15.0 vs other 19.5）
- `cs_投诉安抚_00338` delta=-2.5（finetuned 17.0 vs other 19.5）
- `cs_投诉安抚_00082` delta=-4.0（finetuned 14.0 vs other 18.0）
- `cs_投诉安抚_00317` delta=-6.0（finetuned 14.0 vs other 20.0）
- `cs_投诉安抚_00730` delta=-6.0（finetuned 14.0 vs other 20.0）
- `cs_投诉安抚_01218` delta=-2.0（finetuned 18.0 vs other 20.0）
- `cs_投诉安抚_00868` delta=-2.0（finetuned 17.0 vs other 19.0）
- `cs_投诉安抚_00618` delta=-2.5（finetuned 17.5 vs other 20.0）
- `cs_投诉安抚_00724` delta=-3.0（finetuned 17.0 vs other 20.0）
- `cs_投诉安抚_00830` delta=-3.0（finetuned 17.0 vs other 20.0）
- `cs_投诉安抚_01124` delta=-4.0（finetuned 16.0 vs other 20.0）
- `cs_投诉安抚_00487` delta=-4.0（finetuned 16.0 vs other 20.0）
- `cs_投诉安抚_00723` delta=-2.0（finetuned 18.0 vs other 20.0）

## 5. 结论（人工复核于 2026-09-11）

- 客服域能否替代 DeepSeek：部分替代。本模型（5k 数据微调，消融组）对 DeepSeek 不输率 48%（win 12.5% / tie 35.6%），四维总分 18.18 vs 19.16；在政策引用准确率（52.3% vs 44.3%）、OOD 拒答率（72.7% vs 48.5%）、多轮保持率（98.2%）上优于 DeepSeek，幻觉率（19.0% vs 34.6%，规则口径）亦更低。具备承接 Tier-1 高频客服的能力，但综合表现不及 15k 版本（见 ④）。
- 适用边界：售前咨询、售后退换货、物流发货、投诉安抚、退款发票等客服域内场景可承接；域外拒答率 72.7%（未达 90% 目标），线上需关键词预过滤 + 置信度阈值回退兜底。
- 数据/口径备注：① Judge 为 deepseek-chat 四维评分（1-5）×2 次重复取均值，存在 ±0.1 级噪声；② 幻觉率为规则口径（政策库匹配失败即计幻觉），对长回答偏严，需人工抽检复核；③ 消融结论：对照组 reports_run_a/（15k 数据）四维总分 18.57、政策引用 78.5%、OOD 拒答 87.9%，全面优于本模型 —— 训练 loss 显示 15k 有过拟合倾向（loss 0.36 vs 本组 0.38），但下游评估反转，证明 loss 曲线不能代替下游评估；④ 主线确定为 15k 版本（reports_run_a/），本报告保留作消融对照。

