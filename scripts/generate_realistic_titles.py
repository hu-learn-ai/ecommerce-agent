"""
使用 DeepSeek API 生成真实风格的淘宝商品标题训练数据

按 15 个类别各生成 600+ 条，混合中英文，包含品牌名、产品型号、规格、营销词。
总计 9000+ 条，远超现有 2000 条合成数据。

使用方法:
    python scripts/generate_realistic_titles.py

可选参数:
    --per_category  每个类别生成数量 (默认 600)
    --batch_size    每次 API 调用生成数量 (默认 50)
    --output        输出文件路径 (默认 data/processed/classify_train_v2.csv)
"""

import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ============================================================
# 配置
# ============================================================

CATEGORIES = [
    "手机数码",
    "服装鞋包",
    "美妆护肤",
    "食品生鲜",
    "家用电器",
    "母婴用品",
    "汽车用品",
    "珠宝首饰",
    "家居家装",
    "运动户外",
    "图书音像",
    "宠物用品",
    "箱包配饰",
    "医药保健",
    "礼品鲜花",
]

# 每个类别的品类描述 + 关键词提示，帮助 LLM 生成更真实的标题
CATEGORY_HINTS = {
    "手机数码": "手机、平板、耳机、充电宝、数据线、智能手表、相机、键盘鼠标、存储卡、手机壳膜",
    "服装鞋包": "T恤、牛仔裤、连衣裙、羽绒服、运动鞋、皮鞋、靴子、西装、卫衣、短裤、内衣",
    "美妆护肤": "面霜、精华液、口红、粉底液、面膜、香水、洁面乳、眼霜、防晒霜、卸妆水",
    "食品生鲜": "零食、咖啡、茶叶、坚果、牛奶、水果、大米、食用油、保健品、速食、调味品",
    "家用电器": "冰箱、洗衣机、空调、电饭煲、吸尘器、吹风机、电风扇、微波炉、扫地机器人、净水器",
    "母婴用品": "纸尿裤、奶粉、婴儿车、玩具、湿巾、辅食、孕产妇用品、儿童安全座椅、婴儿床",
    "汽车用品": "行车记录仪、汽车贴膜、车载支架、机油、轮胎、汽车脚垫、车载空气净化器、洗车机",
    "珠宝首饰": "项链、戒指、手链、耳环、黄金、白银、翡翠、钻石、珍珠、水晶、琥珀",
    "家居家装": "床品、窗帘、沙发、灯具、收纳盒、墙纸、床垫、拖把、花瓶、抱枕、四件套",
    "运动户外": "跑步鞋、瑜伽垫、帐篷、健身器材、自行车、钓鱼竿、登山包、游泳衣、羽毛球拍",
    "图书音像": "小说、教材、漫画、杂志、CD、DVD、电子书、字帖、绘本、考试用书",
    "宠物用品": "猫粮、狗粮、猫砂、宠物窝、牵引绳、宠物玩具、鱼缸、鸟笼、宠物药品、宠物服饰",
    "箱包配饰": "双肩包、手提包、钱包、皮带、墨镜、帽子、围巾、手表、胸针、发饰",
    "医药保健": "维生素、感冒药、创可贴、血压计、体温计、中药材、保健食品、按摩仪、护具",
    "礼品鲜花": "鲜花花束、永生花、礼品卡、巧克力礼盒、贺卡、工艺品、定制礼品、干花",
}

# 格式变化指令，让 LLM 生成不同风格的标题
FORMAT_STYLES = [
    "品牌名+产品名+规格参数",
    "纯中文产品名+规格",
    "英文品牌+中文产品描述+型号",
    "促销风格：满减/包邮/限时等关键词+产品名",
    "长标题：品牌+系列+型号+颜色+容量+适用人群",
    "短标题：仅产品名+核心规格",
    "品牌+产品名+适用场景",
    "无品牌：纯产品描述+规格参数",
    "电商SEO风格：多个关键词堆砌+产品名",
    "官方旗舰+品牌+产品名+规格+赠品信息",
]


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek 生成真实商品标题")
    parser.add_argument("--per_category", type=int, default=600, help="每个类别生成数量")
    parser.add_argument("--batch_size", type=int, default=50, help="每次 API 调用生成数量")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    return parser.parse_args()


def call_deepseek(api_key, base_url, model, prompt, max_retries=3):
    """调用 DeepSeek API，返回文本响应"""
    import requests

    url = f"{base_url}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是一个电商平台商品标题生成专家。你擅长生成真实的、多样化的淘宝风格商品标题。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.9,
        "max_tokens": 4096,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  [Retry {attempt + 1}/{max_retries}] API 错误: {e}")
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
    return None


def build_prompt(category, hint, style, count):
    """构造生成 prompt"""
    return f"""请生成 {count} 条真实的淘宝风格「{category}」商品标题。

品类范围：{hint}

格式要求：{style}

重要规则：
1. 标题中【不要】直接出现「{category}」这四个字，用具体产品名代替
2. 每条标题必须不同，不要重复
3. 混合使用中文品牌和英文品牌
4. 包含规格参数（如 500ml、256GB、XL码、4L 等）
5. 包含真实品牌名（如华为、小米、兰蔻、戴森、耐克等）
6. 长度 10-60 个字符
7. 体现真实电商标题风格：关键词堆砌、促销信息、规格参数等

输出格式：每行一条标题，不要编号，不要引号，不要解释。
直接输出 {count} 行纯文本标题，每行一条。"""


def parse_titles(response, category):
    """解析 API 响应，提取商品标题"""
    if not response:
        return []

    titles = []
    for line in response.strip().split("\n"):
        line = line.strip()
        # 去掉可能的编号前缀
        for prefix_pattern in ["1.", "2.", "3.", "- ", "* ", "• "]:
            if line.startswith(prefix_pattern):
                line = line[len(prefix_pattern) :].strip()
        # 去掉引号
        line = line.strip('"').strip("'").strip("`").strip()
        # 过滤空行和太短的
        if len(line) < 5:
            continue
        # 过滤包含类别名的（我们要模型学会分类而不是关键词匹配）
        if category in line:
            continue
        # 过滤重复
        if line not in titles:
            titles.append(line)

    return titles


def generate_for_category(category, hint, per_category, batch_size, api_key, base_url, model):
    """为一个类别生成所有标题"""
    all_titles = []
    seen = set()
    num_batches = (per_category + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        remaining = per_category - len(all_titles)
        this_batch = min(batch_size, remaining)
        style = random.choice(FORMAT_STYLES)

        prompt = build_prompt(category, hint, style, this_batch)
        response = call_deepseek(api_key, base_url, model, prompt)

        if response:
            titles = parse_titles(response, category)
            for t in titles:
                if t not in seen:
                    seen.add(t)
                    all_titles.append(t)

        print(
            f"  [{category}] Batch {batch_idx + 1}/{num_batches}: "
            f"累计 {len(all_titles)}/{per_category}"
        )

        # 如果数量不够，继续追加
        if len(all_titles) >= per_category:
            break

        # 避免 API 限流
        time.sleep(0.5)

    # 如果还不够，用已有标题做轻微变换
    while len(all_titles) < per_category and all_titles:
        base = random.choice(all_titles)
        variants = [
            f"【官方旗舰】{base}",
            f"{base} 正品包邮",
            f"{base} 限时特惠",
            f"新款 {base}",
            f"{base} 送运费险",
        ]
        v = random.choice(variants)
        if v not in seen:
            seen.add(v)
            all_titles.append(v)

    return all_titles[:per_category]


def main():
    args = parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    if not api_key:
        print("[Error] DEEPSEEK_API_KEY 未配置，请检查 .env 文件")
        sys.exit(1)

    output_path = args.output or os.path.join(
        PROJECT_ROOT, "data", "processed", "classify_train_v2.csv"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("[Generate] DeepSeek 商品标题生成")
    print(f"[Generate] 模型: {model}")
    print(f"[Generate] 类别数: {len(CATEGORIES)}")
    print(f"[Generate] 每类目标: {args.per_category} 条")
    print(f"[Generate] 每次批次: {args.batch_size} 条")
    print(f"[Generate] 预计总量: {len(CATEGORIES) * args.per_category} 条")
    print(f"[Generate] 输出文件: {output_path}")
    print()

    all_data = []  # [(text, label), ...]
    category_stats = defaultdict(int)

    for cat_idx, category in enumerate(CATEGORIES):
        print(f"\n[{cat_idx + 1}/{len(CATEGORIES)}] 生成类别: {category}")
        hint = CATEGORY_HINTS[category]

        titles = generate_for_category(
            category, hint, args.per_category, args.batch_size, api_key, base_url, model
        )

        for title in titles:
            all_data.append((title, category))
            category_stats[category] += 1

        print(f"  完成: {len(titles)} 条")

    # 打乱顺序
    random.shuffle(all_data)

    # 写入 CSV
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in all_data:
            writer.writerow([text, label])

    print(f"\n{'=' * 60}")
    print("生成完成！")
    print(f"总样本数: {len(all_data)}")
    print(f"输出文件: {output_path}")
    print("\n各类别分布:")
    for cat in CATEGORIES:
        print(f"  {cat}: {category_stats[cat]}")
    print("\n示例标题 (前 20 条):")
    for text, label in all_data[:20]:
        print(f"  [{label}] {text}")

    # 同时生成训练集/验证集分割
    val_ratio = 0.2
    val_count = int(len(all_data) * val_ratio)
    val_data = all_data[:val_count]
    train_data = all_data[val_count:]

    train_path = output_path.replace(".csv", "_train.csv")
    val_path = output_path.replace(".csv", "_val.csv")

    for path, data in [(train_path, train_data), (val_path, val_data)]:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            for text, label in data:
                writer.writerow([text, label])

    print(f"\n训练集: {train_path} ({len(train_data)} 条)")
    print(f"验证集: {val_path} ({len(val_data)} 条)")
    print("\n训练命令:")
    print(
        "  python scripts/train_classify_model.py --data data/processed/classify_train_v2_train.csv --epochs 10 --batch_size 32 --lr 2e-5"
    )


if __name__ == "__main__":
    main()
