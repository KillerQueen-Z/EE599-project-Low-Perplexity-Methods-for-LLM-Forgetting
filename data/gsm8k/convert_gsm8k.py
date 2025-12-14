"""
转换 GSM8K 为 instruction/JSON（instruction/input/output 字段，含 CoT），
便于与 ag_news_train.json 的格式对齐。
输出：gsm8k_train.json（默认）或自定义 --out。

用法：
  python convert_gsm8k.py --split train --out gsm8k_train.json
  python convert_gsm8k.py --split test  --out gsm8k_test.json

依赖：datasets
  pip install datasets
"""
import argparse
import json
from datasets import load_dataset


def to_instruction(example):
    q = example["question"]
    a = example["answer"]  # 已包含推理过程 + 最终答案（格式如 "#### 42"）
    return {
        "instruction": "Solve step by step and give the final answer (format like #### 42).",
        "input": q,
        "output": a,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "test"], help="数据切分")
    parser.add_argument("--out", default="gsm8k_train.json", help="输出文件名")
    args = parser.parse_args()

    ds = load_dataset("gsm8k", "main", split=args.split)
    records = [to_instruction(ex) for ex in ds]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"保存完成: {args.out}, 数量={len(records)}")


if __name__ == "__main__":
    main()

