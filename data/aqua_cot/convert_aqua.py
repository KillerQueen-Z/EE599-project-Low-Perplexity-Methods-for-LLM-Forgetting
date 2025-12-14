"""
转换 AQuA-RAT 为 instruction/JSON（instruction/input/output 字段，含 CoT），
便于与 ag_news_train.json 的格式对齐。
输出：aqua_train.json（默认）或自定义 --out。

用法：
  python convert_aqua.py --split train --out aqua_train.json
  python convert_aqua.py --split test  --out aqua_test.json

依赖：datasets
  pip install datasets
"""
import argparse
import json
from datasets import load_dataset


def to_instruction(example):
    q = example["question"]
    # AQuA-RAT 的 rationale 字段包含完整的推理过程，correct 字段包含答案选项
    rationale = example.get("rationale", example.get("rationales", ""))
    options = example.get("options", "")
    correct = example.get("correct", "")
    
    # 构建完整的答案：推理过程 + 最终答案
    # rationale 已经包含了推理步骤和答案，我们直接使用
    if rationale:
        # rationale 通常已经包含了 "The answer is E." 这样的结尾
        # 如果还没有，我们添加
        if correct and correct not in rationale:
            answer = f"{rationale}\nThe correct answer is {correct}."
        else:
            answer = rationale
    else:
        answer = f"The correct answer is {correct}."
    
    # 将选项添加到问题中
    if options:
        question_with_options = f"{q}\n{options}"
    else:
        question_with_options = q
    
    return {
        "instruction": "Solve step by step and provide the correct answer.",
        "input": question_with_options,
        "output": answer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "test"], help="数据切分")
    parser.add_argument("--out", default="aqua_train.json", help="输出文件名")
    args = parser.parse_args()

    try:
        ds = load_dataset("aqua_rat", split=args.split)
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("尝试使用备用数据集名称...")
        try:
            ds = load_dataset("allenai/Aqua_rat", split=args.split)
        except Exception as e2:
            print(f"备用加载也失败: {e2}")
            return
    
    records = [to_instruction(ex) for ex in ds]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"保存完成: {args.out}, 数量={len(records)}")


if __name__ == "__main__":
    main()

