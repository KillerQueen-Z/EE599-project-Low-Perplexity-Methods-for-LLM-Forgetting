"""
转换 OpenBookQA 为 instruction/JSON（instruction/input/output 字段，含 CoT），
便于与 ag_news_train.json 的格式对齐。
输出：openbookqa_train.json（默认）或自定义 --out。

用法：
  python convert_openbookqa.py --split train --out openbookqa_train.json
  python convert_openbookqa.py --split validation --out openbookqa_val.json
  python convert_openbookqa.py --split test --out openbookqa_test.json

依赖：datasets
  pip install datasets
"""
import argparse
import json
from datasets import load_dataset


def to_instruction(example):
    question_stem = example.get("question_stem", example.get("question", ""))
    choices = example.get("choices", {})
    answer_key = example.get("answerKey", example.get("answer_key", ""))
    
    # 构建选项文本
    if choices:
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        options_text = "\n".join([f"{label}. {text}" for label, text in zip(labels, texts)])
        full_question = f"{question_stem}\n{options_text}"
    else:
        full_question = question_stem
    
    # 构建答案
    answer = f"The correct answer is {answer_key}."
    
    return {
        "instruction": "Answer this open-book question by reasoning step by step, then provide the correct answer.",
        "input": full_question,
        "output": answer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"], help="数据切分")
    parser.add_argument("--out", default="openbookqa_train.json", help="输出文件名")
    args = parser.parse_args()

    try:
        ds = load_dataset("openbookqa", "main", split=args.split)
    except Exception as e:
        print(f"加载数据集失败: {e}")
        try:
            ds = load_dataset("allenai/openbookqa", split=args.split)
        except Exception as e2:
            print(f"备用加载也失败: {e2}")
            return
    
    records = [to_instruction(ex) for ex in ds]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"保存完成: {args.out}, 数量={len(records)}")


if __name__ == "__main__":
    main()

