import os
import json
import subprocess
import datetime
from pathlib import Path
from typing import List, Dict, Any

import pytz
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --------------------------
# 基础配置（独立于原 train_pipeline.py）
# --------------------------
BASE_MODEL_PATH = "/media/volume/V3/Llama-3.2-1B-Instruct"
TRAIN_TEMPLATE = "./examples/train_full/llama3_full_sft_llama_stage1_change.yaml"
EVAL_TEMPLATE = "./examples/train_full/llama3_lora_eval_test.yaml"
LLAMA_ROOT = "."
# 数据根 & 数据登记文件
TRAIN_DATA_ROOT = "/home/exouser/Desktop/vscode/LLaMA-Factory-Queen/data/auto_set_small_test_CoT"
DATA_DIR = os.path.join(LLAMA_ROOT, "data")
DATASET_INFO_PATH = os.path.join(DATA_DIR, "dataset_info.json")
# 训练输出根目录（改为 stm_experiment）
OUTPUT_ROOT = "/media/volume/V3/stm_experiment"

# LoRA 默认参数（与原流水线保持一致）
TRAIN_EPOCHS = 3.0
LEARNING_RATE = 1.0e-4
BATCH_SIZE = 16  # 增大训练批次大小
GRADIENT_ACCUMULATION = 4
EVAL_BATCH_SIZE = 16  # 评估批次大小
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET = "all"

# robust-llm-finetunes 脚本路径
RLLF_ROOT = "/home/exouser/Desktop/vscode/robust-llm-finetunes"
TRAIN_WITH_MASK = os.path.join(RLLF_ROOT, "train_with_mask.py")
GENERATE_STM = os.path.join(RLLF_ROOT, "generate_stm_training_data.py")

# 时间戳目录
california_tz = pytz.timezone("America/Los_Angeles")
now_utc = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
now_california = now_utc.astimezone(california_tz)
TIMESTAMP = now_california.strftime("%Y%m%d_%H%M%S")
TIMESTAMP_DIR = os.path.join(OUTPUT_ROOT, f"{TIMESTAMP}_mask")
Path(TIMESTAMP_DIR).mkdir(parents=True, exist_ok=True)


# --------------------------
# 生成/读取 YAML 辅助
# --------------------------
def generate_train_yaml(
    dataset: str,
    output_dir: str,
    finetuning_mode: str,
    replay_dataset: str | None = None,
    load_path: str | None = None,
) -> str:
    """根据 finetuning_mode 生成训练 YAML（简化版，仅 LoRA 需求）。"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    yaml_path = os.path.join(output_dir, "train_config.yaml")

    with open(TRAIN_TEMPLATE, "r", encoding="utf-8") as f:
        yaml_content = f.read()

    # 删除需覆盖的行
    lines = yaml_content.split("\n")
    new_lines = []
    for line in lines:
        if any(
            x in line.strip()
            for x in [
                "model_name_or_path:",
                "dataset:",
                "output_dir:",
                "finetuning_type:",
                "adapter_name_or_path:",
                "lora_rank",
                "lora_alpha",
                "lora_dropout",
                "lora_target",
            ]
        ):
            continue
        new_lines.append(line)

    cfg = []
    # 模型与适配器配置
    cfg.append(f"model_name_or_path: {BASE_MODEL_PATH}")
    if load_path:
        cfg.append(f"adapter_name_or_path: {load_path}")
    cfg.append("finetuning_type: lora")
    cfg.append(f"lora_rank: {LORA_RANK}")
    cfg.append(f"lora_alpha: {LORA_ALPHA}")
    cfg.append(f"lora_dropout: {LORA_DROPOUT}")
    cfg.append(f"lora_target: {LORA_TARGET}")

    # 数据集
    if replay_dataset:
        cfg.append(f'dataset: "{dataset},{replay_dataset}_1p"')
    else:
        cfg.append(f"dataset: {dataset}")

    cfg.append(f"output_dir: {output_dir}")

    # 插入配置到 training_args 前
    yaml_content = "\n".join(new_lines)
    lines_clean = yaml_content.split("\n")
    insert_pos = -1
    for i, line in enumerate(lines_clean):
        if line.strip().startswith("training_args:"):
            insert_pos = i
            break
    if insert_pos != -1:
        final_lines = lines_clean[:insert_pos] + cfg + lines_clean[insert_pos:]
    else:
        final_lines = lines_clean + cfg

    yaml_content = "\n".join(final_lines)
    # 参数覆盖
    yaml_content = yaml_content.replace("save_only_model: false", "save_only_model: true")
    yaml_content = yaml_content.replace("predict_with_generate: true", "predict_with_generate: false")
    yaml_content = yaml_content.replace("num_train_epochs: 3.0", f"num_train_epochs: {TRAIN_EPOCHS}")
    yaml_content = yaml_content.replace("learning_rate: 1.0e-5", f"learning_rate: {LEARNING_RATE}")
    yaml_content = yaml_content.replace("per_device_train_batch_size: 4", f"per_device_train_batch_size: {BATCH_SIZE}")
    yaml_content = yaml_content.replace("gradient_accumulation_steps: 2", f"gradient_accumulation_steps: {GRADIENT_ACCUMULATION}")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    return yaml_path


def generate_eval_yaml(
    eval_dataset: str,
    output_dir: str,
    finetuning_mode: str,
    model_path: str,
    adapter_path: str | None = None,
) -> str:
    """生成 eval_config.yaml（简化版，仅 LoRA 场景）。"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    yaml_path = os.path.join(output_dir, "eval_config.yaml")

    with open(EVAL_TEMPLATE, "r", encoding="utf-8") as f:
        yaml_content = f.read()

    lines = yaml_content.split("\n")
    new_lines = []
    for line in lines:
        if any(x in line.strip() for x in ["model_name_or_path:", "eval_dataset:", "output_dir:", "finetuning_type:", "adapter_name_or_path:", "per_device_eval_batch_size:"]):
            continue
        new_lines.append(line)

    cfg = []
    cfg.append(f"model_name_or_path: {model_path}")
    if adapter_path:
        cfg.append(f"adapter_name_or_path: {adapter_path}")
    cfg.append("finetuning_type: lora")
    cfg.append(f"eval_dataset: {eval_dataset}_test")
    cfg.append(f"output_dir: {output_dir}")
    cfg.append(f"per_device_eval_batch_size: {EVAL_BATCH_SIZE}")

    yaml_content = "\n".join(new_lines + cfg)
    yaml_content = yaml_content.replace("do_predict: true", "do_predict: true")
    yaml_content = yaml_content.replace("predict_with_generate: true", "predict_with_generate: true")

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    return yaml_path


def run_training(yaml_path: str) -> bool:
    """调用 llamafactory-cli 运行训练。"""
    try:
        env = os.environ.copy()
        env["FORCE_TORCHRUN"] = "1"
        print(f"开始训练，配置：{yaml_path}")
        proc = subprocess.Popen(
            ["llamafactory-cli", "train", yaml_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=LLAMA_ROOT,
        )
        for line in proc.stdout:
            print(line.strip())
        proc.wait()
        return proc.returncode == 0
    except Exception as e:
        print(f"训练出错: {e}")
        return False


# --------------------------
# 数据集发现（简单版）
# --------------------------
def list_datasets(root_dir: str) -> List[str]:
    """扫描 root_dir 下的子目录作为数据集名。"""
    datasets = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            datasets.append(item)
    return sorted(datasets)


# --------------------------
# 数据集登记到 dataset_info.json
# --------------------------
def register_datasets_to_info(root_dir: str, dataset_names: List[str]) -> None:
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

    if os.path.exists(DATASET_INFO_PATH):
        with open(DATASET_INFO_PATH, "r", encoding="utf-8") as f:
            try:
                dataset_info = json.load(f)
            except json.JSONDecodeError:
                print(f"警告：{DATASET_INFO_PATH} 格式错误，重建文件")
                dataset_info = {}
    else:
        dataset_info = {}
        print(f"创建新的 {DATASET_INFO_PATH}")

    added = []
    for dataset in dataset_names:
        train_file_abs = os.path.join(root_dir, dataset, f"{dataset}_remaining_train.json")
        test_file_abs = os.path.join(root_dir, dataset, f"{dataset}_test.json")
        replay_file_abs = os.path.join(root_dir, dataset, f"{dataset}_train_1p.json")

        if os.path.exists(train_file_abs):
            train_file_rel = os.path.relpath(train_file_abs, DATA_DIR)
            if dataset not in dataset_info:
                dataset_info[dataset] = {"file_name": train_file_rel}
                added.append(dataset)
                print(f"登记训练集 {dataset}: {train_file_rel}")
        else:
            print(f"警告：缺少训练文件 {train_file_abs}")

        if os.path.exists(test_file_abs):
            test_key = f"{dataset}_test"
            test_file_rel = os.path.relpath(test_file_abs, DATA_DIR)
            if test_key not in dataset_info:
                dataset_info[test_key] = {"file_name": test_file_rel}
                added.append(test_key)
                print(f"登记测试集 {test_key}: {test_file_rel}")
        else:
            print(f"警告：缺少测试文件 {test_file_abs}")

        if os.path.exists(replay_file_abs):
            replay_key = f"{dataset}_1p"
            replay_file_rel = os.path.relpath(replay_file_abs, DATA_DIR)
            if replay_key not in dataset_info:
                dataset_info[replay_key] = {"file_name": replay_file_rel}
                added.append(replay_key)
                print(f"登记1%重放 {replay_key}: {replay_file_rel}")

    if added:
        with open(DATASET_INFO_PATH, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        print(f"已登记 {len(added)} 个数据文件到 {DATASET_INFO_PATH}")
    else:
        print("数据文件已全部登记，无需更新")


# --------------------------
# STM 数据生成（直接计算 ppl，不依赖 generate_stm_training_data.py）
# --------------------------
def calculate_token_ppl(logits, target_ids):
    """计算每个 token 的困惑度"""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = target_ids[..., 1:].contiguous()
    loss_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none'
    ).view(shift_labels.size())
    perplexity_per_token = torch.exp(loss_per_token.float())
    return perplexity_per_token.float().cpu().numpy(), loss_per_token.float().cpu().numpy()


def ensure_stm_dataset(dataset: str, adapter_path: str, output_base_dir: str, max_length: int = 2048) -> str:
    """
    用指定 adapter 重新计算每个 token 的 ppl，保存到模型输出目录下的 stm_data 子文件夹。
    直接从 LLaMA-Factory 数据集格式计算，不依赖 generate_stm_training_data.py
    
    Args:
        dataset: 数据集名称
        adapter_path: adapter 路径
        output_base_dir: 模型输出基础目录（stm_data 将创建在此目录下）
        max_length: 最大序列长度
    
    Returns:
        data_path: 保存的 stm 数据路径
    """
    # stm_data 保存在模型输出目录下，不污染源数据
    stm_dir = os.path.join(output_base_dir, "stm_data", dataset)
    Path(stm_dir).mkdir(parents=True, exist_ok=True)
    data_path = stm_dir

    # 如果已存在且非空，直接返回
    if os.path.exists(data_path):
        try:
            # 尝试从磁盘加载
            from datasets import load_from_disk
            existing_ds = load_from_disk(data_path)
            if hasattr(existing_ds, '__len__') and len(existing_ds) > 0:
                sample = existing_ds[0] if hasattr(existing_ds, '__getitem__') else existing_ds['train'][0]
                if sample.get('ppl') is not None and len(sample.get('ppl', [])) > 0:
                    print(f"[STM] 数据集 {dataset} 已存在，跳过计算")
                    return data_path
        except Exception as e:
            print(f"[STM] 检查已存在数据集时出错，将重新计算: {e}")
            pass

    print(f"[STM] 生成数据集 {dataset}（adapter: {adapter_path}）")
    
    # 加载模型和 adapter
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map='cuda'
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.eval()

    # 加载 LLaMA-Factory 数据集
    # 数据集路径：TRAIN_DATA_ROOT/{dataset}/{dataset}_remaining_train.json
    dataset_file = os.path.join(TRAIN_DATA_ROOT, dataset, f"{dataset}_remaining_train.json")
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_file}")
    
    raw_dataset = load_dataset("json", data_files=dataset_file, split="train")
    
    # 计算 ppl
    data_with_ppl = []
    for example in tqdm(raw_dataset, desc=f"计算 {dataset} ppl"):
        # 获取对话格式
        conversations = example.get('conversations', [])
        if not conversations:
            # 如果没有 conversations，尝试从其他字段构建
            if 'instruction' in example and 'output' in example:
                conversations = [
                    {"role": "user", "content": example['instruction']},
                    {"role": "assistant", "content": example['output']}
                ]
            else:
                print(f"警告：跳过无法处理的样本: {example.keys()}")
                continue
        
        # 格式化消息
        try:
            formatted_message = tokenizer.apply_chat_template(
                conversations, tokenize=False, add_generation_prompt=False
            )
        except Exception as e:
            print(f"警告：chat_template 失败，使用简单拼接: {e}")
            # 简单拼接
            formatted_message = ""
            for msg in conversations:
                role = msg.get('role', 'user')
                content = msg.get('content', msg.get('value', ''))
                formatted_message += f"<|{role}|>\n{content}\n"
            formatted_message += "<|assistant|>\n"
        
        # Tokenize
        tokenized = tokenizer(
            formatted_message,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors='pt',
            add_special_tokens=False,
        )
        tokenized = tokenized.to(model.device)
        
        # 计算 ppl
        with torch.no_grad():
            outputs = model(tokenized.input_ids, labels=tokenized['input_ids'])
        
        token_ppls, _ = calculate_token_ppl(
            outputs.logits[0], 
            tokenized['input_ids']
        )
        # ppl 列表：第一个 token 设为 10（占位），后续是实际 ppl
        ppl_list = [10.0] + token_ppls[0, :].tolist()
        
        # 保存结果：确保保留 conversations 字段
        new_example = dict(example)  # 复制原始数据
        new_example['ppl'] = ppl_list
        new_example['text'] = formatted_message
        # 确保有 conversations 字段（collate_fn 需要）
        if 'conversations' not in new_example:
            new_example['conversations'] = conversations
        data_with_ppl.append(new_example)
    
    # 保存到磁盘
    Dataset.from_list(data_with_ppl).save_to_disk(data_path)
    print(f"[STM] 数据集已保存到: {data_path}")
    return data_path


# --------------------------
# 运行 train_with_mask (两种 apply)
# --------------------------
def run_mask_training(first_dataset: str, second_dataset: str, adapter_path: str, apply_mode: str) -> str:
    """
    使用已有 adapter 在第二个数据集上做 mask 微调。
    apply_mode: bottom10_group / highest
    """
    output_dir = os.path.join(TIMESTAMP_DIR, "mask", apply_mode, f"{first_dataset}_to_{second_dataset}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # stm_data 保存在模型输出目录下
    data_path = ensure_stm_dataset(second_dataset, adapter_path, output_dir)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        "python", TRAIN_WITH_MASK,
        "--target_model", BASE_MODEL_PATH,
        "--model_name", "mask_lora",
        "--dataset", second_dataset,
        "--learning_rate", "2e-5",
        # 放宽掩码强度：阈值调高（highest），组更小且重叠更低（bottom10）
        "--threshold", "30.0",
        "--group_size", "4",
        "--overlap_ratio", "0.20",
        "--adapter_path", adapter_path,
        "--data_path", data_path,
        "--apply", apply_mode,
        "--output_dir", output_dir,
        "--per_device_train_batch_size", str(BATCH_SIZE),
        "--gradient_accumulation_steps", str(GRADIENT_ACCUMULATION),
    ]
    print(f"[MASK] 开始训练 {first_dataset} -> {second_dataset} ({apply_mode})")
    subprocess.check_call(cmd, cwd=RLLF_ROOT, env=env)

    # 生成 eval 配置，评估第一个数据集以观察遗忘程度
    eval_dir = os.path.join(output_dir, "eval")
    generate_eval_yaml(
        eval_dataset=first_dataset,  # 评估第一个数据集，观察遗忘程度
        output_dir=eval_dir,
        finetuning_mode="lora",
        model_path=BASE_MODEL_PATH,
        adapter_path=output_dir,  # train_with_mask 输出目录即为 adapter_path
    )
    return output_dir


# --------------------------
# 阶段一：基础 LoRA（与原脚本一致）
# --------------------------
def stage1_train(datasets: List[str]) -> Dict[str, str]:
    base_adapters: Dict[str, str] = {}
    for dataset in datasets:
        print(f"\n===== 阶段一 (LoRA) 训练：{dataset} =====")
        output_dir = os.path.join(TIMESTAMP_DIR, "lora_base", dataset)
        yaml_path = generate_train_yaml(
            dataset=dataset,
            output_dir=output_dir,
            finetuning_mode="lora",
            replay_dataset=None,
            load_path=None,
        )
        if run_training(yaml_path):
            base_adapters[dataset] = output_dir
            eval_dir = os.path.join(output_dir, "eval")
            generate_eval_yaml(
                eval_dataset=dataset,
                output_dir=eval_dir,
                finetuning_mode="lora",
                model_path=BASE_MODEL_PATH,
                adapter_path=output_dir,
            )
        else:
            print(f"阶段一训练 {dataset} 失败，跳过其后续迁移")
    return base_adapters


# --------------------------
# 阶段二：三种微调策略
# 1) llamafactory LoRA（与原脚本第二阶段无重放类似）
# 2) train_with_mask bottom10_group
# 3) train_with_mask highest
# --------------------------
def stage2_train(datasets: List[str], base_adapters: Dict[str, str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for i, first_ds in enumerate(datasets):
        first_adapter = base_adapters.get(first_ds)
        if not first_adapter:
            continue
        for j, second_ds in enumerate(datasets):
            if i == j:
                continue

            # 1) llamafactory LoRA（无重放）
            print(f"\n===== 阶段二 (llamafactory) {first_ds} → {second_ds} =====")
            output_dir = os.path.join(TIMESTAMP_DIR, "lora_transfer", f"{first_ds}_to_{second_ds}")
            yaml_path = generate_train_yaml(
                dataset=second_ds,
                output_dir=output_dir,
                finetuning_mode="lora",
                replay_dataset=None,
                load_path=first_adapter,
            )
            run_training(yaml_path)

            eval_dir = os.path.join(output_dir, "eval")
            generate_eval_yaml(
                eval_dataset=first_ds,  # 评估第一个数据集，观察遗忘程度
                output_dir=eval_dir,
                finetuning_mode="lora",
                model_path=BASE_MODEL_PATH,
                adapter_path=output_dir,
            )

            # 2) train_with_mask bottom10_group
            mask_bottom_dir = run_mask_training(first_ds, second_ds, first_adapter, apply_mode="bottom10_group")

            # 3) train_with_mask highest
            mask_high_dir = run_mask_training(first_ds, second_ds, first_adapter, apply_mode="highest")

            entries.append(
                {
                    "first": first_ds,
                    "second": second_ds,
                    "llamafactory": output_dir,
                    "mask_bottom10": mask_bottom_dir,
                    "mask_high": mask_high_dir,
                }
            )
    return entries


# --------------------------
# 主入口
# --------------------------
def main():
    print("当前工作目录:", os.getcwd())
    print(f"时间戳输出目录: {TIMESTAMP_DIR}")
    datasets = list_datasets(TRAIN_DATA_ROOT)
    if not datasets:
        print("未找到数据集，退出")
        return

    # 登记数据到 dataset_info.json（便于 llamafactory 解析）
    register_datasets_to_info(TRAIN_DATA_ROOT, datasets)

    # 阶段一
    base_adapters = stage1_train(datasets)
    # 阶段二
    stage2_train(datasets, base_adapters)

    print("\n===== 新流水线训练完成 =====")
    print(f"输出目录：{TIMESTAMP_DIR}")
    print("阶段二包含三种策略：llamafactory LoRA / mask bottom10_group / mask highest")


if __name__ == "__main__":
    main()

