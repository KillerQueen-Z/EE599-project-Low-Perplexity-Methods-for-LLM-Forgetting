# EE599 Project: Low-Perplexity Masking for Catastrophic Forgetting Mitigation

## Project Overview

This project implements a low-perplexity masking strategy for continuous fine-tuning to mitigate catastrophic forgetting in large language models during continual task learning.

## Project Structure

```
EE599-project/
├── scripts/                    # Training and evaluation scripts
│   ├── train_pipeline_mask.py  # Two-stage training pipeline (base LoRA + mask strategies)
│   └── eval_pipeline_mask.py   # Evaluation pipeline (with performance heatmap generation)
├── data/                       # Dataset directory
│   ├── gsm8k/                 # GSM8K mathematical reasoning dataset
│   ├── svamp/                 # SVAMP mathematical reasoning dataset
│   ├── aqua_cot/              # AQuA-RAT mathematical reasoning dataset
│   ├── simpleQA/              # SimpleQA question answering dataset
│   └── openbookqa/             # OpenBookQA question answering dataset
└── README.md                   # This file
```

## Core Methodology

### Two-Stage Training Pipeline

1. **Stage 1**: Base fine-tuning using standard LoRA on individual datasets
2. **Stage 2**: Continuous fine-tuning on a second dataset using three strategies:
   - **LoRA Transfer**: Standard LoRA continuous fine-tuning (baseline)
   - **Mask Bottom10**: Mask low-perplexity groups (bottom 10% confidence groups)
   - **Mask Highest**: Mask high-perplexity tokens (perplexity > threshold)

### Low-Perplexity Masking Strategy

- **Bottom10 Group**: Group tokens by perplexity and mask low-perplexity groups (parts where the model is more confident)
- **Highest**: Mask tokens with perplexity above a threshold (parts where the model is uncertain)

## Requirements

### Dependencies

```bash
pip install torch transformers peft datasets tqdm pytz pyyaml numpy seaborn matplotlib
```

### External Dependencies

1. **LLaMA-Factory**: For LoRA training and evaluation
   - Requires installation and configuration of `llamafactory-cli`
   - Requires training and evaluation YAML template files

2. **robust-llm-finetunes**: For mask training
   - Requires `train_with_mask.py` script
   - Path needs to be configured in `train_pipeline_mask.py`

3. **Base Model**: Llama-3.2-1B-Instruct
   - Path needs to be configured in `train_pipeline_mask.py`

## Configuration

### Modify Path Settings in `scripts/train_pipeline_mask.py`

Locate the configuration section at the beginning of the script and modify the following paths:

```python
# Base model path
BASE_MODEL_PATH = "/path/to/Llama-3.2-1B-Instruct"

# LLaMA-Factory root directory
LLAMA_ROOT = "/path/to/LLaMA-Factory"

# robust-llm-finetunes project directory
RLLF_ROOT = "/path/to/robust-llm-finetunes"

# Training output root directory
OUTPUT_ROOT = "/path/to/output"

# Training data root directory
TRAIN_DATA_ROOT = "/path/to/EE599-project/data"
```

### Training Parameters

The following training parameters can be adjusted in `train_pipeline_mask.py`:

- `TRAIN_EPOCHS = 3.0`: Number of training epochs
- `LEARNING_RATE = 1.0e-4`: Learning rate
- `BATCH_SIZE = 16`: Batch size
- `GRADIENT_ACCUMULATION = 4`: Gradient accumulation steps
- `LORA_RANK = 16`: LoRA rank
- `LORA_ALPHA = 16`: LoRA alpha
- `LORA_DROPOUT = 0.05`: LoRA dropout

### Mask Strategy Parameters

Mask parameters can be adjusted in the `run_mask_training` function:

- `--threshold "30.0"`: Perplexity threshold for the highest method
- `--group_size "4"`: Group size for the bottom10 method
- `--overlap_ratio "0.20"`: Group overlap ratio

## Usage

### 1. Prepare Datasets

Ensure each dataset in the `data/` directory contains the following files:
- `{dataset}_remaining_train.json`: Training set
- `{dataset}_test.json`: Test set
- `{dataset}_train_1p.json`: 1% replay data (optional)

Dataset format should be a JSON array, with each sample containing:
```json
{
  "instruction": "Solve step by step...",
  "input": "Question text",
  "output": "Answer with reasoning"
}
```

### 2. Run Training Pipeline

```bash
cd /path/to/LLaMA-Factory  # Must run from LLaMA-Factory root directory
python /path/to/EE599-project/scripts/train_pipeline_mask.py
```

The training process includes:
- Automatic scanning of datasets in the `data/` directory
- Automatic registration of datasets to `dataset_info.json`
- Stage 1: Training base LoRA adapters for each dataset
- Stage 2: Continuous fine-tuning for each dataset pair using three strategies

Training output will be saved in `OUTPUT_ROOT/{TIMESTAMP}_mask/` with the following structure:
```
{TIMESTAMP}_mask/
├── lora_base/              # Stage 1 base models
│   └── {dataset}/
├── lora_transfer/          # Stage 2: Standard LoRA
│   └── {dataset1}_to_{dataset2}/
├── mask/                   # Stage 2: Mask strategies
│   ├── bottom10_group/
│   │   └── {dataset1}_to_{dataset2}/
│   └── highest/
│       └── {dataset1}_to_{dataset2}/
```

### 3. Run Evaluation Pipeline

```bash
python /path/to/EE599-project/scripts/eval_pipeline_mask.py --root /path/to/{TIMESTAMP}_mask
```

The evaluation process includes:
- Scanning all `eval_config.yaml` files
- Executing evaluations and collecting metrics (ROUGE-L, Exact Match)
- Generating performance visualizations:
  - Base LoRA performance bar chart
  - Performance heatmaps for three strategies (Exact Match and ROUGE-L)

Evaluation results are saved in:
- `{TIMESTAMP}_mask/eval_results_mask.json`: Evaluation results JSON
- `{TIMESTAMP}_mask/performance_matrices_mask/`: Visualization images

If `eval_results_mask.json` already exists, the script will load it directly and generate plots, skipping re-evaluation.

## Evaluation Metrics

- **ROUGE-L**: Text similarity based on longest common subsequence
- **Exact Match (EM)**: Exact match rate

Evaluation focuses on the model's performance retention on the **first dataset** (to observe forgetting), rather than performance on the second dataset.

## Visualization Output

The evaluation script generates the following visualizations:

1. **Base LoRA Performance Bar Chart**: `bar_base_lora_exact_match.png`
2. **LoRA Transfer Heatmaps**:
   - `heatmap_exact_match_lora_transfer.png`
   - `heatmap_rougeL_lora_transfer.png`
3. **Mask Bottom10 Heatmaps**:
   - `heatmap_exact_match_mask_bottom10.png`
   - `heatmap_rougeL_mask_bottom10.png`
4. **Mask Highest Heatmaps**:
   - `heatmap_exact_match_mask_highest.png`
   - `heatmap_rougeL_mask_highest.png`

In the heatmaps:
- Rows (source): First training dataset
- Columns (target): Second training dataset (evaluated on the first dataset)
- Diagonal: Base LoRA performance
- Off-diagonal: Performance retention after continuous fine-tuning

## Important Notes

1. **Path Configuration**: Make sure to check and modify path configurations in the scripts before running
2. **Working Directory**: `train_pipeline_mask.py` must be run from the LLaMA-Factory root directory
3. **GPU Requirements**: Training and evaluation require GPU support (CUDA)
4. **Data Format**: Ensure dataset formats comply with LLaMA-Factory requirements
5. **Dependency Installation**: Ensure all dependency libraries and external tools are properly installed

## Experimental Settings

The experimental settings for this project:
- Model: Llama-3.2-1B-Instruct
- Training epochs: 3
- Learning rate: 1.0e-4
- Batch size: 16 (with 4 gradient accumulation steps, effective batch size 64)
- LoRA configuration: rank=16, alpha=16, dropout=0.05
- Datasets: 5 CoT (Chain-of-Thought) reasoning datasets

## Citation

If you use this project, please cite the relevant paper (to be added).

## License

(To be added)

## Contact

(To be added)
