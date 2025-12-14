import os
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def scan_eval_configs(root: str) -> List[str]:
    eval_files: List[str] = []
    for r, _, files in os.walk(root):
        for f in files:
            if f == "eval_config.yaml":
                eval_files.append(os.path.join(r, f))
    return sorted(eval_files)


def run_eval(yaml_path: str) -> Tuple[bool, Dict[str, Any]]:
    """直接调用 llamafactory-cli 执行评估，返回是否成功及 metrics 字典（若存在 predict_results.json 则读取）。"""
    env = os.environ.copy()
    env["FORCE_TORCHRUN"] = "1"
    print(f"开始评估: {yaml_path}")
    proc = subprocess.Popen(
        ["llamafactory-cli", "train", yaml_path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=".",
    )
    for line in proc.stdout:
        print(line.strip())
    proc.wait()
    ok = proc.returncode == 0

    # 读取结果文件（若存在）
    output_dir = os.path.dirname(yaml_path)
    result_path = os.path.join(output_dir, "predict_results.json")
    result = {}
    if os.path.exists(result_path):
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                result = json.load(f)
        except Exception as e:
            print(f"读取结果失败 {result_path}: {e}")
    return ok, result


def _resolve_adapter_path(base_path: str) -> str:
    """
    如果 adapter_path 目录下没有 adapter_config.json，则向下递归查找，取最新的含 adapter_config.json 的子目录。
    返回可用的 adapter 路径（包含 adapter_config.json 的目录），否则返回原路径。
    """
    base = Path(base_path)
    cfg = base / "adapter_config.json"
    if cfg.exists():
        return str(base)

    candidates = []
    for p in base.rglob("adapter_config.json"):
        try:
            mtime = p.stat().st_mtime
        except Exception:
            mtime = 0
        candidates.append((mtime, p.parent))

    if not candidates:
        print(f"[WARN] 未找到 adapter_config.json 于 {base_path}")
        return str(base_path)

    candidates.sort(reverse=True, key=lambda x: x[0])
    resolved = str(candidates[0][1])
    print(f"[INFO] 解析 adapter 路径: {base_path} -> {resolved}")
    return resolved


def _prepare_yaml(yaml_path: str) -> str:
    """
    读取 eval_config.yaml，若 adapter_path 缺少 adapter_config.json，则解析到最后一个包含 adapter_config.json 的 checkpoint。
    若有修改，生成同目录下的临时文件 {yaml_path}.resolved.yaml 并返回新路径；否则返回原路径。
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"[WARN] 读取 {yaml_path} 失败: {e}")
        return yaml_path

    finetuning_type = data.get("finetuning_type", "")
    adapter_path = data.get("adapter_name_or_path")

    if finetuning_type != "lora" or not adapter_path:
        return yaml_path

    # 适配逗号分隔的多个 adapter（尽管这里应为单个）
    parts = [p.strip() for p in str(adapter_path).split(",") if p.strip()]
    resolved_parts = []
    changed = False
    for p in parts:
        rp = _resolve_adapter_path(p)
        resolved_parts.append(rp)
        if rp != p:
            changed = True

    if not changed:
        return yaml_path

    data["adapter_name_or_path"] = ",".join(resolved_parts)
    new_yaml = yaml_path + ".resolved.yaml"
    try:
        with open(new_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        print(f"[INFO] 生成解析后的 eval 配置: {new_yaml}")
        return new_yaml
    except Exception as e:
        print(f"[WARN] 写入解析文件失败，仍用原始配置: {e}")
        return yaml_path


def evaluate_yaml(yaml_path: str) -> Dict[str, Any]:
    yaml_to_use = _prepare_yaml(yaml_path)
    ok, metrics = run_eval(yaml_to_use)
    eval_dir = os.path.dirname(yaml_path)
    # 解析原始 yaml 以获取 eval_dataset
    eval_dataset = ""
    apply_mode = None
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
            eval_dataset = y.get("eval_dataset", "")
    except Exception:
        pass
    model_dir_path = Path(eval_dir).parent
    model_dir = model_dir_path.name
    mode_folder = model_dir_path.parent.name
    if "mask" in model_dir_path.parts:
        # .../mask/{apply}/{model_dir}/eval
        try:
            apply_mode = model_dir_path.parent.name
        except Exception:
            apply_mode = None
    record = {
        "yaml_path": yaml_path,
        "resolved_yaml": yaml_to_use,
        "status": "success" if ok else "failed",
        "raw_metrics": metrics,
        "eval_dir": eval_dir,
        "eval_dataset": eval_dataset,
        "model_dir": model_dir,
        "mode_folder": mode_folder,
        "apply_mode": apply_mode,
        "rouge_l": None,
        "exact_match": None,
    }
    if ok and metrics:
        if "predict_rouge-l" in metrics:
            record["rouge_l"] = round(metrics.get("predict_rouge-l", 0.0), 4)
        for k in ["predict_exact_match", "predict_em", "exact_match"]:
            if k in metrics:
                record["exact_match"] = round(metrics[k], 4)
                break
    return record


def main():
    import argparse

    parser = argparse.ArgumentParser(description="mask 扩展流水线评估脚本（独立版）")
    parser.add_argument("--root", required=True, help="训练输出根目录（如 *_mask 时间戳目录）")
    args = parser.parse_args()

    root = args.root
    print("扫描根目录:", root)
    result_path = os.path.join(root, "eval_results_mask.json")

    # 如已有结果文件，直接加载并绘图，跳过重新评估
    if os.path.exists(result_path):
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                results = data.get("results", [])
            print(f"检测到已有评估结果，直接绘图: {result_path}，共 {len(results)} 条")
            _plot_all_charts(results, root)
            print("\n===== 评估完成（复用现有结果） =====")
            print(f"结果文件：{result_path}")
            return
        except Exception as e:
            print(f"[WARN] 读取已有结果失败，将重新评估: {e}")

    eval_files = scan_eval_configs(root)
    print(f"发现 eval_config.yaml 数量: {len(eval_files)}")

    results: List[Dict[str, Any]] = []
    for yp in eval_files:
        record = evaluate_yaml(yp)
        results.append(record)
        per_eval_path = os.path.join(record["eval_dir"], "eval_summary.json")
        try:
            with open(per_eval_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"写入 {per_eval_path} 失败: {e}")

    try:
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"写入 {result_path} 失败: {e}")

    print("\n===== 评估完成 =====")
    print(f"结果保存至：{result_path}")
    try:
        _plot_all_charts(results, root)
    except Exception as e:
        print(f"[WARN] 绘制热力图失败: {e}")
        import traceback
        traceback.print_exc()


def _plot_all_charts(results: List[Dict[str, Any]], root: str):
    perf_dir = os.path.join(root, "performance_matrices_mask")
    os.makedirs(perf_dir, exist_ok=True)

    # 1) 基础 LoRA 柱状图（base_*）
    base_records = [r for r in results if r.get("model_dir", "").startswith("base_") and r.get("status") == "success"]
    base_by_ds = {}
    for r in base_records:
        ds = r.get("model_dir", "").replace("base_", "")
        base_by_ds[ds] = r.get("exact_match")
    if base_by_ds:
        plt.figure(figsize=(max(6, 0.6 * len(base_by_ds)), 4))
        xs = list(base_by_ds.keys())
        ys = [base_by_ds[k] if base_by_ds[k] is not None else np.nan for k in xs]
        plt.bar(xs, ys, color="#4C9AFF")
        plt.ylabel("Exact Match")
        plt.title("Base LoRA Performance")
        plt.xticks(rotation=45, ha="right")
        out = os.path.join(perf_dir, "bar_base_lora_exact_match.png")
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[INFO] 保存柱状图: {out}")

    # 辅助函数：构建方阵，行=source，列=target；对角线填 base 的性能
    def build_matrix(group_records, metric_key: str):
        # 收集 source/target
        sources = set()
        targets = set()
        for r in group_records:
            md = r.get("model_dir", "")
            if "_to_" in md:
                s, t = md.split("_to_", 1)
                sources.add(s)
                targets.add(t)
        # 也加入 base 数据集名，确保方阵
        for ds in base_by_ds.keys():
            sources.add(ds)
            targets.add(ds)
        sources = sorted(sources)
        targets = sorted(targets)
        mat = np.full((len(sources), len(targets)), np.nan)
        # 先填对角线 base
        for i, ds in enumerate(sources):
            if ds in base_by_ds and ds in targets:
                j = targets.index(ds)
                mat[i, j] = base_by_ds[ds]
        # 再填 off-diagonal
        for r in group_records:
            if r.get("status") != "success":
                continue
            md = r.get("model_dir", "")
            if "_to_" not in md:
                continue
            s, t = md.split("_to_", 1)
            if s in sources and t in targets:
                val = r.get(metric_key)
                if val is not None:
                    mat[sources.index(s), targets.index(t)] = val
        return sources, targets, mat

    def plot_mat(rows, cols, mat, title, fname):
        if np.isnan(mat).all():
            print(f"[WARN] {title} 全为空，跳过")
            return
        plt.figure(figsize=(max(8, 0.6 * len(cols)), max(6, 0.6 * len(rows))))
        sns.heatmap(mat, annot=True, fmt=".4g", xticklabels=cols, yticklabels=rows, cmap="YlGnBu")
        plt.title(title)
        plt.xlabel("target(eval_dataset)")
        plt.ylabel("source(model_dir)")
        out = os.path.join(perf_dir, fname)
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"[INFO] 保存热力图: {out}")

    # 2) 三种第二阶段：lora_transfer / mask/bottom10_group / mask/highest
    def filter_group(name: str, apply_mode: str = None):
        recs = []
        for r in results:
            if r.get("status") != "success":
                continue
            md = r.get("model_dir", "")
            if name == "lora_transfer":
                if "_to_" in md and "lora_transfer" in r.get("eval_dir", ""):
                    recs.append(r)
            elif name == "mask":
                if "_to_" in md and "mask" in r.get("eval_dir", ""):
                    if apply_mode and r.get("apply_mode") != apply_mode:
                        continue
                    recs.append(r)
        return recs

    groups = [
        ("lora_transfer", None, "lora_transfer"),
        ("mask", "bottom10_group", "mask_bottom10"),
        ("mask", "highest", "mask_highest"),
    ]
    for gname, apply_mode, tag in groups:
        recs = filter_group(gname, apply_mode)
        if not recs:
            print(f"[WARN] 组 {tag} 无可用结果，跳过")
            continue
        rows, cols, mat_em = build_matrix(recs, "exact_match")
        plot_mat(rows, cols, mat_em, f"Exact Match ({tag})", f"heatmap_exact_match_{tag}.png")
        rows, cols, mat_rouge = build_matrix(recs, "rouge_l")
        plot_mat(rows, cols, mat_rouge, f"ROUGE-L ({tag})", f"heatmap_rougeL_{tag}.png")


if __name__ == "__main__":
    main()

