import os
import json
import random

def split_dataset(dataset_dir):
    """
    处理单个数据集文件夹中的[文件夹名]_train.json文件
    主训练集保留remaining命名，保存在各自数据集文件夹内
    """
    # 获取数据集名称（文件夹名称）
    dataset_name = os.path.basename(dataset_dir)
    
    # 定义原始数据文件路径：[文件夹名]_train.json
    original_train_file = os.path.join(dataset_dir, f"{dataset_name}_train.json")
    
    # 检查原始数据文件是否存在
    if not os.path.exists(original_train_file):
        print(f"警告: {dataset_dir} 中未找到 {dataset_name}_train.json，已跳过")
        return
    
    try:
        # 读取原始数据
        with open(original_train_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证数据格式
        if not isinstance(data, list):
            print(f"警告: {original_train_file} 内容不是列表格式，已跳过")
            return
        
        total_count = len(data)
        if total_count == 0:
            print(f"警告: {original_train_file} 内容为空，已跳过")
            return
        
        print(f"处理数据集: {dataset_name} - 原始数据量: {total_count}")
        
        # 为每条数据添加数据集名称标识
        labeled_data = []
        for item in data:
            if isinstance(item, dict):
                if 'dataset' not in item:
                    item['dataset'] = dataset_name
                labeled_data.append(item)
            else:
                labeled_data.append({
                    'dataset': dataset_name,
                    'data': item
                })
        
        # 随机打乱数据（固定种子确保可复现）
        random.shuffle(labeled_data)
        
        # 计算各数据集大小（独立采样）
        train_1p_size = max(1, int(total_count * 0.01))  # 总数据的1%（小训练集）
        test_size = max(1, int(total_count * 0.1))       # 总数据的10%（测试集）
        remaining_train_size = total_count - test_size   # 剩余90%（主训练集）
        
        # 拆分数据（独立采样逻辑）
        # 10%测试集和剩余主训练集
        test_data = labeled_data[:test_size]
        remaining_train_data = labeled_data[test_size:]
        
        # 独立抽取1%小训练集（重新打乱原始数据）
        temp_data = labeled_data.copy()
        random.shuffle(temp_data)
        train_1p_data = temp_data[:train_1p_size]
        
        # 输出路径：当前数据集文件夹内
        output_dir = dataset_dir
        
        # 恢复包含remaining的命名规则
        remaining_train_path = os.path.join(output_dir, f"{dataset_name}_remaining_train.json")  # 保留remaining
        test_path = os.path.join(output_dir, f"{dataset_name}_test.json")                      # 测试集
        train_1p_path = os.path.join(output_dir, f"{dataset_name}_train_1p.json")              # 1%小训练集
        
        # 保存文件
        with open(remaining_train_path, 'w', encoding='utf-8') as f:
            json.dump(remaining_train_data, f, ensure_ascii=False, indent=2)
        
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        with open(train_1p_path, 'w', encoding='utf-8') as f:
            json.dump(train_1p_data, f, ensure_ascii=False, indent=2)
        
        print(f"  主训练集数据量: {remaining_train_size} (剩余90%总数据) → {os.path.basename(remaining_train_path)}")
        print(f"  测试集数据量: {test_size} (总数据的10%) → {os.path.basename(test_path)}")
        print(f"  1%小训练集数据量: {train_1p_size} (总数据的1%) → {os.path.basename(train_1p_path)}")
        print(f"  输出路径: {output_dir}\n")
        
    except Exception as e:
        print(f"处理 {dataset_name} 时出错: {str(e)}\n")

def process_all_datasets(root_dir):
    """遍历总目录下的所有一级子文件夹（每个子文件夹为一个数据集）"""
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path):
            split_dataset(item_path)

if __name__ == "__main__":
    # 设置随机种子，确保结果可复现
    random.seed(42)
    
    # 总数据文件夹路径
    root_directory = "/home/exouser/Desktop/vscode/LLaMA-Factory-Queen/data/auto_set_small_test_CoT"
    
    # 检查总目录是否存在
    if not os.path.isdir(root_directory):
        print(f"错误: 总数据文件夹 {root_directory} 不存在")
    else:
        print(f"开始处理总目录: {root_directory}\n")
        process_all_datasets(root_directory)
        print("所有数据集处理完成!")
