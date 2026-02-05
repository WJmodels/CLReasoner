import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel, PeftConfig
from datasets import Dataset
import pandas as pd
import re
import random
import os
import json
from tqdm import tqdm
import concurrent.futures
import math
import torch.multiprocessing as mp
from typing import List, Tuple

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

import importlib.util
import sys
import inspect


########################################## 计算推理指标模块辅助函数 ##########################################
def get_subdirectories(path):
    """
    获取指定路径下的一级子文件夹路径列表

    参数:
    path (str): 需要获取子文件夹的路径

    返回:
    list: 包含一级子文件夹路径的列表

    使用方法:
    subdirs = get_subdirectories('your_path_here')
    print(subdirs)
    """
    # 检查路径是否存在
    if not os.path.exists(path):
        raise ValueError(f"路径 '{path}' 不存在")

    # 获取路径下的所有子文件夹
    subdirectories = [os.path.join(path, name) for name in os.listdir(path) 
                      if os.path.isdir(os.path.join(path, name))]
    
    return subdirectories

def get_files_with_suffix(folder_path, suffix, traverse_sub_folder=False):
    """
    获取指定文件夹中指定后缀的所有文件路径
    
    参数：
        folder_path (str): 文件夹路径
        suffix (str): 文件的后缀，如 '.jpg' 或 '.png'
        traverse_sub_folder (bool): 是否遍历子文件夹，默认为 False
    
    返回：
        list: 包含所有符合条件的文件路径的列表
    """
    list_target_file_path = []

    # 遍历文件夹及子文件夹（如果设置）
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(suffix):
                list_target_file_path.append(os.path.join(root, file))
        
        # 如果不遍历子文件夹，则中断遍历
        if not traverse_sub_folder:
            break
    
    return list_target_file_path

def extract_dataset_name_from_txt_file_name_with_suffix(txt_file_name_with_suffix):
    """
    从文件名中提取数据集名称。
    兼容两种模式：
    1. 带step前缀: model_step_datasetname.txt -> datasetname
    2. 纯文件名: datasetname.txt -> datasetname
    """
    # 1. 尝试使用正则表达式匹配 'step_' 后面的内容
    match = re.search(r"step_([^.]+)\.txt", txt_file_name_with_suffix)
    if match:
        return match.group(1)
    
    # 2. [新增] 兜底逻辑：如果不是 step_ 格式，直接去除 .txt 后缀作为数据集名称
    if txt_file_name_with_suffix.endswith(".txt"):
        # 处理可能的双后缀情况 (.txt.txt)
        if txt_file_name_with_suffix.endswith(".txt.txt"):
             return txt_file_name_with_suffix[:-8]
        return txt_file_name_with_suffix[:-4]
        
    return None

def get_key_index_by_value(dict, target_value):
    # 接收一个字典的值，返回这个值对应键在字典中的索引
    # 获取字典的所有值的列表
    values = list(dict.values())
    
    # 查找目标值对应的索引
    try:
        index = values.index(target_value)
        return index
    except ValueError:
        return None  # 如果找不到该值，返回 None

def import_functions(list_tuple_py_file_abs_path_function_name):
    """
    动态导入指定列表中的每个 Python 文件的指定函数，
    同时将其添加到调用模块和__main__模块的命名空间。
    
    参数:
    list_tuple_py_file_abs_path_function_name: 一个列表，列表中的每个元素是一个元组，元组包含：
        - py_file_abs_path: Python 文件的绝对路径
        - function_name: 要导入的函数名称
    """
    # 获取调用者的帧和模块
    caller_frame = inspect.currentframe().f_back
    caller_globals = caller_frame.f_globals
    caller_module_name = caller_globals['__name__']
    
    # 获取调用者的模块对象
    if caller_module_name == '__main__':
        import __main__
        caller_module = __main__
    else:
        caller_module = sys.modules[caller_module_name]
    
    # 也获取__main__模块（无论调用者是不是__main__）
    import __main__
    main_globals = __main__.__dict__
    
    # 导入的函数列表（返回值）
    imported_functions = {}
    
    for py_file_abs_path, function_name in list_tuple_py_file_abs_path_function_name:
        try:
            # 通过绝对路径加载模块
            module_name = f"dynamic_module_{function_name}_{hash(py_file_abs_path)}"
            spec = importlib.util.spec_from_file_location(module_name, py_file_abs_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # 确保该函数存在
            if hasattr(module, function_name):
                func = getattr(module, function_name)
                
                # 设置到多个位置，确保函数在各种情况下都可用
                
                # 1. 设置到调用者模块
                setattr(caller_module, function_name, func)
                
                # 2. 设置到调用者的全局命名空间
                caller_globals[function_name] = func
                
                # 3. 设置到__main__模块（如果调用者不是__main__）
                if caller_module_name != '__main__':
                    main_globals[function_name] = func
                
                # 4. 加入返回字典
                imported_functions[function_name] = func
                
                print(f"成功导入 {function_name} 函数，来自文件：{py_file_abs_path}")
            else:
                print(f"警告：{py_file_abs_path} 中未找到函数 {function_name}")
        except Exception as e:
            print(f"错误：无法加载 {function_name} 函数，来自文件：{py_file_abs_path}，错误信息：{e}")
    
    return imported_functions

def replace_keys_with_values(one_result_folder_dict_stat, dict_key_test_arrow_folder_path_value_dataset_name):
    # 创建一个新的字典来存储替换后的结果
    updated_result = {}

    # 遍历 one_result_folder_dict_stat 中的键
    for key, value in one_result_folder_dict_stat.items():
        # 查找 dict_key_test_arrow_folder_path_value_dataset_name 中对应的值
        if key in dict_key_test_arrow_folder_path_value_dataset_name:
            new_key = dict_key_test_arrow_folder_path_value_dataset_name[key]
            updated_result[new_key] = value
        else:
            # 如果找不到对应的值，则保留原来的键
            updated_result[key] = value

    return updated_result

def save_evaluation_results_as_json_and_csv(dict_results, save_folder_path, dict_key_test_arrow_folder_path_value_dataset_name, save_result_filename_prefix="evaluation_result", decimal_places=4):
    """
    将评估结果保存为JSON和CSV文件。
    
    更新内容:
    1. 在JSON中注入 "_task_type_debug" 字段。
    2. 根据任务类型动态生成 CSV。
    3. [新增] 强制指定 NMR Forward 指标在 CSV 中的显示顺序 (MAE优先)。
    4. [新增] 修复 Pandas Pivot 导致的字母自动排序问题。
    """
    import json
    import csv
    import pandas as pd
    import os
    
    # 获取所有数据集名称
    all_dataset_names = list(dict_key_test_arrow_folder_path_value_dataset_name.values())
    
    # 找出所有训练步数并排序
    try:
        sorted_train_steps = sorted([int(step) for step in dict_results.keys()])
    except:
        sorted_train_steps = sorted(dict_results.keys())
    
    if not sorted_train_steps:
        return []

    # =========================================================
    # 1. 注入调试信息 (Task Type) 并分类数据集
    # =========================================================
    datasets_general = []
    datasets_nmr_forward = []
    
    for step in sorted_train_steps:
        step_str = str(step)
        step_data = dict_results.get(step_str, {})
        
        for ds_name in all_dataset_names:
            if ds_name not in step_data:
                continue
            
            metrics_dict = step_data[ds_name]
            
            # 判定逻辑：是否存在 NMR Forward 特有的 MAE 指标
            if "average_mae_zero_padding_1H_nmr_shift" in metrics_dict:
                task_type = "nmr_forward"
                if ds_name not in datasets_nmr_forward and ds_name not in datasets_general:
                    datasets_nmr_forward.append(ds_name)
            else:
                task_type = "general"
                if ds_name not in datasets_general and ds_name not in datasets_nmr_forward:
                    datasets_general.append(ds_name)
            
            metrics_dict["_task_type_debug"] = task_type

    print(f"检测到任务分类 (基于指标内容):")
    print(f"  - General Task Datasets ({len(datasets_general)}): {datasets_general}")
    print(f"  - NMR Forward Datasets  ({len(datasets_nmr_forward)}): {datasets_nmr_forward}")

    # =========================================================
    # 2. 保存 JSON
    # =========================================================
    json_file_path = os.path.join(save_folder_path, f"{save_result_filename_prefix}.json")
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(dict_results, json_file, indent=4)
    print(f"已保存完整 JSON: {json_file_path}")
    
    generated_files = [json_file_path]

    # =========================================================
    # 3. 定义两套指标模板 (控制顺序的关键位置)
    # =========================================================
    metrics_general = [
        "accuracy", "prediction_validity_rate", "average_tanimoto", "valid_smiles",
        "accuracy_040", "accuracy_060", "accuracy_080", "accuracy_090"
    ]
    
    # --- 修改: 调整了 NMR Forward 指标的顺序，MAE和Delta在前 ---
    metrics_nmr_forward = [
        # 核心误差指标
        "average_mae_zero_padding_13C_nmr_shift",
        "average_mae_zero_padding_1H_nmr_shift",
        "average_deta_13C_nmr_shift_count",
        "average_deta_1H_nmr_shift_count",
        
        # 其他误差指标
        "average_rmsd_zero_padding_13C_nmr_shift",
        "average_rmsd_zero_padding_1H_nmr_shift",
        "valid_samples_for_error_metrics",
        
        # CoT 结构相关指标
        "average_cot_13c_accuracy",
        "average_cot_1h_accuracy",
        "average_cot_brics_accuracy",
        "average_cot_13c_tanimoto",
        "average_cot_1h_tanimoto",
        "average_cot_brics_tanimoto",

        # ========================== 新增部分 START ==========================
        "total_samples",                        # 总样本数
        "valid_samples_for_deta_13c_count",     # 用于计算13C Delta的样本数
        "valid_samples_for_deta_1h_count",      # 用于计算1H Delta的样本数
        # ========================== 新增部分 END ============================
    ]
    # --------------------------------------------------------

    # =========================================================
    # 4. 辅助函数：生成 CSV
    # =========================================================
    def _write_csv(filename_suffix, target_datasets, target_metrics):
        if not target_datasets:
            return None
            
        csv_data = []
        for train_step in sorted_train_steps:
            train_step_str = str(train_step)
            step_data = dict_results.get(train_step_str, {})
            
            for metric in target_metrics:
                row = [train_step_str, metric]
                for ds_name in target_datasets:
                    val = step_data.get(ds_name, {}).get(metric, "")
                    if isinstance(val, (int, float)):
                        formatted_val = f"{val:.{decimal_places}f}"
                    else:
                        formatted_val = str(val)
                    row.append(formatted_val)
                csv_data.append(row)
        
        full_path = os.path.join(save_folder_path, f"{save_result_filename_prefix}{filename_suffix}.csv")
        header = ["train_step", "metric"] + target_datasets
        
        try:
            with open(full_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(csv_data)
            print(f"已保存 CSV: {full_path}")
            return full_path
        except Exception as e:
            print(f"保存 CSV 失败: {e}")
            return None

    # =========================================================
    # 5. 执行保存
    # =========================================================
    
    path_gen = _write_csv("_general", datasets_general, metrics_general)
    if path_gen: generated_files.append(path_gen)
    
    path_nmr = _write_csv("_nmr_forward", datasets_nmr_forward, metrics_nmr_forward)
    if path_nmr: generated_files.append(path_nmr)
    
    # 额外生成: 仅 Accuracy / MAE 简化表
    if datasets_general:
        acc_path = os.path.join(save_folder_path, f"{save_result_filename_prefix}_general_accuracy.csv")
        acc_rows = []
        for step in sorted_train_steps:
            row = [str(step)]
            for ds in datasets_general:
                val = dict_results.get(str(step), {}).get(ds, {}).get("accuracy", "")
                row.append(f"{val:.{decimal_places}f}" if isinstance(val, (int, float)) else str(val))
            acc_rows.append(row)
        with open(acc_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(["train_step"] + datasets_general)
            csv.writer(f).writerows(acc_rows)
        generated_files.append(acc_path)

    if datasets_nmr_forward:
        mae_path = os.path.join(save_folder_path, f"{save_result_filename_prefix}_nmr_forward_mae_1h.csv")
        mae_rows = []
        for step in sorted_train_steps:
            row = [str(step)]
            for ds in datasets_nmr_forward:
                val = dict_results.get(str(step), {}).get(ds, {}).get("average_mae_zero_padding_1H_nmr_shift", "")
                row.append(f"{val:.{decimal_places}f}" if isinstance(val, (int, float)) else str(val))
            mae_rows.append(row)
        with open(mae_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(["train_step"] + datasets_nmr_forward)
            csv.writer(f).writerows(mae_rows)
        generated_files.append(mae_path)

    # =========================================================
    # 6. 生成 Formatted CSV (Pandas 透视表) - 包含排序修复
    # =========================================================
    try:
        # 处理 General Formatted CSV
        if path_gen:
            df = pd.read_csv(path_gen)
            for c in df.columns[2:]: df[c] = pd.to_numeric(df[c], errors='ignore')
            
            # [Fix] 强制指定指标顺序
            valid_metrics = [m for m in metrics_general if m in df['metric'].unique()]
            df['metric'] = pd.Categorical(df['metric'], categories=valid_metrics, ordered=True)
            
            fmt_path = path_gen.replace(".csv", "_formatted.csv")
            df.pivot(index='train_step', columns='metric', values=datasets_general).to_csv(fmt_path)
            generated_files.append(fmt_path)
            
        # 处理 NMR Forward Formatted CSV
        if path_nmr:
            df = pd.read_csv(path_nmr)
            for c in df.columns[2:]: df[c] = pd.to_numeric(df[c], errors='ignore')
            
            # [Fix] 强制指定指标顺序 (关键步骤)
            valid_metrics = [m for m in metrics_nmr_forward if m in df['metric'].unique()]
            df['metric'] = pd.Categorical(df['metric'], categories=valid_metrics, ordered=True)
            
            fmt_path = path_nmr.replace(".csv", "_formatted.csv")
            df.pivot(index='train_step', columns='metric', values=datasets_nmr_forward).to_csv(fmt_path)
            generated_files.append(fmt_path)
            
    except Exception as e:
        print(f"生成 formatted csv 时出错: {e}")
        import traceback
        traceback.print_exc()

    return generated_files

def extract_formula_from_generate_result(str_generate_result_without_prompt):
    """
    从生成结果中提取分子式。
    
    参数:
        str_generate_result_without_prompt (str): 去除prompt后的生成结果字符串
        
    返回值:
        str: 提取的分子式字符串，如果未找到则返回None
    """
    import re
    
    # 使用正则表达式匹配 "Formula:" 到下一个 ";" 之间的内容
    pattern = r"Formula:([^;]+);"
    match = re.search(pattern, str_generate_result_without_prompt)
    
    if match:
        formula = match.group(1).strip()  # 去除可能的空格
        return formula
    else:
        return None

def normalize_molecular_formula(formula):
    """
    标准化分子式格式，处理不同顺序写法的分子式。
    将分子式转换为标准格式：C优先，然后H，再按字母顺序排列其他元素。
    
    参数:
        formula (str): 原始分子式字符串
        
    返回值:
        str: 标准化后的分子式字符串
    """
    import re
    from collections import defaultdict
    
    if not formula:
        return ""
    
    # 解析分子式，提取元素和数量
    element_count = defaultdict(int)
    
    # 使用正则表达式匹配元素符号和数量
    pattern = r'([A-Z][a-z]?)(\d*)'
    matches = re.findall(pattern, formula)
    
    for element, count_str in matches:
        count = int(count_str) if count_str else 1
        element_count[element] += count
    
    # 构建标准化的分子式：C优先，然后H，再按字母顺序排列其他元素
    result_parts = []
    
    # 优先处理C
    if 'C' in element_count:
        count = element_count['C']
        result_parts.append(f"C{count if count > 1 else ''}")
        del element_count['C']
    
    # 然后处理H
    if 'H' in element_count:
        count = element_count['H']
        result_parts.append(f"H{count if count > 1 else ''}")
        del element_count['H']
    
    # 其他元素按字母顺序排列
    for element in sorted(element_count.keys()):
        count = element_count[element]
        result_parts.append(f"{element}{count if count > 1 else ''}")
    
    return ''.join(result_parts)

def is_valid_smiles_with_formula_check(smiles, target_formula):
    """
    检查SMILES是否合格并且分子式匹配。
    
    参数:
        smiles (str): 待检查的SMILES字符串
        target_formula (str): 目标分子式
        
    返回值:
        tuple: (bool, str) - (是否合格, 标准SMILES)，如果不合格则标准SMILES为None
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    
    # 检查SMILES是否有效
    if not smiles or smiles == "F" or len(smiles) > 5000:
        return False, None
    
    try:
        # 转换为mol对象
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, None
        
        # 获取分子式
        mol_formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
        
        # 标准化两个分子式进行比较
        normalized_mol_formula = normalize_molecular_formula(mol_formula)
        normalized_target_formula = normalize_molecular_formula(target_formula)
        
        # 检查分子式是否匹配
        if normalized_mol_formula != normalized_target_formula:
            return False, None
        
        # 生成标准SMILES
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        return True, canonical_smiles
        
    except Exception as e:
        return False, None

########################################## 计算推理指标模块 ##########################################
def extract_str_generate_result_without_prompt(str_generate_result):
    return str_generate_result.split('###Response:')[-1]

def remove_disordered_tags(s, tuple_first_second):
    """
    移除字符串中不符合顺序的标签（如果 second 在 first 之前，则删除该 second）。
    
    参数:
        s (str): 输入字符串
        tuple_first_second (tuple): 包含两个标签的元组 (first, second)
        
    返回:
        str: 处理后的字符串
    """
    first, second = tuple_first_second  # 解析标签
    
    # 记录 first 和 second 的索引位置
    first_positions = [match.start() for match in re.finditer(re.escape(first), s)]
    second_positions = [match.start() for match in re.finditer(re.escape(second), s)]

    # 过滤掉不符合顺序的 second（即在所有 first 之前的 second）
    filtered_s = s
    for pos in second_positions:
        if not any(f_pos < pos for f_pos in first_positions):  # 如果 second 之前没有 first
            filtered_s = filtered_s[:pos] + filtered_s[pos + len(second):]  # 删除 second

    return filtered_s

def extract_str_answer(str_generate_result_without_prompt):
    str_generate_result_without_prompt = remove_disordered_tags(str_generate_result_without_prompt, ('<answer>', '</answer>'))
    # 检查是否有<think>和</think>标签
    bool_answer_start = '<answer>' in str_generate_result_without_prompt
    bool_answer_end = '</answer>' in str_generate_result_without_prompt
    bool_specific_str = '])' in str_generate_result_without_prompt
    bool_nat_question_str = 'infer the molecular SMILES:' in str_generate_result_without_prompt
    if bool_answer_start and bool_answer_end:
        # 开始和结束都有, 提取中间的内容
        str_answer = str_generate_result_without_prompt.split('</answer>')[-2].split('<answer>')[-1]
    elif bool_answer_start and not bool_answer_end:
        # 只有开始, 没有结束, 提取开始到最后的内容
        str_answer = str_generate_result_without_prompt.split('<answer>')[1]
    elif not bool_answer_start and bool_answer_end:
        # 只有结束, 没有开始, 则默认答案错误, SMILES为F
        str_answer = "F"
    else:
        # 没有开始和结束
        # 自然语言question, 提取最后一个"infer the molecular SMILES:"到整个字符串结束的部分
        if bool_nat_question_str:
            str_answer = str_generate_result_without_prompt.split("infer the molecular SMILES:")[-1].split('"')[0]
        # 化学语言question, 提取最后一个")]"到整个字符串结束的部分
        elif bool_specific_str:
            str_answer = str_generate_result_without_prompt.split("])")[-1].split('"')[0]
        else:
            # 如果连")]"都没有，则默认答案错误
            str_answer = "F"
    return str_answer

def extract_str_cot(str_generate_result_without_prompt):
    # 去掉</think>前的<think>
    str_generate_result_without_prompt = remove_disordered_tags(str_generate_result_without_prompt, ('<think>', '</think>'))
    # 检查是否有<think>和</think>标签
    bool_think_start = '<think>' in str_generate_result_without_prompt
    bool_think_end = '</think>' in str_generate_result_without_prompt
    # print(f"bool_think_start:{bool_think_start},bool_think_end:{bool_think_end}")
    if bool_think_start and bool_think_end:
        # 开始和结束都有, 提取中间的内容
        str_cot = str_generate_result_without_prompt.split('</think>')[-2].split('<think>')[-1]
    elif bool_think_start and not bool_think_end:
        # 只有开始, 没有结束, 提取开始到最后的内容
        str_cot = str_generate_result_without_prompt.split('<think>')[1]
    elif not bool_think_start and bool_think_end:
        # 只有结束, 没有开始, 提取最开始到结束的内容
        str_cot = str_generate_result_without_prompt.split('</think>')[0]
    else:
        # 没有开始和结束, 提取全部内容
        str_cot = str_generate_result_without_prompt
    # 不能带上answer标签部分
    if '<answer>' in str_cot:
        str_cot = str_cot.split('<answer>')[0]
    if '</answer>' in str_cot:
        str_cot = str_cot.split('</answer>')[0]
    return str_cot

def get_df_sample_info_from_arrow_file(arrow_file_path):
    # 通过datasets库加载.arrow文件
    dataset = Dataset.from_file(arrow_file_path)
    # 转为Pandas DataFrame
    df = dataset.to_pandas()
    return df

def load_question_cot_answer_list_from_arrow_folder(arrow_folder_path):
    """
    从arrow文件夹中加载所有arrow文件的问题、CoT和答案列表。
    
    参数:
        arrow_folder_path (str): 包含arrow文件的文件夹路径
        
    返回值:
        tuple: (list_question, list_cot, list_answer) - 合并所有arrow文件的数据列表
    """
    # 获取文件夹下所有.arrow文件
    list_arrow_file_path = get_files_with_suffix(arrow_folder_path, ".arrow", traverse_sub_folder=False)
    
    if not list_arrow_file_path:
        raise ValueError(f"在文件夹 '{arrow_folder_path}' 中未找到任何.arrow文件")
    
    # 按文件名排序，确保处理顺序的一致性
    list_arrow_file_path.sort()
    
    # 初始化合并后的列表
    combined_questions = []
    combined_cots = []
    combined_answers = []
    
    # 逐个处理arrow文件
    for arrow_file_path in list_arrow_file_path:
        print(f"正在加载arrow文件: {arrow_file_path}")
        df = get_df_sample_info_from_arrow_file(arrow_file_path)
        
        # 将当前文件的数据添加到合并列表中
        combined_questions.extend(df['question'].tolist())
        combined_cots.extend(df['cot'].tolist())
        combined_answers.extend(df['answer'].tolist())
    
    print(f"已成功合并 {len(list_arrow_file_path)} 个arrow文件，总样本数: {len(combined_questions)}")
    
    return combined_questions, combined_cots, combined_answers

def load_question_cot_answer_list_from_arrow_file(arrow_file_path):
    df = get_df_sample_info_from_arrow_file(arrow_file_path)
    list_question = df['question'].tolist()
    list_cot = df['cot'].tolist()
    list_answer = df['answer'].tolist()
    return list_question, list_cot, list_answer

def read_model_predictions(model_predict_result_txt_file_path):
    """
    生成器函数：逐行读取模型预测结果文件，并返回每行的内容（字符串形式）。

    参数:
        model_predict_result_txt_file_path (str): 模型预测结果的 .txt 文件路径。

    生成:
        str: 每次返回文件中的一行内容（去除换行符）。
    """
    try:
        with open(model_predict_result_txt_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                yield line.strip()  # 去掉首尾的换行符和空格
    except FileNotFoundError:
        print(f"错误：文件 '{model_predict_result_txt_file_path}' 未找到。")
    except Exception as e:
        print(f"发生错误：{e}")

def get_certain_line_content_from_model_predict_result_txt_file(model_predict_result_txt_file_path, line_count):
    """
    从模型预测结果文件中获取指定行的内容。
    
    参数:
        model_predict_result_txt_file_path (str): 模型预测结果的 .txt 文件路径
        line_count (int): 要返回的目标行号（从1开始计数）
        
    返回值:
        str: 指定行的内容；如果行号超出文件范围，则返回None
    """
    # 检查行号是否有效
    if line_count < 1:
        print(f"错误：行号 {line_count} 无效，行号应从1开始。")
        return None
        
    current_line = 1
    
    # 使用read_model_predictions生成器逐行读取文件
    for line in read_model_predictions(model_predict_result_txt_file_path):
        # 找到目标行
        if current_line == line_count:
            return line
        current_line += 1
    
    # 如果到达这里，说明文件行数少于目标行号
    print(f"错误：文件只有 {current_line - 1} 行，无法获取第 {line_count} 行。")
    return None

def get_one_line_cot_and_answer(model_predict_result_txt_file_path, line_count):
    content = get_certain_line_content_from_model_predict_result_txt_file(model_predict_result_txt_file_path, line_count)
    str_generate_result_without_prompt = extract_str_generate_result_without_prompt(content)
    str_cot = extract_str_cot(str_generate_result_without_prompt)
    str_answer = extract_str_answer(str_generate_result_without_prompt)
    return str_cot, str_answer

def read_list_generate_result_from_txt(model_predict_result_txt_file_path):
    """
    从模型预测结果txt文件中读取所有预测答案，并返回答案列表。
    
    参数:
        model_predict_result_txt_file_path (str): 模型预测结果的 .txt 文件路径
        
    返回值:
        list: 包含所有预测答案的列表
    
    说明:
        该函数会逐行读取txt文件，对每一行：
        1. 提取"###Response:"后的生成结果
        2. 从生成结果中提取<answer>标签内的答案
        3. 将答案添加到列表中并返回
    """
    list_generate_answer = []
    
    # 使用read_model_predictions生成器逐行读取文件
    for line in read_model_predictions(model_predict_result_txt_file_path):
        # 提取生成结果（去掉prompt部分）
        str_generate_result_without_prompt = extract_str_generate_result_without_prompt(line)
        # 提取答案部分
        str_answer = extract_str_answer(str_generate_result_without_prompt)
        # 添加到列表
        list_generate_answer.append(str_answer)
    
    return list_generate_answer

def calculate_smiles_validity_accuracy_from_ref_list_and_pred_list(
    list_answer, 
    list_generate_answer, 
    str_task="nmr",
    dechirality=False,
    check_length=5000,
    list_false_prefix=None,
    retrosyn_tanimoto_method="concatenated"  # 逆合成/合成任务通用的指纹计算方法
):
    """
    计算SMILES有效性和准确率（已修复 syn/retrosyn 任务下的 Validity 和 Tanimoto 计算问题）。
    """
    
    # 设置默认值
    if list_false_prefix is None:
        list_false_prefix = ["F_"]
    
    # 检查输入
    if len(list_answer) != len(list_generate_answer):
        raise ValueError(f"答案列表长度不匹配: {len(list_answer)} vs {len(list_generate_answer)}")
    
    # 选择判断函数
    if str_task.lower() == "nmr":
        judge_func = judge_single_nmr_answer
    elif str_task.lower() == "retrosyn":
        judge_func = judge_single_retrosyn_answer
    elif str_task.lower() == "syn":
        judge_func = judge_single_syn_answer
    else:
        raise ValueError(f"不支持的任务类型: {str_task}")
    
    # 初始化统计变量
    total_count = len(list_answer)
    valid_smiles_count = 0
    correct_count = 0
    
    # Tanimoto相似度统计
    tanimoto_sum = 0.0
    tanimoto_count = 0
    
    # 不同相似度阈值下的正确数
    correct_040_count = 0
    correct_060_count = 0
    correct_080_count = 0
    correct_090_count = 0
    
    # 逐个比较
    for gt_raw, pred_raw in zip(list_answer, list_generate_answer):
        
        # --- 步骤1：数据清理 (针对 RDKit 解析做准备) ---
        # 我们需要清理掉标签才能让 RDKit 识别有效性并计算指纹
        pred_clean = pred_raw
        gt_clean = gt_raw
        
        if str_task.lower() in ["retrosyn", "syn"]:
            # 1. 移除 [Reactant1], [Product1] 等标签
            pred_clean = remove_reaction_tags(pred_raw)
            gt_clean = remove_reaction_tags(gt_raw)
            
            # 2. 如果是逆合成，处理特定的 Reactant 占位符
            if str_task.lower() == "retrosyn":
                pred_clean = _replace_reactant_placeholders(pred_clean)
                gt_clean = _replace_reactant_placeholders(gt_clean)
                # 如果GT包含反应式 A>>B，提取反应物部分(A)
                if ">>" in gt_clean:
                    gt_clean = gt_clean.split(">>")[0].strip()
            
            # 3. 如果是合成，提取产物部分
            if str_task.lower() == "syn":
                # 处理可能存在的 [ProductN] 占位符
                pred_clean = _replace_product_placeholders(pred_clean)
                gt_clean = _replace_product_placeholders(gt_clean)
                # 如果GT包含反应式 A>>B，提取产物部分(B)
                if ">>" in gt_clean:
                    gt_clean = gt_clean.split(">>")[-1].strip()

        # --- 步骤2：检查有效性 (Validity) ---
        try:
            if not pred_clean or any(pred_clean.startswith(p) for p in list_false_prefix):
                is_valid = False
            elif str_task.lower() == "nmr":
                pred_mol = Chem.MolFromSmiles(pred_clean)
                is_valid = pred_mol is not None and len(pred_clean) < check_length
            else:
                # 反应任务：按点分割检查
                parts = pred_clean.split('.')
                is_valid = True
                for part in parts:
                    if not part.strip(): continue
                    mol = Chem.MolFromSmiles(part.strip())
                    if mol is None or len(part) >= check_length:
                        is_valid = False
                        break
        except:
            is_valid = False
        
        if is_valid:
            valid_smiles_count += 1
        
        # --- 步骤3：判断精确准确率 (Accuracy) ---
        # 这里直接使用原本的 judge_func，它内部已经处理了清理逻辑
        try:
            is_correct = judge_func(
                pred_raw,
                gt_raw,
                dechirality=dechirality,
                check_length=check_length,
                list_false_prefix=list_false_prefix
            )
            if is_correct:
                correct_count += 1
        except:
            pass
        
        # --- 步骤4：计算 Tanimoto 相似度 ---
        if is_valid:
            try:
                if str_task.lower() == "nmr":
                    # NMR任务：单分子Tanimoto
                    gt_mol = Chem.MolFromSmiles(gt_clean)
                    pred_mol = Chem.MolFromSmiles(pred_clean)
                    
                    if gt_mol is not None and pred_mol is not None:
                        if dechirality:
                            Chem.RemoveStereochemistry(gt_mol)
                            Chem.RemoveStereochemistry(pred_mol)
                        
                        gt_fp = AllChem.GetMorganFingerprintAsBitVect(gt_mol, 2, nBits=1024)
                        pred_fp = AllChem.GetMorganFingerprintAsBitVect(pred_mol, 2, nBits=1024)
                        
                        tanimoto = DataStructs.TanimotoSimilarity(gt_fp, pred_fp)
                        tanimoto_sum += tanimoto
                        tanimoto_count += 1
                
                else:
                    # 逆合成和合成任务：使用多分子集合的 Tanimoto 计算
                    # calculate_retrosyn_tanimoto_similarity 内部会自动处理 . 分隔的多个分子
                    tanimoto = calculate_retrosyn_tanimoto_similarity(
                        gt_clean,
                        pred_clean,
                        method=retrosyn_tanimoto_method,
                        dechirality=dechirality
                    )
                    
                    if tanimoto > 0:
                        tanimoto_sum += tanimoto
                        tanimoto_count += 1
                
                # 统计不同阈值下的准确率（仅在 valid 情况下统计）
                if 'tanimoto' in locals() or 'tanimoto' in globals():
                    if tanimoto >= 0.4: correct_040_count += 1
                    if tanimoto >= 0.6: correct_060_count += 1
                    if tanimoto >= 0.8: correct_080_count += 1
                    if tanimoto >= 0.9: correct_090_count += 1
            except:
                pass
    
    # 计算最终统计
    stats = {
        'prediction_validity_rate': valid_smiles_count / total_count if total_count > 0 else 0.0,
        'accuracy': correct_count / total_count if total_count > 0 else 0.0,
        'average_tanimoto': tanimoto_sum / tanimoto_count if tanimoto_count > 0 else 0.0,
        'valid_smiles': valid_smiles_count / total_count if total_count > 0 else 0.0,
        'accuracy_040': correct_040_count / total_count if total_count > 0 else 0.0,
        'accuracy_060': correct_060_count / total_count if total_count > 0 else 0.0,
        'accuracy_080': correct_080_count / total_count if total_count > 0 else 0.0,
        'accuracy_090': correct_090_count / total_count if total_count > 0 else 0.0,
    }
    
    return stats
def remove_extra_txt_extension(file_path):
    # 获取文件的完整路径和文件名
    file_name = os.path.basename(file_path)
    
    # 如果文件名以 '.txt.txt' 结尾，则去掉一个 '.txt'
    if file_name.endswith('.txt.txt'):
        new_file_name = file_name[:-4]  # 去掉最后的 '.txt'
        return os.path.join(os.path.dirname(file_path), new_file_name)
    
    # 如果文件名没有 '.txt.txt' 后缀，直接返回原路径
    return file_path

#### CoT-SC部分 ####
def vote_for_best_answer(list_candidate_info, list_priority_index):
    """
    对候选答案进行投票，选择最佳答案。
    
    参数:
        list_candidate_info (list): 候选信息列表，每个元素为(canonical_smiles, original_index)
        list_priority_index (list): 优先级索引列表，对应list_model_predict_result_txt_file_path的索引
        
    返回值:
        tuple: (best_canonical_smiles, best_original_index) - 最佳的标准SMILES和对应的原始索引
    """
    from collections import Counter, defaultdict
    
    if not list_candidate_info:
        return None, None
    
    # 统计每个标准SMILES的投票数
    smiles_votes = Counter(info[0] for info in list_candidate_info)
    
    # 找到最高票数
    max_votes = max(smiles_votes.values())
    
    # 找到所有最高票数的SMILES
    top_smiles = [smiles for smiles, votes in smiles_votes.items() if votes == max_votes]
    
    if len(top_smiles) == 1:
        # 没有平票，直接返回最高票的SMILES
        best_smiles = top_smiles[0]
        # 找到这个SMILES对应的原始索引中最小的那个
        candidates_with_best_smiles = [info for info in list_candidate_info if info[0] == best_smiles]
        best_original_index = min(info[1] for info in candidates_with_best_smiles)
        return best_smiles, best_original_index
    else:
        # 有平票，选择优先级最高（索引最小）的
        best_smiles = None
        best_priority = float('inf')
        best_original_index = None
        
        for smiles in top_smiles:
            candidates_with_this_smiles = [info for info in list_candidate_info if info[0] == smiles]
            min_original_index = min(info[1] for info in candidates_with_this_smiles)
            priority = list_priority_index[min_original_index]
            
            if priority < best_priority:
                best_priority = priority
                best_smiles = smiles
                best_original_index = min_original_index
        
        return best_smiles, best_original_index

def process_single_sample_cot_sc(args):
    """
    处理单个样本的CoT Self-Consistency，用于并行处理。
    
    参数:
        args (tuple): 包含(sample_idx, list_model_predict_result_txt_file_path, list_priority_index)
        
    返回值:
        str: 该样本的最终预测答案
    """
    sample_idx, list_model_predict_result_txt_file_path, list_priority_index = args
    
    # 收集该样本在所有预测文件中的候选答案
    list_candidate_info = []  # 存储 (canonical_smiles, original_file_index)
    list_all_raw_answers = []  # 存储所有原始答案，用于特殊情况处理
    
    for file_idx, txt_file_path in enumerate(list_model_predict_result_txt_file_path):
        try:
            # 获取该文件中该样本的预测结果
            line_content = get_certain_line_content_from_model_predict_result_txt_file(
                txt_file_path, sample_idx + 1
            )
            
            if line_content is None:
                continue
            
            # 提取生成结果
            str_generate_result_without_prompt = extract_str_generate_result_without_prompt(line_content)
            
            # 提取答案和分子式
            str_answer = extract_str_answer(str_generate_result_without_prompt)
            str_formula = extract_formula_from_generate_result(str_generate_result_without_prompt)
            
            # 保存原始答案用于特殊情况处理
            list_all_raw_answers.append((str_answer, file_idx))
            
            # 检查答案是否合格
            if str_formula:
                is_valid, canonical_smiles = is_valid_smiles_with_formula_check(str_answer, str_formula)
                if is_valid:
                    list_candidate_info.append((canonical_smiles, file_idx))
                    
        except Exception as e:
            # 在并行处理中，异常处理更加重要
            continue
    
    # 根据投票结果选择最佳答案
    if list_candidate_info:
        # 有合格的候选答案，进行投票
        best_canonical_smiles, best_file_index = vote_for_best_answer(
            list_candidate_info, list_priority_index
        )
        return best_canonical_smiles
    else:
        # 没有合格的候选答案，选择优先级最高的原始答案
        if list_all_raw_answers:
            # 按文件优先级排序，选择索引最小的
            list_all_raw_answers.sort(key=lambda x: x[1])
            fallback_answer = list_all_raw_answers[0][0]
            return fallback_answer
        else:
            # 连原始答案都没有，使用默认值
            return "F"

def test_model_predict_result_cot_sc(arrow_folder_path, list_model_predict_result_txt_file_path, use_parallel=True, max_workers=64, save_json_path=None):
    """
    测试模型预测结果，支持CoT Self-Consistency功能。
    对同一个问题的多个预测结果进行投票，选择最佳答案。
    
    参数:
        arrow_folder_path (str): 包含arrow文件的文件夹路径
        list_model_predict_result_txt_file_path (list): 模型预测结果txt文件路径列表，按优先级排序
        use_parallel (bool): 是否使用并行处理，默认True
        max_workers (int): 最大并行进程数，默认None（自动选择）
        save_json_path (str): 保存统计结果成.json文件的路径，None则不保存
    返回值:
        dict: 统计结果字典
    """
    import multiprocessing as mp
    from functools import partial
    
    # 从arrow文件夹中加载问题、CoT和答案列表
    list_question, list_cot, list_answer = load_question_cot_answer_list_from_arrow_folder(arrow_folder_path)
    
    # 获取样本数量
    num_samples = len(list_question)
    
    print(f"开始处理 {num_samples} 个样本，使用 {len(list_model_predict_result_txt_file_path)} 个预测文件")
    print(f"并行处理: {'启用' if use_parallel else '禁用'}")
    
    # 为每个样本生成优先级索引
    list_priority_index = list(range(len(list_model_predict_result_txt_file_path)))
    
    if use_parallel and num_samples > 1:
        # 使用并行处理
        if max_workers is None:
            # 自动选择进程数：CPU核心数，但不超过样本数
            max_workers = min(mp.cpu_count(), num_samples)
        
        print(f"使用 {max_workers} 个进程进行并行处理")
        
        # 准备参数列表
        args_list = [
            (sample_idx, list_model_predict_result_txt_file_path, list_priority_index)
            for sample_idx in range(num_samples)
        ]
        
        # 使用进程池进行并行处理
        try:
            with mp.Pool(processes=max_workers) as pool:
                # 使用imap_unordered获得进度反馈，但结果可能乱序
                results_dict = {}
                
                # 使用tqdm显示进度条
                from tqdm import tqdm
                
                for i, result in enumerate(tqdm(
                    pool.imap(process_single_sample_cot_sc, args_list),
                    total=num_samples,
                    desc="处理样本"
                )):
                    results_dict[args_list[i][0]] = result
                
                # 按样本索引顺序重新排列结果
                list_final_generate_answer = [
                    results_dict[sample_idx] for sample_idx in range(num_samples)
                ]
                
        except Exception as e:
            print(f"并行处理出现错误: {e}")
            print("回退到串行处理...")
            use_parallel = False
    
    if not use_parallel or num_samples <= 1:
        # 使用串行处理（原始逻辑）
        print("使用串行处理")
        list_final_generate_answer = []
        
        # 逐个处理每个样本
        for sample_idx in range(num_samples):
            if (sample_idx + 1) % 100 == 0:
                print(f"正在处理第 {sample_idx + 1}/{num_samples} 个样本")
            
            args = (sample_idx, list_model_predict_result_txt_file_path, list_priority_index)
            result = process_single_sample_cot_sc(args)
            list_final_generate_answer.append(result)
    
    # 计算统计指标
    stats = calculate_smiles_validity_accuracy_from_ref_list_and_pred_list(
        list_answer, list_final_generate_answer
    )
    
    # 打印结果
    print("\nCoT Self-Consistency 评估结果:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # 保存为json文件
    if save_json_path is not None:
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=4)
        print(f"统计结果已保存到 {save_json_path}")
    return stats

def _build_cot_sc_structure_with_accuracy_sorting(
    list_model_infer_save_folder_path,
    list_dataset_name, 
    save_folder_path,
    save_result_filename_prefix
):
    """
    按准确率排序构建CoT SC数据结构（需要读取CSV）
    """
    import pandas as pd
    import os
    
    # 存储所有模型的CSV数据
    list_model_csv_data = []
    
    # 1. 读取所有模型的accuracy.csv文件
    for model_idx, model_folder_path in enumerate(list_model_infer_save_folder_path):
        print(f"处理第 {model_idx + 1} 个模型文件夹: {model_folder_path}")
        
        # 查找_accuracy.csv文件
        list_accuracy_csv_files = get_files_with_suffix(model_folder_path, "_accuracy.csv", traverse_sub_folder=False)
        
        if len(list_accuracy_csv_files) == 0:
            raise FileNotFoundError(f"在文件夹 {model_folder_path} 中未找到 _accuracy.csv 文件")
        elif len(list_accuracy_csv_files) > 1:
            print(f"警告：在文件夹 {model_folder_path} 中找到多个 _accuracy.csv 文件，使用第一个: {list_accuracy_csv_files[0]}")
        
        accuracy_csv_path = list_accuracy_csv_files[0]
        print(f"读取文件: {accuracy_csv_path}")
        
        # 读取CSV文件
        try:
            df_accuracy = pd.read_csv(accuracy_csv_path)
            
            # 验证必要的列是否存在
            if 'train_step' not in df_accuracy.columns:
                raise ValueError(f"CSV文件 {accuracy_csv_path} 中缺少 'train_step' 列")
            
            # 存储模型数据
            model_data = {
                'model_folder_path': model_folder_path,
                'csv_path': accuracy_csv_path,
                'df': df_accuracy,
                'model_index': model_idx
            }
            list_model_csv_data.append(model_data)
            
            print(f"成功读取CSV，共 {len(df_accuracy)} 行数据")
            
        except Exception as e:
            raise Exception(f"读取CSV文件 {accuracy_csv_path} 时出错: {e}")
    
    # 2. 验证所有CSV文件的行数是否一致
    num_rows = len(list_model_csv_data[0]['df'])
    for model_data in list_model_csv_data:
        if len(model_data['df']) != num_rows:
            raise ValueError(f"所有模型的CSV文件行数必须一致，发现不一致的文件: {model_data['csv_path']}")
    
    print(f"所有CSV文件行数验证通过，共 {num_rows} 个训练阶段")
    
    # 3. 构建最终的数据结构
    list_dict_result = []
    
    # 对每个训练阶段（CSV中的每一行）进行处理
    for row_idx in range(num_rows):
        print(f"处理第 {row_idx + 1}/{num_rows} 个训练阶段")
        
        # 当前训练阶段的字典
        dict_current_stage = {}
        
        # 对每个数据集进行处理
        for dataset_name in list_dataset_name:
            print(f"  处理数据集: {dataset_name}")
            
            # 存储当前数据集的所有模型预测文件信息
            list_model_file_info = []
            
            # 遍历所有模型，记录模型索引
            for model_data in list_model_csv_data:
                model_folder_path = model_data['model_folder_path']
                df = model_data['df']
                model_idx = model_data['model_index']
                
                # 获取当前行的训练步数，确保转换为整数格式
                train_step = str(int(df.iloc[row_idx]['train_step']))
                
                # 构建对应的子文件夹路径
                step_folder_path = os.path.join(model_folder_path, train_step)
                
                if not os.path.exists(step_folder_path):
                    print(f"    警告：文件夹不存在 {step_folder_path}")
                    continue
                
                # 查找对应数据集的txt文件
                pattern_txt_files = get_files_with_suffix(step_folder_path, ".txt", traverse_sub_folder=False)
                
                # 筛选出匹配数据集名称的文件
                matching_txt_files = [
                    txt_file for txt_file in pattern_txt_files 
                    if txt_file.endswith(f"_{dataset_name}.txt")
                ]
                
                if len(matching_txt_files) == 0:
                    print(f"    警告：在 {step_folder_path} 中未找到匹配 {dataset_name} 的txt文件")
                    continue
                elif len(matching_txt_files) > 1:
                    print(f"    警告：在 {step_folder_path} 中找到多个匹配 {dataset_name} 的txt文件，使用第一个")
                
                txt_file_path = matching_txt_files[0]
                
                # 获取当前行中该数据集的准确率
                if dataset_name in df.columns:
                    accuracy = float(df.iloc[row_idx][dataset_name])
                else:
                    print(f"    警告：CSV中缺少数据集 {dataset_name} 的列，使用默认准确率0.0")
                    accuracy = 0.0
                
                # 存储文件信息，包含模型索引
                list_model_file_info.append({
                    'txt_file_path': txt_file_path,
                    'accuracy': accuracy,
                    'model_folder': model_folder_path,
                    'model_index': model_idx
                })
                
                print(f"    找到文件: {txt_file_path}, 准确率: {accuracy}")
            
            # 按准确率降序排序
            list_model_file_info.sort(key=lambda x: x['accuracy'], reverse=True)
            print(f"    数据集 {dataset_name} 按准确率排序")
            
            # 提取排序后的文件路径列表
            list_sorted_txt_files = [info['txt_file_path'] for info in list_model_file_info]
            
            # 存储到当前阶段的字典中
            dict_current_stage[dataset_name] = list_sorted_txt_files
            
            print(f"    数据集 {dataset_name} 处理完成，共 {len(list_sorted_txt_files)} 个文件")
        
        # 将当前训练阶段的字典添加到最终结果中
        list_dict_result.append(dict_current_stage)
    
    return list_dict_result

# 公共的保存摘要信息函数
def _save_summary_info(list_dict_result, list_model_infer_save_folder_path, list_dataset_name, save_folder_path, save_result_filename_prefix, sort_by_accuracy):
    """保存处理摘要信息"""
    import json
    import os
    
    # 4. 保存结果摘要（可选）
    summary_info = {
        'num_training_stages': len(list_dict_result),
        'num_models': len(list_model_infer_save_folder_path),
        'dataset_names': list_dataset_name,
        'model_folders': list_model_infer_save_folder_path,
        'sort_by_accuracy': sort_by_accuracy
    }
    
    print("\n处理完成摘要:")
    print(f"训练阶段数: {summary_info['num_training_stages']}")
    print(f"模型数量: {summary_info['num_models']}")
    print(f"数据集数量: {len(summary_info['dataset_names'])}")
    print(f"排序方式: {'按准确率排序' if sort_by_accuracy else '按训练步数排序'}")
    
    # 可选：将摘要信息保存到文件
    if save_folder_path and os.path.exists(save_folder_path):
        summary_file_path = os.path.join(save_folder_path, f"{save_result_filename_prefix}_cot_sc_summary.json")
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            json.dump(summary_info, f, indent=4, ensure_ascii=False)
        print(f"摘要信息已保存至: {summary_file_path}")

def _build_cot_sc_structure_with_step_sorting(
    list_model_infer_save_folder_path,
    list_dataset_name,
    save_folder_path, 
    save_result_filename_prefix
):
    """
    按训练步数排序构建CoT SC数据结构（不读取CSV，直接基于文件夹结构）
    """
    import os
    import re
    
    print("使用基于文件夹结构的处理方式，不读取CSV文件")
    
    # 1. 收集所有模型文件夹下的数字文件夹
    dict_model_step_folders = {}  # {model_idx: {step_num: step_folder_path}}
    
    for model_idx, model_folder_path in enumerate(list_model_infer_save_folder_path):
        print(f"扫描第 {model_idx + 1} 个模型文件夹: {model_folder_path}")
        
        # 获取所有子文件夹
        subdirs = get_subdirectories(model_folder_path)
        
        # 筛选出纯数字命名的文件夹
        step_folders = {}
        for subdir in subdirs:
            folder_name = os.path.basename(subdir)
            # 检查是否为纯数字
            if re.match(r'^\d+$', folder_name):
                step_num = int(folder_name)
                step_folders[step_num] = subdir
                
        if not step_folders:
            print(f"    警告：在 {model_folder_path} 中未找到数字命名的文件夹")
            continue
            
        dict_model_step_folders[model_idx] = step_folders
        print(f"    找到 {len(step_folders)} 个训练步数文件夹: {sorted(step_folders.keys())}")
    
    if not dict_model_step_folders:
        raise ValueError("所有模型文件夹中都没有找到数字命名的子文件夹")
    
    # 2. 对每个模型的训练步数进行排序，按位置索引对应
    dict_model_sorted_steps = {}  # {model_idx: [sorted_step_numbers]}
    
    for model_idx, step_folders in dict_model_step_folders.items():
        sorted_steps = sorted(step_folders.keys())
        dict_model_sorted_steps[model_idx] = sorted_steps
        print(f"    模型 {model_idx} 排序后的训练步数: {sorted_steps}")
    
    # 确定最小的训练阶段数（以最少的模型为准）
    if dict_model_sorted_steps:
        step_counts = [len(steps) for steps in dict_model_sorted_steps.values()]
        min_stages = min(step_counts)
        max_stages = max(step_counts)
        
        if min_stages != max_stages:
            print(f"警告：模型间训练阶段数不一致，最少 {min_stages} 个，最多 {max_stages} 个")
            print(f"将使用前 {min_stages} 个训练阶段进行对齐")
        
        print(f"共处理 {min_stages} 个训练阶段（按排序位置对应）")
    else:
        raise ValueError("没有找到任何有效的训练步数")
    
    # 3. 构建最终的数据结构（按位置索引对应）
    list_dict_result = []
    
    for stage_idx in range(min_stages):
        print(f"处理第 {stage_idx + 1}/{min_stages} 个训练阶段（位置索引 {stage_idx}）")
        
        # 当前训练阶段的字典
        dict_current_stage = {}
        
        # 显示当前阶段各模型对应的实际步数
        stage_steps_info = []
        for model_idx in sorted(dict_model_sorted_steps.keys()):
            actual_step = dict_model_sorted_steps[model_idx][stage_idx]
            stage_steps_info.append(f"模型{model_idx}:{actual_step}")
        print(f"  当前阶段对应步数: {', '.join(stage_steps_info)}")
        
        # 对每个数据集进行处理
        for dataset_name in list_dataset_name:
            print(f"  处理数据集: {dataset_name}")
            
            # 收集当前阶段所有模型的预测文件
            list_txt_files = []
            
            # 按模型索引顺序处理（保持list_model_infer_save_folder_path的顺序）
            for model_idx in sorted(dict_model_sorted_steps.keys()):
                # 获取当前模型在当前位置索引对应的实际步数
                actual_step_num = dict_model_sorted_steps[model_idx][stage_idx]
                step_folder_path = dict_model_step_folders[model_idx][actual_step_num]
                
                # 查找对应数据集的txt文件
                pattern_txt_files = get_files_with_suffix(step_folder_path, ".txt", traverse_sub_folder=False)
                
                # 筛选出匹配数据集名称的文件
                matching_txt_files = [
                    txt_file for txt_file in pattern_txt_files 
                    if txt_file.endswith(f"_{dataset_name}.txt")
                ]
                
                if len(matching_txt_files) == 0:
                    print(f"    警告：在 {step_folder_path} 中未找到匹配 {dataset_name} 的txt文件")
                    continue
                elif len(matching_txt_files) > 1:
                    print(f"    警告：在 {step_folder_path} 中找到多个匹配 {dataset_name} 的txt文件，使用第一个")
                
                txt_file_path = matching_txt_files[0]
                list_txt_files.append(txt_file_path)
                print(f"    模型{model_idx}(步数{actual_step_num}): {os.path.basename(txt_file_path)}")
            
            # 存储到当前阶段的字典中（已按模型索引顺序排列）
            dict_current_stage[dataset_name] = list_txt_files
            print(f"    数据集 {dataset_name} 按模型顺序排序，共 {len(list_txt_files)} 个文件")
        
        # 将当前训练阶段的字典添加到最终结果中
        list_dict_result.append(dict_current_stage)
    
    return list_dict_result

def get_list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path(
    list_model_infer_save_folder_path, 
    dict_key_test_arrow_folder_path_value_dataset_name, 
    save_result_filename_prefix, 
    save_folder_path,
    sort_by_accuracy=False
):
    """
    获取用于CoT Self-Consistency的数据结构，按准确率排序多个模型的预测结果文件。
    
    参数:
        list_model_infer_save_folder_path (list): 多个模型推理结果文件夹路径列表
        dict_key_test_arrow_folder_path_value_dataset_name (dict): 测试文件夹路径到数据集名称的映射
        save_result_filename_prefix (str): 结果文件名前缀
        save_folder_path (str): 输出结果的文件夹路径
        sort_by_accuracy (bool): 排序方式，True为按准确率降序排序（默认），False为按模型文件夹顺序排序
        
    返回值:
        list: list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path
              列表中每个元素是一个字典，键为数据集名称，值为排序后的预测文件路径列表
    """
    print(f"开始处理 {len(list_model_infer_save_folder_path)} 个模型文件夹")
    print(f"排序方式: {'按准确率排序' if sort_by_accuracy else '按训练步数排序'}")
    
    # 获取数据集名称列表
    list_dataset_name = list(dict_key_test_arrow_folder_path_value_dataset_name.values())
    print(f"数据集列表: {list_dataset_name}")
    
    if sort_by_accuracy:
        # 方式1：按准确率排序，需要读取CSV文件
        result = _build_cot_sc_structure_with_accuracy_sorting(
            list_model_infer_save_folder_path,
            list_dataset_name,
            save_folder_path,
            save_result_filename_prefix
        )
    else:
        # 方式2：按训练步数排序，直接基于文件夹结构
        result = _build_cot_sc_structure_with_step_sorting(
            list_model_infer_save_folder_path,
            list_dataset_name,
            save_folder_path,
            save_result_filename_prefix
        )
    
    # 保存摘要信息
    _save_summary_info(
        result, 
        list_model_infer_save_folder_path, 
        list_dataset_name, 
        save_folder_path, 
        save_result_filename_prefix,
        sort_by_accuracy
    )
    
    return result

def batch_evaluate_cot_sc_and_save_results(
    list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path,
    dict_key_test_arrow_folder_path_value_dataset_name,
    save_folder_path,
    save_result_filename_prefix="cot_sc_evaluation_result",
    use_parallel=True,
    max_workers=None,
    decimal_places=3
):
    """
    批量执行CoT Self-Consistency评估并保存结果为JSON和CSV格式。
    
    参数:
        list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path (list): 
            CoT SC数据结构，每个元素对应一个训练阶段的数据集-文件路径映射
        dict_key_test_arrow_folder_path_value_dataset_name (dict): 
            测试文件夹路径到数据集名称的映射字典
        save_folder_path (str): 保存结果文件的文件夹路径
        save_result_filename_prefix (str): 保存结果文件的前缀名称，默认"cot_sc_evaluation_result"
        use_parallel (bool): 是否使用并行处理，默认True
        max_workers (int): 最大并行进程数，默认None（自动选择）
        decimal_places (int): CSV文件中小数值保留的小数位数，默认3
        
    返回值:
        dict: 格式化后的评估结果字典，键为训练阶段（字符串），值为该阶段各数据集的评估统计
    """
    import os
    from tqdm import tqdm
    
    print("开始批量CoT Self-Consistency评估...")
    
    # 1. 验证输入参数
    if not list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path:
        raise ValueError("输入的CoT SC数据结构为空")
    
    # 创建数据集名称到arrow文件夹路径的反向映射
    dict_key_dataset_name_value_arrow_folder_path = {
        dataset_name: arrow_folder_path 
        for arrow_folder_path, dataset_name in dict_key_test_arrow_folder_path_value_dataset_name.items()
    }
    
    # 获取数据集名称列表
    list_dataset_name = list(dict_key_dataset_name_value_arrow_folder_path.keys())
    
    print(f"将评估 {len(list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path)} 个训练阶段")
    print(f"涉及数据集: {list_dataset_name}")
    print(f"并行处理: {'启用' if use_parallel else '禁用'}")
    
    # 2. 构建保存结果用的字典结构
    dict_results = {}
    
    # 3. 遍历每个训练阶段进行评估
    for stage_idx, dict_current_stage in enumerate(
        tqdm(list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path, 
             desc="评估训练阶段")
    ):
        print(f"\n处理第 {stage_idx + 1} 个训练阶段")
        
        # 为当前阶段创建结果字典
        stage_results = {}
        
        # 遍历当前阶段的每个数据集
        for dataset_name in list_dataset_name:
            if dataset_name not in dict_current_stage:
                print(f"  警告：当前阶段缺少数据集 {dataset_name}，跳过")
                continue
            
            list_model_predict_result_txt_file_path = dict_current_stage[dataset_name]
            
            if not list_model_predict_result_txt_file_path:
                print(f"  警告：数据集 {dataset_name} 没有可用的预测文件，跳过")
                continue
            
            print(f"  评估数据集 {dataset_name}，使用 {len(list_model_predict_result_txt_file_path)} 个预测文件")
            
            # 获取对应的arrow文件夹路径
            arrow_folder_path = dict_key_dataset_name_value_arrow_folder_path[dataset_name]
            
            try:
                # 调用CoT Self-Consistency评估函数
                stats = test_model_predict_result_cot_sc(
                    arrow_folder_path=arrow_folder_path,
                    list_model_predict_result_txt_file_path=list_model_predict_result_txt_file_path,
                    use_parallel=use_parallel,
                    max_workers=max_workers
                )
                
                # 存储当前数据集的评估结果
                stage_results[dataset_name] = stats
                print(f"  数据集 {dataset_name} 评估完成")
                
            except Exception as e:
                print(f"  错误：评估数据集 {dataset_name} 时出现异常: {e}")
                # 创建默认的错误结果
                stage_results[dataset_name] = {
                    "total_samples": 0,
                    "valid_references": 0,
                    "reference_validity_rate": 0.0,
                    "valid_predictions": 0,
                    "prediction_validity_rate": 0.0,
                    "correct_predictions": 0,
                    "accuracy": 0.0,
                    "average_tanimoto": 0.0,
                    "tanimoto_pairs_count": 0,
                    "correct_indices_start_0": []
                }
                continue
        
        # 将当前阶段的结果存储到总结果字典中
        # 使用阶段索引作为键（转换为字符串以与原有格式兼容）
        dict_results[str(stage_idx)] = stage_results
    
    print(f"\n所有训练阶段评估完成，共处理了 {len(dict_results)} 个阶段")
    
    # 4. 保存评估结果为JSON和CSV格式
    print("开始保存评估结果...")
    
    try:
        json_file_path, csv_file_path, formatted_csv_path, accuracy_csv_path = save_evaluation_results_as_json_and_csv(
            dict_results=dict_results,
            save_folder_path=save_folder_path,
            dict_key_test_arrow_folder_path_value_dataset_name=dict_key_test_arrow_folder_path_value_dataset_name,
            save_result_filename_prefix=save_result_filename_prefix,
            decimal_places=decimal_places
        )
        
        print("CoT Self-Consistency评估结果已成功保存:")
        print(f"  JSON文件: {json_file_path}")
        print(f"  CSV文件: {csv_file_path}")
        if formatted_csv_path:
            print(f"  格式化CSV: {formatted_csv_path}")
        print(f"  准确率CSV: {accuracy_csv_path}")
        
    except Exception as e:
        print(f"保存结果时出现错误: {e}")
        raise
    
    # 5. 输出评估摘要
    print("\n评估摘要:")
    print(f"训练阶段数: {len(dict_results)}")
    
    # 统计每个阶段的数据集数量
    dataset_counts = [len(stage_data) for stage_data in dict_results.values()]
    if dataset_counts:
        print(f"平均每阶段数据集数: {sum(dataset_counts) / len(dataset_counts):.1f}")
        print(f"数据集数量范围: {min(dataset_counts)} - {max(dataset_counts)}")
    
    return dict_results

def get_cot_sc_pipeline_results(
    list_model_infer_save_folder_path,
    dict_key_test_arrow_folder_path_value_dataset_name,
    save_result_filename_prefix,
    save_folder_path,
    use_parallel=True,
    max_workers=None,
    decimal_places=3
):
    """
    完整的CoT Self-Consistency评估管道：从模型结果文件夹到最终保存的评估结果。
    
    这是一个便捷函数，将整个CoT SC评估流程封装在一起：
    1. 从模型文件夹构建CoT SC数据结构
    2. 执行批量CoT SC评估
    3. 保存结果为JSON和CSV格式
    
    参数:
        list_model_infer_save_folder_path (list): 多个模型推理结果文件夹路径列表
        dict_key_test_arrow_folder_path_value_dataset_name (dict): 测试文件夹路径到数据集名称的映射
        save_result_filename_prefix (str): 结果文件名前缀
        save_folder_path (str): 输出结果的文件夹路径
        use_parallel (bool): 是否使用并行处理，默认True
        max_workers (int): 最大并行进程数，默认None（自动选择）
        decimal_places (int): CSV文件中小数值保留的小数位数，默认3
        
    返回值:
        tuple: (list_dict_cot_sc_data, dict_evaluation_results)
               - list_dict_cot_sc_data: CoT SC数据结构
               - dict_evaluation_results: 评估结果字典
    """
    print("=" * 60)
    print("开始完整的CoT Self-Consistency评估管道")
    print("=" * 60)
    
    # 步骤1：构建CoT SC数据结构
    print("步骤1：构建CoT Self-Consistency数据结构...")
    list_dict_cot_sc_data = get_list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path(
        list_model_infer_save_folder_path=list_model_infer_save_folder_path,
        dict_key_test_arrow_folder_path_value_dataset_name=dict_key_test_arrow_folder_path_value_dataset_name,
        save_result_filename_prefix=save_result_filename_prefix,
        save_folder_path=save_folder_path
    )
    
    print(f"CoT SC数据结构构建完成，共 {len(list_dict_cot_sc_data)} 个训练阶段")
    
    # 步骤2：执行批量CoT SC评估并保存结果
    print("\n步骤2：执行批量CoT Self-Consistency评估...")
    dict_evaluation_results = batch_evaluate_cot_sc_and_save_results(
        list_dict_key_dataset_name_value_list_model_predict_result_txt_file_path=list_dict_cot_sc_data,
        dict_key_test_arrow_folder_path_value_dataset_name=dict_key_test_arrow_folder_path_value_dataset_name,
        save_folder_path=save_folder_path,
        save_result_filename_prefix=save_result_filename_prefix,
        use_parallel=use_parallel,
        max_workers=max_workers,
        decimal_places=decimal_places
    )
    
    print("=" * 60)
    print("CoT Self-Consistency评估管道完成！")
    print("=" * 60)
    
    return list_dict_cot_sc_data, dict_evaluation_results

########## 新增函数（与之前相同）##########
########################################## NMR Forward 专用辅助函数 ##########################################
def _is_content_hallucinated(text: str) -> bool:
    """
    检测字符串是否包含 NMR 数据中不该出现的“幻觉/崩坏”特征。
    用于在提取数值前拦截严重崩坏的生成结果（如死循环、SMILES泄漏、异常关键词）。
    如果不拦截，这些数据可能会被错误地解析出数值，导致统计指标异常。
    
    注意：不应过滤 [MASK]，因为它是合法的占位符。
    """
    if not text: return False
    
    # 1. 检查死循环重复字符 (匹配连续5个以上的 ] . > )
    # 例如: "]]]]]]", "......", ">>>>>>"
    if re.search(r'([\]\.\>\)])\1{4,}', text):
        return True

    # 2. 检查怪异的重复模式 (例如: ):):):): )
    if text.count("):):") > 3:
        return True
        
    # 3. 检查 SMILES 泄漏特征 (NMR 数据区不该出现这些化学键符号)
    forbidden_markers = ["=C", "c1", "c2", "*):", ">>", "(:*)"]
    for marker in forbidden_markers:
        if marker in text:
            return True

    # 4. 检查模型自我崩溃的关键词 (这些词在正常的 NMR 数据中绝对不该出现)
    hallucination_keywords = [
        "corrupted", "irrelevant", "twisted", "terminating", 
        "approached", "letting", "expressing", "Finally"
    ]
    for kw in hallucination_keywords:
        if kw in text:
            return True
            
    return False

def _extract_tag_smiles(text: str, tag_prefix: str) -> dict:
    """提取 [C1]...[/C1] 格式的标签内容"""
    pattern = r'\[' + re.escape(tag_prefix) + r'(\d+)\](.*?)\[/'+ re.escape(tag_prefix) + r'\1\]'
    regex = re.compile(pattern, flags=re.DOTALL)
    result = {}
    for idx_str, content in regex.findall(text):
        idx = int(idx_str)
        smiles = content.strip()
        if smiles:
            result[idx] = smiles
    return result

def str_cot_to_dict(str_cot: str) -> dict:
    """将 CoT 字符串转换为结构化字典"""
    dict_cot = {
        "depth": {"13c": {}, "1h": {}},
        "brics": {}
    }
    if not isinstance(str_cot, str):
        return dict_cot
        
    c_dict = _extract_tag_smiles(str_cot, "C")
    h_dict = _extract_tag_smiles(str_cot, "H")
    brics_dict = _extract_tag_smiles(str_cot, "FS")

    dict_cot["depth"]["13c"] = {idx: c_dict[idx] for idx in sorted(c_dict.keys())} if c_dict else {}
    dict_cot["depth"]["1h"] = {idx: h_dict[idx] for idx in sorted(h_dict.keys())} if h_dict else {}
    dict_cot["brics"] = {idx: brics_dict[idx] for idx in sorted(brics_dict.keys())} if brics_dict else {}

    return dict_cot

def _smiles_to_mol_safe(smiles: str):
    if not smiles or not isinstance(smiles, str): return None
    try:
        return Chem.MolFromSmiles(smiles.strip())
    except:
        return None

def _stat_single_section(section_gt: dict, section_generate: dict, dechirality: bool) -> dict:
    """统计单个 CoT 部分（如 13C 片段）的准确率和相似度"""
    count = len(section_generate)
    if count == 0:
        return {"count": 0, "correct_count": 0, "average_tanimoto": 0.0}

    correct_count = 0
    tanimoto_sum = 0.0
    tanimoto_pair_count = 0

    for idx, smiles_gen in section_generate.items():
        mol_gen = _smiles_to_mol_safe(smiles_gen)
        smiles_gt = section_gt.get(idx, None)
        
        # 即使 GT 没有，或者无法解析，只要生成了就算在总数里，但在计算相似度时需要双方有效
        if smiles_gt is None: continue
        mol_gt = _smiles_to_mol_safe(smiles_gt)
        
        if mol_gen is None or mol_gt is None: continue

        if dechirality:
            mol_gen = Chem.Mol(mol_gen); Chem.RemoveStereochemistry(mol_gen)
            mol_gt = Chem.Mol(mol_gt); Chem.RemoveStereochemistry(mol_gt)

        # Exact Match
        try:
            smi_gen_std = Chem.MolToSmiles(mol_gen, canonical=True)
            smi_gt_std = Chem.MolToSmiles(mol_gt, canonical=True)
            if smi_gen_std == smi_gt_std:
                correct_count += 1
        except: pass

        # Tanimoto
        try:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol_gen, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol_gt, 2, nBits=2048)
            tanimoto_sum += DataStructs.TanimotoSimilarity(fp1, fp2)
            tanimoto_pair_count += 1
        except: pass

    average_tanimoto = tanimoto_sum / tanimoto_pair_count if tanimoto_pair_count > 0 else 0.0
    return {
        "count": count,
        "correct_count": correct_count,
        "average_tanimoto": float(average_tanimoto),
    }

def stat_dict_cot_generate_info(dict_cot_gt: dict, dict_cot_generate: dict, dechirality: bool = True) -> dict:
    """统计整个 CoT 的结构信息"""
    depth_gt = dict_cot_gt.get("depth", {})
    depth_gen = dict_cot_generate.get("depth", {})
    
    gt_13c = depth_gt.get("13c", {}) or {}
    gen_13c = depth_gen.get("13c", {}) or {}
    gt_1h = depth_gt.get("1h", {}) or {}
    gen_1h = depth_gen.get("1h", {}) or {}
    gt_brics = dict_cot_gt.get("brics", {}) or {}
    gen_brics = dict_cot_generate.get("brics", {}) or {}

    return {
        "depth": {
            "13c": _stat_single_section(gt_13c, gen_13c, dechirality),
            "1h": _stat_single_section(gt_1h, gen_1h, dechirality)
        },
        "brics": _stat_single_section(gt_brics, gen_brics, dechirality)
    }

def _extract_nmr_data_from_answer(answer_str: str) -> dict:
    """
    从 Answer 字符串提取 NMR 数值 (13C 和 1H)。
    
    修改说明:
    1. 增加了脏数据预检 (_is_content_hallucinated)，遇到崩坏数据直接返回空结果。
    2. 使用分号 ';' 精准切分 13C 和 1H 区域，避免正则跨区域误匹配。
    3. 在局部解析时也进行了二次脏数据检查。
    
    参数:
        answer_str (str): 模型生成的答案字符串。
        
    返回:
        dict: 包含 '13c_shifts', '1h_shifts' (列表) 和对应的 'shift_count' (整数)。
              如果解析失败或数据崩坏，列表为空，计数为 0。
    """
    result = {'13c_shifts': [], '1h_shifts': [], '13c_shift_count': 0, '1h_shift_count': 0}
    
    # 1. 基础合法性检查
    if not answer_str or not isinstance(answer_str, str): 
        return result
    if answer_str == "F_" or answer_str == "F": # 兼容两种错误标记
        return result

    # 2. 脏数据预检 (针对全文本崩坏的情况)
    if _is_content_hallucinated(answer_str):
        # 如果整段话都崩了，直接放弃提取，返回空结果
        return result
    
    # 3. 利用分号 ';' 进行切分 (精准定位，防止正则贪婪匹配到另一半)
    # 假设标准格式类似: Formula:..; 13C NMR:..; 1H NMR:..
    parts = answer_str.split(';')
    str_13c_content = ""
    str_1h_content = ""
    
    for part in parts:
        part = part.strip()
        if "13C NMR:" in part:
            str_13c_content = part.split("13C NMR:")[-1].strip()
        elif "1H NMR:" in part:
            str_1h_content = part.split("1H NMR:")[-1].strip()
            
    # 如果 split 没找到 (模型可能没输出分号)，尝试回退到 regex search (兼容旧逻辑)
    if not str_13c_content and not str_1h_content:
         # 13C 搜索
        match_13c = re.search(r'13C NMR:(.*?)(?:1H NMR:|$)', answer_str)
        if match_13c: str_13c_content = match_13c.group(1)
        # 1H 搜索
        match_1h = re.search(r'1H NMR:(.*)', answer_str)
        if match_1h: str_1h_content = match_1h.group(1)

    # --- 解析 13C ---
    # 增加局部脏数据检查
    if str_13c_content and not _is_content_hallucinated(str_13c_content):
        result['13c_nmr'] = str_13c_content # 用于调试记录
        
        # 提取策略: [NCP:xx]1.9,39.7
        # 先尝试去掉方括号头
        idx_bracket_close = str_13c_content.find(']')
        if idx_bracket_close != -1:
            raw_nums = str_13c_content[idx_bracket_close+1:]
        else:
            raw_nums = str_13c_content # 容错
            
        # 正则提取: 匹配浮点数或整数
        shifts = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", raw_nums)
        if shifts:
            result['13c_shifts'] = sorted([float(s) for s in shifts if s])
            result['13c_shift_count'] = len(result['13c_shifts'])
    
    # --- 解析 1H ---
    # 增加局部脏数据检查
    if str_1h_content and not _is_content_hallucinated(str_1h_content):
        result['1h_nmr'] = str_1h_content # 用于调试记录
        
        # 提取策略: [NHP:xx](0.05,9,[MASK])...
        idx_bracket_close = str_1h_content.find(']')
        raw_tuples = str_1h_content[idx_bracket_close+1:] if idx_bracket_close != -1 else str_1h_content
        
        # 正则: 匹配左括号 "(" 后面紧跟的数字 (位移值)
        shifts = re.findall(r'\(\s*([-+]?\d*\.?\d+)', raw_tuples)
        
        if shifts:
            result['1h_shifts'] = sorted([float(s) for s in shifts if s])
            result['1h_shift_count'] = len(result['1h_shifts'])
    
    return result

def _calculate_error_metrics(shifts_gen: list, shifts_gt: list):
    """计算 MSE, MAE, RMSD (Zero-padding)"""
    if not shifts_gen and not shifts_gt: return 0.0, 0.0, 0.0
    
    max_len = max(len(shifts_gen), len(shifts_gt))
    gen_padded = shifts_gen + [0.0] * (max_len - len(shifts_gen))
    gt_padded = shifts_gt + [0.0] * (max_len - len(shifts_gt))
    
    abs_errors = [abs(g - t) for g, t in zip(gen_padded, gt_padded)]
    squared_errors = [e ** 2 for e in abs_errors]
    
    mae = sum(abs_errors) / max_len if max_len > 0 else 0.0
    mse = sum(squared_errors) / max_len if max_len > 0 else 0.0
    rmsd = math.sqrt(mse)
    
    return mse, mae, rmsd

def stat_nmr_forward_answer_info(gt_answer: str, generate_answer: str, with_answer_for_deta_count: bool = False, max_length_for_deta_count: int = 5000 * 6) -> dict:
    """
    [修复版] 统计 NMR Forward 的 Answer 部分。
    修复了 13C 和 1H 过滤逻辑耦合的问题，现在它们是独立判断的。
    """
    # 1. 提取数据 (内部已包含 _is_content_hallucinated 的脏数据拦截)
    gt_data = _extract_nmr_data_from_answer(gt_answer)
    gen_data = _extract_nmr_data_from_answer(generate_answer)
    
    # 2. 基础检查
    # 判断 Tag 格式是否严重错误 (根据你的 extract_str_answer 逻辑，失败返回 "F" 或 "F_")
    is_tag_broken = (generate_answer in ["F", "F_", ""])
    
    answer_length = len(generate_answer) if generate_answer else 0
    answer_too_long = answer_length > max_length_for_deta_count
    
    # --- 13C 独立严格过滤 ---
    filter_13c = False
    if is_tag_broken or answer_too_long:
        filter_13c = True
    # 如果开启过滤，且生成的 Count 为 0 (说明没提取到有效数值，或者被脏数据检测拦截了)
    elif with_answer_for_deta_count and gen_data['13c_shift_count'] == 0:
        filter_13c = True
        
    if filter_13c:
        deta_13c = "FILTERED"
    else:
        deta_13c = gen_data['13c_shift_count'] - gt_data['13c_shift_count']

    # --- 1H 独立严格过滤 ---
    filter_1h = False
    if is_tag_broken or answer_too_long:
        filter_1h = True
    elif with_answer_for_deta_count and gen_data['1h_shift_count'] == 0:
        filter_1h = True
        
    if filter_1h:
        deta_1h = "FILTERED"
    else:
        deta_1h = gen_data['1h_shift_count'] - gt_data['1h_shift_count']
    
    # --- 误差计算 (即便被过滤，Raw Error 仍可计算用于调试，但全局统计会排除) ---
    mse_13c, mae_13c, rmsd_13c = _calculate_error_metrics(gen_data['13c_shifts'], gt_data['13c_shifts'])
    mse_1h, mae_1h, rmsd_1h = _calculate_error_metrics(gen_data['1h_shifts'], gt_data['1h_shifts'])
    
    return {
        "deta_13c": deta_13c,
        "deta_1h": deta_1h,
        "mse_13c": mse_13c, "mae_13c": mae_13c, "rmsd_13c": rmsd_13c,
        "mse_1h": mse_1h, "mae_1h": mae_1h, "rmsd_1h": rmsd_1h
    }

def calc_global_stats_from_memory(list_dict_main: list, delta_1h_count: int = 0, delta_13c_count: int = 0) -> dict:
    """
    [修复版] NMR Forward 全局指标统计函数
    实现了严格的 Joint Filter：只有 13C 和 1H 均有效的样本才计入 Delta 平均值。
    """
    if not list_dict_main: return {}
    
    total_count = len(list_dict_main)
    
    # 累加器
    acc = {
        "deta_13c_cnt": 0.0, "deta_1h_cnt": 0.0,
        "cot_13c_tani": 0.0, "cot_1h_tani": 0.0, "cot_brics_tani": 0.0,
        "cot_13c_acc_sum": 0.0, "cot_1h_acc_sum": 0.0, "cot_brics_acc_sum": 0.0,
        
        # 误差累加
        "mse_13c": 0.0, "mae_13c": 0.0, "rmsd_13c": 0.0,
        "mse_1h": 0.0, "mae_1h": 0.0, "rmsd_1h": 0.0,
    }
    
    # 计数器
    valid_joint_sample_count = 0        # 13C 和 1H 都有效的样本数
    filtered_count_for_error_metrics = 0 # 满足 Delta 阈值的样本数
    
    for item in list_dict_main:
        ans = item.get("answer", {})
        cot = item.get("cot", {})
        
        # 1. Delta Count 统计
        deta_13c = ans.get("deta_13c") 
        deta_1h = ans.get("deta_1h")
        
        # 检查是否为数字 (排除 "FILTERED")
        is_13c_valid = isinstance(deta_13c, (int, float))
        is_1h_valid = isinstance(deta_1h, (int, float))
        
        # 【严格连坐】只有两者都有效，才计入 Delta 平均
        if is_13c_valid and is_1h_valid:
            acc["deta_13c_cnt"] += deta_13c
            acc["deta_1h_cnt"] += deta_1h
            valid_joint_sample_count += 1
            
            # 2. Error Metrics 统计 (基于 Joint Valid 的样本)
            # 检查是否满足 Delta 阈值 (通常为 0)
            cond_1h = (delta_1h_count is None) or (abs(deta_1h) <= delta_1h_count)
            cond_13c = (delta_13c_count is None) or (abs(deta_13c) <= delta_13c_count)
            
            if cond_1h and cond_13c:
                filtered_count_for_error_metrics += 1
                acc["mse_13c"] += ans.get("mse_13c", 0.0)
                acc["mae_13c"] += ans.get("mae_13c", 0.0)
                acc["rmsd_13c"] += ans.get("rmsd_13c", 0.0)
                acc["mse_1h"] += ans.get("mse_1h", 0.0)
                acc["mae_1h"] += ans.get("mae_1h", 0.0)
                acc["rmsd_1h"] += ans.get("rmsd_1h", 0.0)
        
        # 3. CoT Stats (通常基于所有样本，无需过滤)
        d13c = cot.get("depth", {}).get("13c", {})
        d1h = cot.get("depth", {}).get("1h", {})
        brics = cot.get("brics", {})
        
        acc["cot_13c_tani"] += d13c.get("average_tanimoto", 0.0)
        acc["cot_1h_tani"] += d1h.get("average_tanimoto", 0.0)
        acc["cot_brics_tani"] += brics.get("average_tanimoto", 0.0)
        
        if d13c.get("count", 0) > 0: acc["cot_13c_acc_sum"] += d13c.get("correct_count", 0) / d13c.get("count")
        if d1h.get("count", 0) > 0: acc["cot_1h_acc_sum"] += d1h.get("correct_count", 0) / d1h.get("count")
        if brics.get("count", 0) > 0: acc["cot_brics_acc_sum"] += brics.get("correct_count", 0) / brics.get("count")

    # 计算平均值
    results = {}
    
    results["total_samples"] = total_count
    # 记录用于计算 Delta Count 的样本数 (Joint Valid)
    results["valid_samples_for_deta_13c_count"] = valid_joint_sample_count
    results["valid_samples_for_deta_1h_count"] = valid_joint_sample_count
    results["valid_samples_for_error_metrics"] = filtered_count_for_error_metrics

    # Delta Results
    if valid_joint_sample_count > 0:
        results["average_deta_13C_nmr_shift_count"] = acc["deta_13c_cnt"] / valid_joint_sample_count
        results["average_deta_1H_nmr_shift_count"] = acc["deta_1h_cnt"] / valid_joint_sample_count
    else:
        results["average_deta_13C_nmr_shift_count"] = 0.0
        results["average_deta_1H_nmr_shift_count"] = 0.0
    
    # CoT Results
    results["average_cot_13c_tanimoto"] = acc["cot_13c_tani"] / total_count
    results["average_cot_1h_tanimoto"] = acc["cot_1h_tani"] / total_count
    results["average_cot_brics_tanimoto"] = acc["cot_brics_tani"] / total_count
    results["average_cot_13c_accuracy"] = acc["cot_13c_acc_sum"] / total_count
    results["average_cot_1h_accuracy"] = acc["cot_1h_acc_sum"] / total_count
    results["average_cot_brics_accuracy"] = acc["cot_brics_acc_sum"] / total_count

    # Error Metrics Results
    if filtered_count_for_error_metrics > 0:
        for m in ["mae", "mse", "rmsd"]:
            for iso in ["13c", "1h"]:
                key = f"{m}_{iso}"
                metric_name = f"average_{m}_zero_padding_{iso.upper()}_nmr_shift"
                results[metric_name] = acc[key] / filtered_count_for_error_metrics
    else:
        for m in ["mae", "mse", "rmsd"]:
            for iso in ["13C", "1H"]:
                results[f"average_{m}_zero_padding_{iso}_nmr_shift"] = 0.0
                
    return results
    
def judge_task_from_answer_by_keyword(list_answer, dict_key_task_value_list_task_keyword=None):
    """
    根据list_answer[0]判断任务类型。
    修改规则: 
    1. 使用 nmr_forward=["Formula"], retrosyn=["Reactant", "F1"], syn=["Product"] 进行判断
    2. 如果都不匹配，兜底返回 nmr (通常是纯SMILES输出的结构解析任务)
    """
    # 默认关键词设置
    if dict_key_task_value_list_task_keyword is None:
        dict_key_task_value_list_task_keyword = {
            "nmr_forward": ["Formula"], # 修改：包含 Formula 视为 Forward 任务 (假设Forward输出包含分子式)
            "nmr": [],                  # 修改：兜底项改为 nmr (反推结构，通常只有SMILES)
            "retrosyn": ["Reactant", "F1"], 
            "syn": ["Product"]
        }
    
    if not list_answer:
        raise ValueError("list_answer不能为空")
    
    answer_text = str(list_answer[0])
    
    # 1. 优先判断有关键词的任务
    for task, keywords in dict_key_task_value_list_task_keyword.items():
        if not keywords: continue # 跳过空关键词的任务（兜底项）
        for kw in keywords:
            if kw in answer_text:
                return task
    
    # 2. 如果都没匹配上，返回关键词列表为空的任务 (nmr)
    for task, keywords in dict_key_task_value_list_task_keyword.items():
        if not keywords:
            return task
    
    # 如果没有定义空关键词的任务，默认抛出异常或者返回特定值
    raise ValueError("未匹配到任何任务，且未定义默认兜底任务（关键词为空列表的任务）")

def _replace_reactant_placeholders(content: str) -> str:
    """ 
    将文本中的 [ReactantN] 占位符替换为 SMILES 的点分隔表示。
    """
    def _reactant_repl(m: re.Match) -> str:
        idx = m.group(1)
        return "" if idx == "1" else "."
    
    content = re.sub(r"\[Reactant(\d+)\]", _reactant_repl, content)
    return content

def _replace_product_placeholders(content: str) -> str:
    """ 
    将文本中的 [ProductN] 占位符替换为 SMILES 的点分隔表示。
    
    参数:
        content: 包含占位符的文本
        
    返回值:
        str: 替换后的文本
        
    示例:
        输入: "[Product1].[Product2]"
        输出: ".[Product2]"  (第一个替换为空，第二个开始替换为.)
    """
    import re
    
    # 匹配 [Product] 或 [ProductN] (N是数字)
    pattern = r'\[Product\d*\]'
    
    # 查找所有匹配
    matches = list(re.finditer(pattern, content))
    
    if not matches:
        return content
    
    # 从后往前替换，避免索引问题
    result = content
    for i, match in enumerate(reversed(matches)):
        pos = match.start()
        end = match.end()
        
        # 第一个占位符（最后一个match）替换为空字符串
        # 其他占位符替换为 "."
        replacement = "" if i == len(matches) - 1 else "."
        result = result[:pos] + replacement + result[end:]
    
    return result

def _to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 1024, use_chirality: bool = False):
    """
    将 SMILES 转为 RDKit 分子并计算 Morgan 指纹。
    参数:
        smiles (str): SMILES 字符串
        radius (int): Morgan 指纹半径，默认 2
        n_bits (int): 指纹位数，默认 1024
        use_chirality (bool): 是否考虑手性信息
    返回:
        (mol, fp): 若解析失败返回 (None, None)
    """
    if smiles is None:
        return None, None
    s = smiles.strip()
    if not s:
        return None, None
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None, None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits, useChirality=use_chirality)
        return mol, fp
    except Exception:
        return None, None
    
def judge_single_nmr_answer(infer_answer, gt_answer, dechirality=True, check_length=5000, list_false_prefix=None):
    """
    判断单个NMR答案是否正确。
    """
    if list_false_prefix is None:
        list_false_prefix = ["F_"]
    
    if check_length is not None and len(infer_answer) > check_length:
        return False
    
    for prefix in list_false_prefix:
        if infer_answer.startswith(prefix):
            return False
    
    gt_mol = Chem.MolFromSmiles(gt_answer) if gt_answer else None
    infer_mol = Chem.MolFromSmiles(infer_answer) if infer_answer else None
    
    if gt_mol is None or infer_mol is None:
        return False
    
    if dechirality:
        gt_mol_copy = Chem.Mol(gt_mol)
        infer_mol_copy = Chem.Mol(infer_mol)
        Chem.RemoveStereochemistry(gt_mol_copy)
        Chem.RemoveStereochemistry(infer_mol_copy)
        gt_canonical = Chem.MolToSmiles(gt_mol_copy, canonical=True)
        infer_canonical = Chem.MolToSmiles(infer_mol_copy, canonical=True)
    else:
        gt_canonical = Chem.MolToSmiles(gt_mol, isomericSmiles=True, canonical=True)
        infer_canonical = Chem.MolToSmiles(infer_mol, isomericSmiles=True, canonical=True)
    
    return gt_canonical == infer_canonical

def remove_reaction_tags(smiles_with_tags):
    """
    移除SMILES字符串中的反应物/产物标签。
    
    处理规则:
        - 第一个标签(如[Reactant1]或[Product1])删除
        - 后续标签(如[Reactant2], [Reactant3]或[Product2], [Product3])替换为"."以连接多个分子
    
    参数:
        smiles_with_tags (str): 带标签的SMILES字符串
        
    返回值:
        str: 清理后的SMILES字符串
        
    示例:
        输入: "[Reactant1]CCO[Reactant2]CC(C)C"
        输出: "CCO.CC(C)C"
        输入: "[Product1]CCO[Product2]CC(C)C"
        输出: "CCO.CC(C)C"
    """
    if not isinstance(smiles_with_tags, str):
        return ""
    
    result = smiles_with_tags.strip()
    
    # 使用正则表达式查找所有标签
    import re
    pattern = r'\[(Reactant\d*|Product\d*)\]'  # 修改：支持 Reactant 和 Product，数字可选
    
    # 查找所有标签的位置
    matches = list(re.finditer(pattern, result))
    
    if not matches:
        # 没有标签，直接返回
        return result
    
    # 从后向前替换，避免索引变化
    for i in range(len(matches) - 1, -1, -1):
        match = matches[i]
        start, end = match.span()
        
        if i == 0:
            # 第一个标签：删除
            result = result[:start] + result[end:]
        else:
            # 后续标签：替换为"."
            result = result[:start] + "." + result[end:]
    
    return result

def judge_single_retrosyn_answer(infer_answer, gt_answer, dechirality=True, check_length=5000, list_false_prefix=None):
    """
    判断单个逆合成答案是否正确（基于 Morgan 指纹 Tanimoto==1.0 的等价判定）。

    参数:
        infer_answer (str): 推理得到的答案（SMILES 字符串，可能带标签，如 [Reactant1]SMILES1[Reactant2]SMILES2）
        gt_answer (str):    正确答案（可能带标签的SMILES字符串）
        dechirality (bool): 是否忽略手性信息。True=忽略（默认），False=考虑手性
        check_length (int or None): 长度阈值，超过此长度的答案视为错误，默认 5000；None 表示不检查
        list_false_prefix (list[str] or None): 以列表中任一前缀开头的答案视为错误，默认 ["F_"]

    返回值:
        bool: 推理答案是否正确（Tanimoto 相似度==1.0）
    """
    # --- 0) 默认参数与早期拦截 ---
    if list_false_prefix is None:
        list_false_prefix = ["F_"]

    # 长度检查（在移除标签之前检查原始长度）
    if check_length is not None and isinstance(infer_answer, str) and len(infer_answer) > check_length:
        return False

    # 错误前缀检查（移除标签后检查）
    if isinstance(infer_answer, str):
        infer_clean = remove_reaction_tags(infer_answer)
        for prefix in list_false_prefix:
            if infer_clean.startswith(prefix):
                return False
    else:
        return False

    # --- 1) 移除标签并提取产物 SMILES ---
    try:
        if not isinstance(gt_answer, str):
            return False
        
        # 移除真实答案中的标签
        gt_clean = remove_reaction_tags(gt_answer)
        
        # 移除推理答案中的标签
        infer_clean = remove_reaction_tags(infer_answer)
        
        # 如果真实答案包含">>"，提取产物部分
        if ">>" in gt_clean:
            gt_product = gt_clean.split(">>", 1)[0].strip()
        else:
            # 不含">>"，整个字符串视为产物
            gt_product = gt_clean.strip()
        
        # 检查是否为空
        if not gt_product or not infer_clean:
            return False
            
    except Exception:
        return False

    # --- 2) 计算 Morgan 指纹（dechirality 控制是否考虑手性） ---
    use_chirality = not dechirality  # dechirality=True -> 忽略手性 -> use_chirality=False
    _, gt_fp = _to_morgan_fp(gt_product, radius=2, n_bits=1024, use_chirality=use_chirality)
    _, infer_fp = _to_morgan_fp(infer_clean, radius=2, n_bits=1024, use_chirality=use_chirality)
    
    if gt_fp is None or infer_fp is None:
        return False

    # --- 3) 计算 Tanimoto 相似度并判定 ---
    try:
        tanimoto = DataStructs.FingerprintSimilarity(gt_fp, infer_fp)
        return tanimoto == 1.0
    except Exception:
        return False

def judge_single_syn_answer(infer_answer, gt_answer, dechirality=True, check_length=5000, list_false_prefix=None):
    """
    判断单个合成答案是否正确（基于 Morgan 指纹 Tanimoto==1.0 的等价判定）。

    参数:
        infer_answer (str): 推理得到的答案（SMILES 字符串，可能带标签，如 [Product1]SMILES1[Product2]SMILES2）
        gt_answer (str):    正确答案（可能带标签的SMILES字符串）
        dechirality (bool): 是否忽略手性信息。True=忽略（默认），False=考虑手性
        check_length (int or None): 长度阈值，超过此长度的答案视为错误，默认 5000；None 表示不检查
        list_false_prefix (list[str] or None): 以列表中任一前缀开头的答案视为错误，默认 ["F_"]

    返回值:
        bool: 推理答案是否正确（Tanimoto 相似度==1.0）
    """
    # --- 0) 默认参数与早期拦截 ---
    if list_false_prefix is None:
        list_false_prefix = ["F_"]

    # 长度检查（在移除标签之前检查原始长度）
    if check_length is not None and isinstance(infer_answer, str) and len(infer_answer) > check_length:
        return False

    # 错误前缀检查（移除标签后检查）
    if isinstance(infer_answer, str):
        infer_clean = remove_reaction_tags(infer_answer)
        for prefix in list_false_prefix:
            if infer_clean.startswith(prefix):
                return False
    else:
        return False

    # --- 1) 移除标签并提取产物 SMILES ---
    try:
        if not isinstance(gt_answer, str):
            return False
        
        # 移除真实答案中的标签
        gt_clean = remove_reaction_tags(gt_answer)
        
        # 移除推理答案中的标签
        infer_clean = remove_reaction_tags(infer_answer)
        
        # 如果真实答案包含">>"，提取产物部分（合成任务产物在右侧）
        if ">>" in gt_clean:
            gt_product = gt_clean.split(">>", 1)[1].strip()  # 注意：这里是 [1] 而不是 [0]
        else:
            # 不含">>"，整个字符串视为产物
            gt_product = gt_clean.strip()
        
        # 检查是否为空
        if not gt_product or not infer_clean:
            return False
            
    except Exception:
        return False

    # --- 2) 计算 Morgan 指纹（dechirality 控制是否考虑手性） ---
    use_chirality = not dechirality  # dechirality=True -> 忽略手性 -> use_chirality=False
    _, gt_fp = _to_morgan_fp(gt_product, radius=2, n_bits=1024, use_chirality=use_chirality)
    _, infer_fp = _to_morgan_fp(infer_clean, radius=2, n_bits=1024, use_chirality=use_chirality)
    
    if gt_fp is None or infer_fp is None:
        return False

    # --- 3) 计算 Tanimoto 相似度并判定 ---
    try:
        tanimoto = DataStructs.FingerprintSimilarity(gt_fp, infer_fp)
        return tanimoto == 1.0
    except Exception:
        return False

# ====================================================================
# 逆合成任务的Tanimoto计算辅助函数
# ====================================================================
def calculate_concatenated_fingerprint_similarity(
    smiles_list1: List[str],
    smiles_list2: List[str],
    dechirality: bool = True,
    fp_radius: int = 2,
    fp_nbits: int = 1024
) -> float:
    """
    方法1：串联指纹法 - 计算多分子集合的Tanimoto相似度（推荐）
    
    原理：
        将每个集合中所有分子的指纹串联起来，形成一个大指纹
        然后计算两个大指纹的Tanimoto相似度
    
    参数:
        smiles_list1: 第一个分子列表（ground truth）
        smiles_list2: 第二个分子列表（预测结果）
        dechirality: 是否去除手性
        fp_radius: Morgan指纹的半径，默认2
        fp_nbits: 每个分子指纹的位数，默认1024
    
    返回值:
        float: Tanimoto相似度 [0, 1]，失败返回0.0
    
    示例:
        gt = ["CCO", "CC(C)C"]
        pred = ["CCO", "CCC"]
        sim = calculate_concatenated_fingerprint_similarity(gt, pred)
    """
    try:
        # 生成第一个集合的串联指纹
        fp1_concat = None
        for smiles in smiles_list1:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return 0.0
            
            if dechirality:
                mol_copy = Chem.Mol(mol)
                Chem.RemoveStereochemistry(mol_copy)
                mol = mol_copy
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)
            
            if fp1_concat is None:
                fp1_concat = fp
            else:
                # 串联指纹（使用位或操作）
                fp1_concat = fp1_concat | fp
        
        # 生成第二个集合的串联指纹
        fp2_concat = None
        for smiles in smiles_list2:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return 0.0
            
            if dechirality:
                mol_copy = Chem.Mol(mol)
                Chem.RemoveStereochemistry(mol_copy)
                mol = mol_copy
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)
            
            if fp2_concat is None:
                fp2_concat = fp
            else:
                fp2_concat = fp2_concat | fp
        
        # 计算Tanimoto相似度
        if fp1_concat is not None and fp2_concat is not None:
            return DataStructs.TanimotoSimilarity(fp1_concat, fp2_concat)
        else:
            return 0.0
            
    except Exception:
        return 0.0

def calculate_average_pairwise_similarity(
    smiles_list1: List[str],
    smiles_list2: List[str],
    dechirality: bool = True,
    fp_radius: int = 2,
    fp_nbits: int = 1024
) -> float:
    """
    方法2：平均相似度法 - 最优配对后计算平均Tanimoto相似度
    
    原理：
        1. 计算所有可能的分子对之间的相似度矩阵
        2. 使用贪心算法找到最优配对
        3. 计算配对相似度的平均值
    
    参数:
        smiles_list1: 第一个分子列表
        smiles_list2: 第二个分子列表
        dechirality: 是否去除手性
        fp_radius: Morgan指纹的半径
        fp_nbits: 指纹位数
    
    返回值:
        float: 平均Tanimoto相似度 [0, 1]
    
    注意:
        如果两个列表长度不同，会对较短的列表进行惩罚
    """
    try:
        # 如果长度不同，直接返回0（严格模式）
        # 或者可以选择只计算较短列表的平均相似度（宽松模式）
        if len(smiles_list1) != len(smiles_list2):
            # 宽松模式：计算部分平均
            min_len = min(len(smiles_list1), len(smiles_list2))
            if min_len == 0:
                return 0.0
        
        # 生成所有分子的指纹
        fps1 = []
        for smiles in smiles_list1:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return 0.0
            
            if dechirality:
                mol_copy = Chem.Mol(mol)
                Chem.RemoveStereochemistry(mol_copy)
                mol = mol_copy
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)
            fps1.append(fp)
        
        fps2 = []
        for smiles in smiles_list2:
            mol = Chem.MolFromSmiles(smiles.strip())
            if mol is None:
                return 0.0
            
            if dechirality:
                mol_copy = Chem.Mol(mol)
                Chem.RemoveStereochemistry(mol_copy)
                mol = mol_copy
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_nbits)
            fps2.append(fp)
        
        # 计算相似度矩阵
        n1, n2 = len(fps1), len(fps2)
        sim_matrix = np.zeros((n1, n2))
        
        for i, fp1 in enumerate(fps1):
            for j, fp2 in enumerate(fps2):
                sim_matrix[i, j] = DataStructs.TanimotoSimilarity(fp1, fp2)
        
        # 贪心配对算法
        used1 = set()
        used2 = set()
        total_sim = 0.0
        pairs = min(n1, n2)
        
        for _ in range(pairs):
            # 找到最大相似度的未使用配对
            max_sim = -1
            best_i, best_j = -1, -1
            
            for i in range(n1):
                if i in used1:
                    continue
                for j in range(n2):
                    if j in used2:
                        continue
                    if sim_matrix[i, j] > max_sim:
                        max_sim = sim_matrix[i, j]
                        best_i, best_j = i, j
            
            if best_i != -1:
                used1.add(best_i)
                used2.add(best_j)
                total_sim += max_sim
        
        # 计算平均相似度
        # 如果长度不同，对未配对的分子进行惩罚（相似度为0）
        avg_sim = total_sim / max(n1, n2)
        
        return avg_sim
        
    except Exception:
        return 0.0

def calculate_retrosyn_tanimoto_similarity(
    gt_answer: str,
    pred_answer: str,
    method: str = "concatenated",
    dechirality: bool = True
) -> float:
    """
    计算逆合成任务的Tanimoto相似度（统一接口）
    
    参数:
        gt_answer: ground truth答案（格式：smiles1.smiles2）
        pred_answer: 预测答案（格式：smiles3.smiles4）
        method: 计算方法
            - "concatenated": 串联指纹法（默认，推荐）
            - "average": 平均相似度法（更精细）
        dechirality: 是否去除手性
    
    返回值:
        float: Tanimoto相似度 [0, 1]
    
    使用示例:
        sim = calculate_retrosyn_tanimoto_similarity(
            "CCO.CC(C)C",
            "CCO.CCC",
            method="concatenated"
        )
    """
    try:
        # 分割SMILES
        gt_parts = [s.strip() for s in gt_answer.split('.') if s.strip()]
        pred_parts = [s.strip() for s in pred_answer.split('.') if s.strip()]
        
        if not gt_parts or not pred_parts:
            return 0.0
        
        # 根据方法选择计算函数
        if method == "concatenated":
            return calculate_concatenated_fingerprint_similarity(
                gt_parts, pred_parts, dechirality=dechirality
            )
        elif method == "average":
            return calculate_average_pairwise_similarity(
                gt_parts, pred_parts, dechirality=dechirality
            )
        else:
            raise ValueError(f"不支持的方法: {method}")
            
    except Exception:
        return 0.0

#### 最主要的计算函数 ####
def test_model_predict_result(
    arrow_folder_path, 
    model_predict_result_txt_file_path,
    dict_key_task_value_list_task_keyword=None, 
    dechirality=True,
    check_length=5000,
    list_false_prefix=None,
    # --- 新增参数 ---
    with_answer_for_deta_count=True, 
    max_length_for_deta_count=5000*6
):
    """
    测试模型预测结果（集成 NMR Forward 逻辑）。
    
    参数:
        arrow_folder_path (str): Arrow 数据集文件夹路径
        model_predict_result_txt_file_path (str): 模型预测结果 txt 文件路径
        dict_key_task_value_list_task_keyword (dict): 任务关键词映射
        dechirality (bool): 是否忽略手性
        check_length (int): 长度检查阈值
        list_false_prefix (list): 错误前缀列表
        with_answer_for_deta_count (bool): [NMR Forward] 是否仅对有答案的样本计算delta count
        max_length_for_deta_count (int): [NMR Forward] 答案最大长度阈值，超过则过滤delta count
        
    返回:
        dict: 评估统计结果
    """
    if list_false_prefix is None:
        list_false_prefix = ["F_"]
    
    print(f"正在加载数据: {arrow_folder_path}")
    
    # 加载 GT (Question, CoT, Answer)
    list_gt_question, list_gt_cot, list_gt_answer = load_question_cot_answer_list_from_arrow_folder(arrow_folder_path)
    
    # 判断任务类型
    str_task = judge_task_from_answer_by_keyword(
        list_gt_answer, 
        dict_key_task_value_list_task_keyword=dict_key_task_value_list_task_keyword
    )
    print(f"检测到任务类型: {str_task}")
    
    # ==========================
    # 分支 1: NMR Forward 任务
    # ==========================
    if str_task == "nmr_forward":
        print(f"正在执行 NMR Forward 评估 (使用 calc_global_stats_from_memory)")
        print(f"正在加载预测文件: {model_predict_result_txt_file_path}")
        if with_answer_for_deta_count:
            print("  - 启用过滤: 仅统计有答案的样本的 delta count")
        print(f"  - 答案长度阈值: {max_length_for_deta_count}")
        
        list_dict_main = []
        
        # 逐行读取预测文件 (需要同时提取 CoT 和 Answer)
        gen = read_model_predictions(model_predict_result_txt_file_path)
        
        for idx, (gt_q, gt_c, gt_a) in enumerate(zip(list_gt_question, list_gt_cot, list_gt_answer)):
            try:
                line = next(gen)
                str_gen_raw = extract_str_generate_result_without_prompt(line)
                
                # 提取 Gen CoT 和 Answer
                gen_c = extract_str_cot(str_gen_raw)
                gen_a = extract_str_answer(str_gen_raw)
                
                # 构建 CoT 字典
                dict_cot_gt = str_cot_to_dict(gt_c) if isinstance(gt_c, str) else {"depth": {"13c": {}, "1h": {}}, "brics": {}}
                dict_cot_gen = str_cot_to_dict(gen_c)
                
                # 计算统计信息 (传递参数)
                dict_answer_info = stat_nmr_forward_answer_info(
                    gt_a, 
                    gen_a, 
                    with_answer_for_deta_count=with_answer_for_deta_count, 
                    max_length_for_deta_count=max_length_for_deta_count
                )
                
                # 2. CoT Stats (Structure Validity)
                dict_cot_info = stat_dict_cot_generate_info(dict_cot_gt, dict_cot_gen, dechirality=dechirality)
                
                # 汇总单条结果
                list_dict_main.append({
                    "answer": dict_answer_info,
                    "cot": dict_cot_info
                })
                
            except StopIteration:
                print(f"警告: 预测文件行数少于 GT 样本数，在第 {idx} 行停止。")
                break
            except Exception as e:
                print(f"警告: 处理第 {idx} 行时出错: {e}")
                # 添加空结果以保持计数，避免全局统计偏差太大
                list_dict_main.append({"answer": {}, "cot": {}})
        
        # 计算全局指标 (包含 delta count = 0 的过滤逻辑)
        stats = calc_global_stats_from_memory(list_dict_main, delta_1h_count=0, delta_13c_count=0)
        
        # 打印部分关键结果
        print(f"\nNMR Forward 评估摘要:")
        print(f"Avg 13C MAE (Filter |Δ|=0): {stats.get('average_mae_zero_padding_13C_nmr_shift', 0):.4f}")
        print(f"Avg 1H MAE (Filter |Δ|=0):  {stats.get('average_mae_zero_padding_1H_nmr_shift', 0):.4f}")
        print(f"Valid Samples for Error:    {stats.get('valid_samples_for_error_metrics', 0)}")
        
        return stats

    # ==========================
    # 分支 2: 其他任务 (NMR, Retrosyn, Syn)
    # ==========================
    else:
        # 如果是逆合成，预处理 GT
        if str_task == "retrosyn":
            print("逆合成任务：正在预处理ground truth答案...")
            list_gt_answer = [_replace_reactant_placeholders(ans) for ans in list_gt_answer]
        
        # 加载预测结果 (只需要 Answer 部分)
        print(f"正在加载预测结果: {model_predict_result_txt_file_path}")
        list_generate_answer = read_list_generate_result_from_txt(model_predict_result_txt_file_path)
        
        # 如果是逆合成，预处理 Gen
        if str_task == "retrosyn":
            print("逆合成任务：正在预处理预测答案...")
            list_generate_answer = [_replace_reactant_placeholders(ans) for ans in list_generate_answer]
        
        # 长度检查
        if len(list_gt_answer) != len(list_generate_answer):
            print(f"警告: 答案数量不匹配 (GT: {len(list_gt_answer)}, Pred: {len(list_generate_answer)})")
        
        # 计算常规指标
        stats = calculate_smiles_validity_accuracy_from_ref_list_and_pred_list(
            list_gt_answer, 
            list_generate_answer,
            str_task=str_task,
            dechirality=dechirality,
            check_length=check_length,
            list_false_prefix=list_false_prefix
        )
        
        # 打印结果
        print(f"\n{'='*50}")
        print(f"任务类型: {str_task.upper()}")
        print(f"{'='*50}")
        for key, value in stats.items():
            if value is not None:
                print(f"{key}: {value:.4f}")
        print(f"{'='*50}\n")
        
        return stats
        
def evaluate_multiple_test_files(save_folder_path, list_save_file_name, dict_key_test_arrow_folder_path_value_dataset_name, with_answer_for_deta_count=True, max_length_for_deta_count=5000 * 6):
    """
    批量评估多个测试文件和预测结果文件。
    
    参数:
        save_folder_path: 存储txt结果文件的文件夹路径
        list_save_file_name: 结果文件的文件名列表（不包含.txt后缀）
        dict_key_test_arrow_folder_path_value_dataset_name: 字典，键是测试文件夹路径，值是数据集名称
        
        
    返回值:
        all_stats: 包含所有评估统计结果的字典，键为测试文件夹路径
    """
    # 从字典中提取arrow文件夹路径列表
    list_test_arrow_folder_path = list(dict_key_test_arrow_folder_path_value_dataset_name.keys())
    
    # 检查list_test_arrow_folder_path和list_save_file_name数量是否对应
    if len(list_test_arrow_folder_path) != len(list_save_file_name):
        raise ValueError(f"测试文件夹列表和保存文件名列表的长度不匹配, len(list_test_arrow_folder_path):{len(list_test_arrow_folder_path)}, len(list_save_file_name):{len(list_save_file_name)}")
    
    # 通过save_folder_path和list_save_file_name构造完整的.txt文件路径列表
    # 注意添加.txt后缀
    list_model_predict_result_txt_file_path = [os.path.join(save_folder_path, file_name + '.txt') for file_name in list_save_file_name]
    
    # 存储所有统计结果的字典
    all_stats = {}
    
    # 逐对调用test_model_predict_result获得统计值
    for i, (arrow_folder_path, txt_file_path) in enumerate(zip(list_test_arrow_folder_path, list_model_predict_result_txt_file_path)):
        txt_file_path = remove_extra_txt_extension(txt_file_path)
        print(f"\n评估文件 {i+1}/{len(list_test_arrow_folder_path)}:")
        print(f"Arrow文件夹: {arrow_folder_path}")
        print(f"预测结果文件: {txt_file_path}")
        
        # 调用test_model_predict_result函数获取统计结果
        stats = test_model_predict_result(arrow_folder_path, txt_file_path, with_answer_for_deta_count=with_answer_for_deta_count, max_length_for_deta_count=max_length_for_deta_count)
        
        # 将统计结果存储在字典中，使用arrow文件夹路径作为键
        all_stats[arrow_folder_path] = stats
    
    print("\n所有文件的评估已完成")
    return all_stats

def batch_evaluate_multiple_test_files(save_folder_path, dict_key_test_arrow_folder_path_value_dataset_name, save_result_as_json_and_csv=True, save_result_filename_prefix=None, with_answer_for_deta_count=True, max_length_for_deta_count=5000*6):
    """
    批量评估多个权重模型对多个测试文件的预测结果。
    
    更新：支持单 Checkpoint 模式（直接在目录下寻找 txt 文件）。
    """
    # 1. 尝试获取子目录（原逻辑：寻找 checkpoint-xxx 文件夹）
    try:
        list_result_folder_path = get_subdirectories(save_folder_path)
    except ValueError as e:
        list_result_folder_path = []

    # 2. [新增] 检查当前目录下是否直接包含 txt 文件（单 Checkpoint 模式）
    # 如果没有子目录，或者你想兼容两种情况
    if not list_result_folder_path:
        files_in_root = get_files_with_suffix(save_folder_path, ".txt", traverse_sub_folder=False)
        if files_in_root:
            print(f"检测到单 Checkpoint 模式：在 {save_folder_path} 下直接发现 {len(files_in_root)} 个 txt 文件")
            # 将当前目录本身作为唯一的结果文件夹
            list_result_folder_path = [save_folder_path]
        else:
            print(f"警告：在 {save_folder_path} 中既没有子文件夹，也没有直接的 .txt 结果文件。")
            return {}

    # 根据路径获得训练条件信息
    train_info = os.path.basename(os.path.normpath(save_folder_path))
    dict_key_int_train_step_value_one_result_folder_dict_stat = {}
    
    # 遍历每个 Checkpoint/Step 的文件夹
    for result_folder_path in list_result_folder_path:
        # 如果是单 Checkpoint 模式，result_folder_path 就是 save_folder_path
        # 为了区分，如果是同一路径，我们可以将 step 命名为 "final" 或者 "0" 或者直接用文件夹名
        if result_folder_path == save_folder_path:
             train_step = "final" # 或者使用 "0"
        else:
             train_step = os.path.basename(os.path.normpath(result_folder_path))
             
        print(f"\n正在处理 Checkpoint/Step: {train_step} (路径: {result_folder_path})")
        
        # 获取该文件夹下的所有 txt 结果文件
        list_txt_file_path_unsorted = get_files_with_suffix(result_folder_path, ".txt", traverse_sub_folder=False)
        if not list_txt_file_path_unsorted:
            print(f"  警告: {result_folder_path} 中没有 .txt 文件，跳过")
            continue

        # 对文件进行排序，确保与 dataset 顺序对应
        list_txt_file_name_sorted_index = []
        valid_files = True
        
        # 收集有效的 txt 文件路径
        valid_txt_paths = []
        valid_indices = []

        for txt_file_path_unsorted in list_txt_file_path_unsorted:
            txt_file_name_with_suffix = os.path.basename(txt_file_path_unsorted)
            
            # 使用修改后的函数提取数据集名称
            dataset_name = extract_dataset_name_from_txt_file_name_with_suffix(txt_file_name_with_suffix)
            
            if dataset_name is None:
                # print(f"  跳过: 无法识别的文件名 {txt_file_name_with_suffix}")
                continue
            
            # 检查是否是我们关心的 benchmark 数据集
            txt_file_name_index = get_key_index_by_value(dict_key_test_arrow_folder_path_value_dataset_name, dataset_name)
            
            if txt_file_name_index is not None:
                valid_indices.append(txt_file_name_index)
                valid_txt_paths.append(txt_file_path_unsorted)
            else:
                # print(f"  跳过: 数据集 {dataset_name} 不在目标列表中")
                pass
        
        if not valid_txt_paths:
            print(f"  警告: 在 {result_folder_path} 中未找到任何匹配目标数据集的 txt 文件")
            continue
            
        # 根据 index 排序文件路径
        # zip 排序技巧：将 (index, path) 组合，按 index 排序，然后解压
        list_txt_file_path_sorted = [x for _, x in sorted(zip(valid_indices, valid_txt_paths))]

        # 调用核心评估函数
        one_result_folder_dict_stat = evaluate_multiple_test_files(
            result_folder_path, 
            list_txt_file_path_sorted, 
            dict_key_test_arrow_folder_path_value_dataset_name,
            with_answer_for_deta_count=with_answer_for_deta_count,
            max_length_for_deta_count=max_length_for_deta_count,
        )
        
        # 替换键名为 Dataset Name
        dict_key_int_train_step_value_one_result_folder_dict_stat[train_step] = replace_keys_with_values(
            one_result_folder_dict_stat, 
            dict_key_test_arrow_folder_path_value_dataset_name
        )
    
    if (save_result_filename_prefix is None) and (save_result_as_json_and_csv):
        save_result_filename_prefix = train_info

    # 如果需要保存结果为JSON和CSV
    if save_result_as_json_and_csv and dict_key_int_train_step_value_one_result_folder_dict_stat:
        print("\n正在保存评估汇总结果...")
        try:
            generated_files_list = save_evaluation_results_as_json_and_csv(
                dict_key_int_train_step_value_one_result_folder_dict_stat,
                save_folder_path,
                dict_key_test_arrow_folder_path_value_dataset_name,
                save_result_filename_prefix
            )
            print(f"成功生成以下文件:")
            for f in generated_files_list:
                print(f"  - {f}")
                
        except Exception as e:
            print(f"保存结果时出错: {e}")
            import traceback
            traceback.print_exc()
    
    return dict_key_int_train_step_value_one_result_folder_dict_stat 
