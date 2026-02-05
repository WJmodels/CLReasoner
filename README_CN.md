# CLReasoner (Chain-of-Thought Learning for Chemical Reasoning)
---

## 📋 目录

- [支持的任务](#-支持的任务)
- [环境安装](#️-环境安装)
- [数据准备](#-数据准备)
- [模型训练](#-模型训练)
- [模型推理与评估](#-模型推理与评估)
- [化学感知自洽性(Self-Consistency)](#-化学感知自洽性self-consistency)
- [项目结构](#-项目结构)

---

## 🔬 支持的任务

### 1. 逆合成预测 (Retrosynthesis Prediction)
- **输入**: 目标产物分子
- **输出**: 反应物
- **评估指标**: Top-K准确率, Tanimoto相似度

### 2. 正向合成预测 (Forward Synthesis)
- **输入**: 反应物
- **输出**: 产物分子
- **评估指标**: Top-K准确率, Tanimoto相似度

### 3. 反应补全 (Paired reactants design)
- **输入**: 反应物1
- **输出**: 反应物2,产物分子
- **评估指标**: Top-K准确率, Tanimoto相似度

### 4. NMR结构解析 (Structure elucidation)
- **输入**: NMR谱图数据 (¹H-NMR, ¹³C-NMR)
- **输出**: 分子结构 (SMILES格式)
- **评估指标**: Top-K准确率, Tanimoto相似度

### 5. NMR结构解析 (NMR Structure Elucidation)
- **输入**: 分子结构 (SMILES格式)
- **输出**: ¹H-NMR, ¹³C-NMR
- **评估指标**: MAE, RMSD, Delta Count, Tanimoto相似度



---

## 🛠️ 环境安装

### 系统要求

- **操作系统**: 推荐Linux (Ubuntu 22.04)
- **Python**: 3.11
- **CUDA**: 12.4 (用于GPU训练和推理)
- **显存**: 24+GB

### 使用environment.yml文件

```bash
# 从项目根目录安装
conda env create -f environment.yml
conda activate clreasoner
```

### 验证安装

```bash
# 验证PyTorch和CUDA
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}')"

# 验证Unsloth
# 推荐使用和我们相同的unsloth版本，不同版本unsloth训练出的模型有较大差异

python -c "from unsloth import FastLanguageModel; print('Unsloth安装成功')"

# 验证vLLM
python -c "from vllm import LLM; print('vLLM安装成功')"

# 验证RDKit
python -c "from rdkit import Chem; print('RDKit安装成功')"
```

---

## 📊 数据准备

### 数据格式要求

训练和测试数据需要使用 **Arrow格式** (HuggingFace Datasets格式)。每个样本包含三个字段:

```python
{
    "question": str,  # 输入问题/任务描述
    "cot": str,       # 思维链推理过程
    "answer": str     # 最终答案
}
```

### 数据目录结构

```
your_data_folder/
├── data_file_1.arrow
├── data_file_2.arrow
└── data_file_3.arrow
```

### 数据集准备示例

```python
from datasets import Dataset
import pandas as pd

# 准备数据
data = {
    "question": ["问题1", "问题2", ...],
    "cot": ["推理过程1", "推理过程2", ...],
    "answer": ["答案1", "答案2", ...]
}

# 创建Dataset
dataset = Dataset.from_pandas(pd.DataFrame(data))

# 保存为Arrow格式
dataset.save_to_disk("./train_data")
```

### 示例数据

项目的 `sample/` 文件夹中提供了示例数据:

```
sample/
└── chemistry_aware_self_consistency_sample_ground_truth/
    └── RP_test_arrow/
        └── test_data.arrow
```


---

## 🚀 模型训练

### 训练脚本: `train/train_single_gpu.py`

#### 主要配置参数

在训练脚本的开头,你需要修改以下关键配置:

```python
# ============ 训练数据配置 ============
TRAIN_DATA_DIR = r"/path/to/your/train/data"  # 训练数据文件夹路径

# ============ 训练超参数 ============
NUM_EPOCHS = 20              # 训练轮数
BATCH_SIZE = 64              # 批次大小
GRADIENT_ACCUMULATION_STEPS = 4  # 梯度累积步数
LEARNING_RATE = 2e-4         # 学习率
WARMUP_STEPS = 5             # 预热步数
WEIGHT_DECAY = 0.01          # 权重衰减
max_grad_norm = 0.15         # 梯度裁剪
SAVE_STEPS = 8000            # 每隔多少步保存一次检查点

# ============ 模型配置 ============
MODEL_NAME = r"/path/to/base/model"  # 预训练模型路径
MAX_SEQ_LENGTH = 5000        # 最大序列长度
DTYPE = torch.bfloat16       # 数据类型

# ============ LoRA配置 ============
LORA_R = 512                 # LoRA秩
LORA_ALPHA = 2048            # LoRA alpha参数
LORA_DROPOUT = 0             # LoRA dropout
LORA_TARGET_MODULES = [      # 目标模块
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# ============ GPU配置 ============
GPU_IDS = [0]                # 使用的GPU编号列表
```

#### 启动训练

```bash
# 激活环境
conda activate clreasoner

# 进入train目录
cd train

# 开始训练
python train_single_gpu.py
```

#### 训练输出

训练过程中会生成以下文件(推荐在新建的文件夹中使用train_single_gpu.py脚本):

```
train/
├── outputs/                    # 模型检查点
│   ├── checkpoint-8000/
│   ├── checkpoint-16000/
│   └── ...
├── final_model/                # 最终模型
├── training_logs/              # 训练日志
│   ├── terminal_output_YYYYMMDD_HHMMSS.log
│   └── training_log_YYYYMMDD_HHMMSS.txt
├── hf_cache/                   # HuggingFace缓存
└── temp/                       # 临时文件
```

#### 恢复训练

如果训练中断,可以从检查点恢复:

```python
# 在train_single_gpu.py中修改
trainer_stats = trainer.train(
    resume_from_checkpoint = r"/path/to/checkpoint-xxxxx"
)
```


#### WandB日志配置 (可选)

```python
# 设置WandB API Key
MY_WANDB_KEY = "your_wandb_api_key"  # 如果不使用WandB,设置为None
```

---

## 🔍 模型推理与评估

### 推理脚本: `test/test.py`

#### 单检查点推理完整流程

使用 `single_checkpoint_infer_pipeline` 函数进行完整的推理→转换→评估流程:

```python
from test import single_checkpoint_infer_pipeline

# 配置参数
checkpoint_folder_path = r"/path/to/checkpoint-xxxxx"
ground_truth_arrow_folder_path = r"/path/to/test/data/arrow"
dataset_name = "test_dataset_name"
output_folder_path = r"/path/to/output/folder"
result_name = "experiment_result_name"
stat_metrics_scription_file_path = r"./stat_metrics_scription.py"

# 执行推理和评估
single_checkpoint_infer_pipeline(
    checkpoint_folder_path=checkpoint_folder_path,
    ground_truth_arrow_folder_path=ground_truth_arrow_folder_path,
    dataset_name=dataset_name,
    output_folder_path=output_folder_path,
    result_name=result_name,
    stat_metrics_scription_file_path=stat_metrics_scription_file_path,
    beam_width=1,              # Beam Search宽度 (>1的推理会导致推理速度显著变慢)
    max_tokens=5000,           # 最大生成token数
    temperature=0,             # 采样温度
    top_p=0.95,               # Nucleus采样
    top_k=20,                 # Top-K采样
    adjust_decimal_places=False,  # 是否调整NMR数据小数位数
    c_shift_decimal_place=1,      # 13C-NMR小数位数
    h_shift_decimal_place=2       # 1H-NMR小数位数
)
```

#### 推理输出文件

```
output_folder_path/
└── result_name/
    ├── result_name_dataset_name.jsonl      # JSONL格式推理结果(可用于Chemistry-Aware Self-Consistency)
    ├── result_name_dataset_name.txt        # TXT格式推理结果
    ├── evaluation_result.json              # 完整评估指标(JSON)
    ├── evaluation_result_formatted.csv     # 格式化评估表格
    └── evaluation_result_accuracy.csv      # 准确率表格
```

#### 运行推理

```bash
# 激活环境
conda activate clreasoner

# 进入test目录
cd test

# 运行推理脚本
python test.py
```

---

## 🎲 化学感知自洽性(Self-Consistency)

### 原理说明

化学感知自洽性(Chemistry-Aware Self-Consistency)是一种多模型集成技术,通过以下步骤提升预测准确性:

1. **多模型推理**: 使用多个训练检查点对同一问题生成多个候选答案
3. **答案过滤**: 基于任务的答案合理性过滤
2. **化学感知投票**: 基于过滤后标准化SMILES进行投票
4. **Top-K聚合**: 输出投票后的Top-K最优答案

### 使用脚本: `test/chemistry_aware_self_consistency.py`

#### 准备推理结果

首先,使用不同的模型检查点对测试集进行推理,生成多个JSONL文件:

```bash
# 模型1推理
python test.py  # 生成 model1_result.jsonl

# 模型2推理
python test.py  # 生成 model2_result.jsonl

# ... 更多模型
```

#### 配置Self-Consistency

```python
from chemistry_aware_self_consistency import cot_sc_multi_jsonl_topk_result_to_dict_key_str_topk_value_float_acc

# ============ 配置参数 ============
# Ground Truth数据路径
arrow_folder_path = r"/path/to/ground_truth/arrow/folder"

# 推理结果文件列表 (按优先级排序,越靠前优先级越高)
list_jsonl_file_path = [
    r"/path/to/model1_result.jsonl",
    r"/path/to/model2_result.jsonl",
    r"/path/to/model3_result.jsonl",
    # ... 添加更多模型结果
]

# 输出配置
save_folder_path = r"/path/to/output/folder"
save_folder_name = "self_consistency_results"

# Self-Consistency参数
topk = 10                      # 计算Top-K准确率的K值
ignore_order = True            # 是否忽略混合物顺序
dechirality = True             # 是否去除立体化学信息
force_to_topk = None           # 强制每个问题的候选数量 (None表示不限制)
check_length = 5000            # 最大SMILES长度
num_process = 8                # 并行处理进程数
extract_key = "text"           # JSONL中提取的字段名
fix_escaped_backslash = True   # 是否修复转义的反斜杠
check_nmr_formula = True       # 是否检查NMR分子式一致性 (仅用于NMR任务)

# 执行Self-Consistency
final_metrics = cot_sc_multi_jsonl_topk_result_to_dict_key_str_topk_value_float_acc(
    arrow_folder_path=arrow_folder_path,
    list_jsonl_file_path=list_jsonl_file_path,
    save_folder_path=save_folder_path,
    save_folder_name=save_folder_name,
    topk=topk,
    ignore_order=ignore_order,
    dechirality=dechirality,
    force_to_topk=force_to_topk,
    check_length=check_length,
    num_process=num_process,
    extract_key=extract_key,
    fix_escaped_backslash=fix_escaped_backslash,
    check_nmr_formula=check_nmr_formula
)

print("Self-Consistency评估完成!")
print(f"最终指标: {final_metrics}")
```

#### 运行Self-Consistency

```bash
# 激活环境
conda activate clreasoner

# 进入test目录
cd test

# 运行脚本
python chemistry_aware_self_consistency.py
```

#### 输出结果

```
save_folder_path/
└── save_folder_name/
    ├── sc_results.json                    # 投票详细结果
    ├── summary_metrics.json               # 汇总评估指标
    ├── sc_results_formatted.csv           # CSV格式结果
    └── voting_statistics.json             # 投票统计信息
```

#### 示例运行

项目提供了完整的示例:

```bash
cd test

# 查看示例数据
ls ../sample/chemistry_aware_self_consistency_sample_ground_truth/
ls ../sample/chemistry_aware_self_consistency_sample_infer_result_for_sc/

# 运行示例 (代码中的__main__部分)
python chemistry_aware_self_consistency.py
```

---

## 📁 项目结构

```
CLReasoner/
├── train/                                # 训练模块
│   └── train_single_gpu.py              # 单GPU/多GPU训练脚本
│
├── test/                                 # 测试与评估模块
│   ├── test.py                          # 推理主脚本
│   ├── chemistry_aware_self_consistency.py  # Self-Consistency脚本
│   └── stat_metrics_scription.py        # 评估指标计算脚本
│
├── sample/                              # 示例数据
│   ├── chemistry_aware_self_consistency_sample_ground_truth/
│   ├── chemistry_aware_self_consistency_sample_infer_result_for_sc/
│   └── chemistry_aware_self_consistency_sample_output/
│
├── environment.yml                      # Conda环境配置文件
└── README_CN.md                         # 中文说明文档 (本文件)
```

---

## ⭐ Star History

如果这个项目对你有帮助,请给我们一个Star⭐!

