# CLReasoner (Chain-of-Thought Learning for Chemical Reasoning)

**For paper:** *Mediating chemistry and large language models via a chemical language chain-of-thought strategy*


---

## 📋 Table of Contents

- [Supported Tasks](#-supported-tasks)
- [Environment Setup](#️-environment-setup)
- [Data Preparation](#-data-preparation)
- [Model Training](#-model-training)
- [Model Inference and Evaluation](#-model-inference-and-evaluation)
- [Chemistry-Aware Self-Consistency](#-chemistry-aware-self-consistency)
- [Project Structure](#-project-structure)

---

## 🔬 Supported Tasks

### 1. Retrosynthesis Prediction
- **Input**: Target product molecule
- **Output**: Reactants
- **Evaluation Metrics**: Top-K accuracy, Tanimoto similarity

### 2. Forward Synthesis
- **Input**: Reactants
- **Output**: Product molecule
- **Evaluation Metrics**: Top-K accuracy, Tanimoto similarity

### 3. Paired Reactants Design
- **Input**: Reactant 1
- **Output**: Reactant 2, Product molecule
- **Evaluation Metrics**: Top-K accuracy, Tanimoto similarity

### 4. Structure Elucidation
- **Input**: NMR spectroscopy data (¹H-NMR, ¹³C-NMR)
- **Output**: Molecular structure (SMILES format)
- **Evaluation Metrics**: Top-K accuracy, Tanimoto similarity

### 5. NMR Structure Elucidation
- **Input**: Molecular structure (SMILES format)
- **Output**: ¹H-NMR, ¹³C-NMR
- **Evaluation Metrics**: MAE, RMSD, Delta Count, Tanimoto similarity

---

## 🛠️ Environment Setup

### System Requirements

- **Operating System**: Linux recommended (Ubuntu 22.04)
- **Python**: 3.11
- **CUDA**: 12.4 (for GPU training and inference)
- **GPU Memory**: 24+GB

### Using environment.yml File
```bash
# Install from project root directory
conda env create -f environment.yml
conda activate clreasoner
```

### Verify Installation
```bash
# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Verify Unsloth
# We recommend using the same Unsloth version (unsloth==2025.10.10) as ours, as models trained with different Unsloth versions can exhibit significant differences in performance
python -c "from unsloth import FastLanguageModel; print('Unsloth installed successfully')"

# Verify vLLM
python -c "from vllm import LLM; print('vLLM installed successfully')"

# Verify RDKit
python -c "from rdkit import Chem; print('RDKit installed successfully')"
```

---

## 📊 Data Preparation

### Data Format Requirements

Training and testing data must use **Arrow format** (HuggingFace Datasets format). Each sample contains three fields:
```python
{
    "question": str,  # Input question/task description
    "cot": str,       # Chain-of-thought reasoning process
    "answer": str     # Final answer
}
```

### Data Directory Structure
```
your_data_folder/
├── data_file_1.arrow
├── data_file_2.arrow
└── data_file_3.arrow
```

### Dataset Preparation Example
```python
from datasets import Dataset
import pandas as pd

# Prepare data
data = {
    "question": ["Question 1", "Question 2", ...],
    "cot": ["Reasoning process 1", "Reasoning process 2", ...],
    "answer": ["Answer 1", "Answer 2", ...]
}

# Create Dataset
dataset = Dataset.from_pandas(pd.DataFrame(data))

# Save in Arrow format
dataset.save_to_disk("./train_data")
```

### Sample Data

Sample data is provided in the `sample/` folder:
```
sample/
└── chemistry_aware_self_consistency_sample_ground_truth/
    └── RP_test_arrow/
        └── test_data.arrow
```

---

## 🚀 Model Training

### Training Script: `train/train_single_gpu.py`

#### Main Configuration Parameters

At the beginning of the training script, you need to modify the following key configurations:
```python
# ============ Training Data Configuration ============
TRAIN_DATA_DIR = r"/path/to/your/train/data"  # Training data folder path

# ============ Training Hyperparameters ============
NUM_EPOCHS = 20              # Number of training epochs
BATCH_SIZE = 64              # Batch size
GRADIENT_ACCUMULATION_STEPS = 4  # Gradient accumulation steps
LEARNING_RATE = 2e-4         # Learning rate
WARMUP_STEPS = 5             # Warmup steps
WEIGHT_DECAY = 0.01          # Weight decay
max_grad_norm = 0.15         # Gradient clipping
SAVE_STEPS = 8000            # Save checkpoint every N steps

# ============ Model Configuration ============
MODEL_NAME = r"/path/to/base/model"  # Pre-trained model path
MAX_SEQ_LENGTH = 5000        # Maximum sequence length
DTYPE = torch.bfloat16       # Data type

# ============ LoRA Configuration ============
LORA_R = 512                 # LoRA rank
LORA_ALPHA = 2048            # LoRA alpha parameter
LORA_DROPOUT = 0             # LoRA dropout
LORA_TARGET_MODULES = [      # Target modules
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# ============ GPU Configuration ============
GPU_IDS = [0]                # List of GPU IDs to use
```

#### Start Training
```bash
# Activate environment
conda activate clreasoner

# Enter train directory
cd train

# Start training
python train_single_gpu.py
```

#### Training Output

During training, the following files will be generated (it's recommended to use the train_single_gpu.py script in a newly created folder):
```
train/
├── outputs/                    # Model checkpoints
│   ├── checkpoint-8000/
│   ├── checkpoint-16000/
│   └── ...
├── final_model/                # Final model
├── training_logs/              # Training logs
│   ├── terminal_output_YYYYMMDD_HHMMSS.log
│   └── training_log_YYYYMMDD_HHMMSS.txt
├── hf_cache/                   # HuggingFace cache
└── temp/                       # Temporary files
```

#### Resume Training

If training is interrupted, you can resume from a checkpoint:
```python
# Modify in train_single_gpu.py
trainer_stats = trainer.train(
    resume_from_checkpoint = r"/path/to/checkpoint-xxxxx"
)
```

#### WandB Logging Configuration (Optional)
```python
# Set WandB API Key
MY_WANDB_KEY = "your_wandb_api_key"  # Set to None if not using WandB
```

---

## 🔍 Model Inference and Evaluation

### Inference Script: `test/test.py`

#### Single Checkpoint Inference Complete Pipeline

Use the `single_checkpoint_infer_pipeline` function for complete inference → conversion → evaluation pipeline:
```python
from test import single_checkpoint_infer_pipeline

# Configuration parameters
checkpoint_folder_path = r"/path/to/checkpoint-xxxxx"
ground_truth_arrow_folder_path = r"/path/to/test/data/arrow"
dataset_name = "test_dataset_name"
output_folder_path = r"/path/to/output/folder"
result_name = "experiment_result_name"
stat_metrics_scription_file_path = r"./stat_metrics_scription.py"

# Execute inference and evaluation
single_checkpoint_infer_pipeline(
    checkpoint_folder_path=checkpoint_folder_path,
    ground_truth_arrow_folder_path=ground_truth_arrow_folder_path,
    dataset_name=dataset_name,
    output_folder_path=output_folder_path,
    result_name=result_name,
    stat_metrics_scription_file_path=stat_metrics_scription_file_path,
    beam_width=1,              # Beam search width (>1 significantly slows down inference)
    max_tokens=5000,           # Maximum tokens to generate
    temperature=0,             # Sampling temperature
    top_p=0.95,               # Nucleus sampling
    top_k=20,                 # Top-K sampling
    adjust_decimal_places=False,  # Whether to adjust NMR decimal places
    c_shift_decimal_place=1,      # 13C-NMR decimal places
    h_shift_decimal_place=2       # 1H-NMR decimal places
)
```

#### Inference Output Files
```
output_folder_path/
└── result_name/
    ├── result_name_dataset_name.jsonl      # JSONL format inference results (can be used for Chemistry-Aware Self-Consistency)
    ├── result_name_dataset_name.txt        # TXT format inference results
    ├── evaluation_result.json              # Complete evaluation metrics (JSON)
    ├── evaluation_result_formatted.csv     # Formatted evaluation table
    └── evaluation_result_accuracy.csv      # Accuracy table
```

#### Run Inference
```bash
# Activate environment
conda activate clreasoner

# Enter test directory
cd test

# Run inference script
python test.py
```

---

## 🎲 Chemistry-Aware Self-Consistency

### Principle

Chemistry-Aware Self-Consistency is a multi-model ensemble technique that improves prediction accuracy through the following steps:

1. **Multi-model Inference**: Generate multiple candidate answers for the same question using multiple training checkpoints
2. **Answer Filtering**: Filter based on task-specific answer validity
3. **Chemistry-Aware Voting**: Vote based on filtered and standardized SMILES
4. **Top-K Aggregation**: Output Top-K optimal answers after voting

### Using Script: `test/chemistry_aware_self_consistency.py`

#### Prepare Inference Results

First, perform inference on the test set using different model checkpoints to generate multiple JSONL files:
```bash
# Model 1 inference
python test.py  # Generates model1_result.jsonl

# Model 2 inference
python test.py  # Generates model2_result.jsonl

# ... More models
```

#### Configure Self-Consistency
```python
from chemistry_aware_self_consistency import cot_sc_multi_jsonl_topk_result_to_dict_key_str_topk_value_float_acc

# ============ Configuration Parameters ============
# Ground truth data path
arrow_folder_path = r"/path/to/ground_truth/arrow/folder"

# List of inference result files (ordered by priority, higher priority first)
list_jsonl_file_path = [
    r"/path/to/model1_result.jsonl",
    r"/path/to/model2_result.jsonl",
    r"/path/to/model3_result.jsonl",
    # ... Add more model results
]

# Output configuration
save_folder_path = r"/path/to/output/folder"
save_folder_name = "self_consistency_results"

# Self-Consistency parameters
topk = 10                      # K value for Top-K accuracy calculation
ignore_order = True            # Whether to ignore mixture order
dechirality = True             # Whether to remove stereochemistry information
force_to_topk = None           # Force number of candidates per question (None = no limit)
check_length = 5000            # Maximum SMILES length
num_process = 8                # Number of parallel processes
extract_key = "text"           # Field name to extract from JSONL
fix_escaped_backslash = True   # Whether to fix escaped backslashes
check_nmr_formula = True       # Whether to check NMR molecular formula consistency (for NMR tasks only)

# Execute Self-Consistency
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

print("Self-Consistency evaluation completed!")
print(f"Final metrics: {final_metrics}")
```

#### Run Self-Consistency
```bash
# Activate environment
conda activate clreasoner

# Enter test directory
cd test

# Run script
python chemistry_aware_self_consistency.py
```

#### Output Results
```
save_folder_path/
└── save_folder_name/
    ├── sc_results.json                    # Detailed voting results
    ├── summary_metrics.json               # Summary evaluation metrics
    ├── sc_results_formatted.csv           # CSV format results
    └── voting_statistics.json             # Voting statistics
```

#### Example Run

A complete example is provided in the project:
```bash
cd test

# View sample data
ls ../sample/chemistry_aware_self_consistency_sample_ground_truth/
ls ../sample/chemistry_aware_self_consistency_sample_infer_result_for_sc/

# Run example (the __main__ section in the code)
python chemistry_aware_self_consistency.py
```

---

## 📁 Project Structure
```
CLReasoner/
├── train/                                # Training module
│   └── train_single_gpu.py              # Single GPU/Multi-GPU training script
│
├── test/                                 # Testing and evaluation module
│   ├── test.py                          # Main inference script
│   ├── chemistry_aware_self_consistency.py  # Self-Consistency script
│   └── stat_metrics_scription.py        # Evaluation metrics calculation script
│
├── sample/                              # Sample data
│   ├── chemistry_aware_self_consistency_sample_ground_truth/
│   ├── chemistry_aware_self_consistency_sample_infer_result_for_sc/
│   └── chemistry_aware_self_consistency_sample_output/
│
├── environment.yml                      # Conda environment configuration file
└── README_CN.md                         # Chinese documentation
```

---

## ⭐ Star History

If this project helps you, please give us a Star⭐!