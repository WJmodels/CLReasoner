import glob, os
import json
from datasets import load_dataset, concatenate_datasets
import re
import time
from pathlib import Path
import inspect, importlib, sys
from typing import Union, List
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # verification check

############################ Helper Functions ############################
def get_immediate_subfolders(target_folder_path):
    """
    Get all immediate subfolder paths under the specified folder.
    
    Args:
        target_folder_path (str): The target parent folder path.
        
    Returns:
        list: A list containing full paths of all immediate subfolders. 
              Returns an empty list if the target path does not exist.
    """
    # Check if path exists
    if not os.path.exists(target_folder_path):
        print(f"Error: Folder '{target_folder_path}' does not exist.")
        return []
    
    subfolders = []
    
    # Use os.scandir for traversal (Recommended for Python 3.5+, high efficiency)
    # It returns DirEntry objects containing path attribute and is_dir() method
    try:
        with os.scandir(target_folder_path) as entries:
            for entry in entries:
                if entry.is_dir():
                    subfolders.append(entry.path)
    except Exception as e:
        print(f"Error reading folder: {e}")
        return []

    # Sort by name to ensure fixed return order (optional)
    subfolders.sort()
    
    return subfolders

# Get subfolders starting with a specific character/string
def get_subdirectories_with_certain_prefix(path, str_prefix):
    """
    Get a list of immediate subfolder paths under the specified path that start with a specific prefix.
    
    Args:
        path (str): The path to search for subfolders.
        str_prefix (str): The prefix of the subfolder names.
    
    Returns:
        list: A list of immediate subfolder paths matching the prefix condition.
    
    Usage:
        subdirs = get_subdirectories_with_certain_prefix('your_path_here', 'data_')
        print(subdirs)
    """
    # Check if path exists
    if not os.path.exists(path):
        raise ValueError(f"Path '{path}' does not exist")
    
    # Get all subfolders under the path matching the prefix condition
    subdirectories = [os.path.join(path, name) for name in os.listdir(path)
                      if os.path.isdir(os.path.join(path, name)) and name.startswith(str_prefix)]
    
    return subdirectories

def extract_checkpoint_iters(list_model_checkpoint_folder_path):
    """
    Extract the numeric part from checkpoint folder paths and convert them to a list of strings.

    Args:
        list_model_checkpoint_folder_path: List of checkpoint paths, where the last part should look like 'checkpoint-1000'.

    Returns:
        list_checkpoint_iter: A list of extracted iteration numbers (as strings).
    """
    list_checkpoint_iter = []
    for folder_path in list_model_checkpoint_folder_path:
        folder_name = os.path.basename(os.path.normpath(folder_path))  # Get folder name
        match = re.match(r"checkpoint-(\d+)", folder_name)  # Extract number using regex
        if match:
            list_checkpoint_iter.append(match.group(1))  # Append the number part as string
    return list_checkpoint_iter

def get_architecture_from_config(model_checkpoint_folder_path: str) -> str:
    # Construct the full path for config.json
    config_path = os.path.join(model_checkpoint_folder_path, "config.json")
    
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json does not exist at path: {config_path}")
    
    # Read JSON config file
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    
    # Check if "architectures" field exists
    if "architectures" not in config_data:
        raise KeyError('"architectures" field not found in config.json')
    
    # Retrieve the field and check if it is a non-empty list
    architectures = config_data["architectures"]
    if not isinstance(architectures, list) or len(architectures) == 0:
        raise ValueError('"architectures" field should be a non-empty list')
    
    # Return the first element
    return architectures[0]

# Determine model type
def detect_qwen_model_type(model_checkpoint_folder_path):
    """
    Determine the model type based on the value of the "architectures" key in the config.json file
    under model_checkpoint_folder_path, to facilitate subsequent question mapping.
    
    Args:
    model_checkpoint_folder_path: The folder path of the model files. Note: this should be the full model path, not the LoRA folder path.

    Returns:
    bool_qwen3_model: True indicates a Qwen3 model, False indicates a Qwen2 model (Deepseek R1 distilled small model). Raises an error if neither.
    """
    str_version = get_architecture_from_config(model_checkpoint_folder_path)
    if "Qwen3" in str_version:
        bool_qwen3_model = True
    elif "Qwen2" in str_version:
        bool_qwen3_model = False
    else:
        raise ValueError(f'The "architectures" field in config.json at {model_checkpoint_folder_path} indicates this model is not a supported Qwen2 or Qwen3 model.')
    return bool_qwen3_model

# Form save file paths
def build_save_folder_paths(parent_parent_save_folder_path, model_train_info, list_checkpoint_iter):
    """
    Construct a list of save path strings with the structure:
        parent_parent_save_folder_path / model_train_info / checkpoint_iter

    Args:
        parent_parent_save_folder_path: Top-level save directory (string or Path).
        model_train_info: Name of the intermediate folder (string).
        list_checkpoint_iter: List of checkpoint numbers (list of strings).

    Returns:
        list_save_folder_path: A list where each path is a string.
    """
    parent_parent_save_folder_path = Path(parent_parent_save_folder_path)

    list_save_folder_path = [
        str(parent_parent_save_folder_path / model_train_info / checkpoint_iter)
        for checkpoint_iter in list_checkpoint_iter
    ]
    return list_save_folder_path

# Unused
def build_save_file_names(model_train_info, list_checkpoint_iter, dict_arrow_folder_path_save_file_name):
    """
    Construct a list of save file names with the format:
        {model_train_info}_{checkpoint_iter}_step_{save_file_name}

    Args:
        model_train_info: Model test info string.
        list_checkpoint_iter: List of checkpoint numbers (list of strings).
        dict_arrow_folder_path_save_file_name: Dictionary mapping arrow paths to save file names.

    Returns:
        list_save_file_name: A list of constructed save file name strings.
    """
    list_save_file_name_raw = list(dict_arrow_folder_path_save_file_name.values())

    if len(list_checkpoint_iter) != len(list_save_file_name_raw):
        raise ValueError("Number of checkpoints does not match number of save file names")

    list_save_file_name = [
        f"{model_train_info}_{list_checkpoint_iter[i]}_step_{list_save_file_name_raw[i]}"
        for i in range(len(list_checkpoint_iter))
    ]
    return list_save_file_name

def import_functions(list_tuple_py_file_abs_path_function_name):
    """
    Dynamically import specific functions from each Python file in the provided list,
    and add them to the namespace of both the calling module and the __main__ module.
    
    Args:
    list_tuple_py_file_abs_path_function_name: A list where each element is a tuple containing:
        - py_file_abs_path: Absolute path to the Python file.
        - function_name: Name of the function to import.
    """
    # Get caller's frame and module
    caller_frame = inspect.currentframe().f_back
    caller_globals = caller_frame.f_globals
    caller_module_name = caller_globals['__name__']
    
    # Get caller's module object
    if caller_module_name == '__main__':
        import __main__
        caller_module = __main__
    else:
        caller_module = sys.modules[caller_module_name]
    
    # Also get the __main__ module (regardless of whether the caller is __main__)
    import __main__
    main_globals = __main__.__dict__
    
    # Dictionary of imported functions (return value)
    imported_functions = {}
    
    for py_file_abs_path, function_name in list_tuple_py_file_abs_path_function_name:
        try:
            # Load module via absolute path
            module_name = f"dynamic_module_{function_name}_{hash(py_file_abs_path)}"
            spec = importlib.util.spec_from_file_location(module_name, py_file_abs_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Ensure the function exists
            if hasattr(module, function_name):
                func = getattr(module, function_name)
                
                # Set to multiple locations to ensure availability
                
                # 1. Set to caller module
                setattr(caller_module, function_name, func)
                
                # 2. Set to caller's global namespace
                caller_globals[function_name] = func
                
                # 3. Set to __main__ module (if caller is not __main__)
                if caller_module_name != '__main__':
                    main_globals[function_name] = func
                
                # 4. Add to return dictionary
                imported_functions[function_name] = func
                
                print(f"Successfully imported function '{function_name}' from file: {py_file_abs_path}")
            else:
                print(f"Warning: Function '{function_name}' not found in {py_file_abs_path}")
        except Exception as e:
            print(f"Error: Unable to load function '{function_name}' from file: {py_file_abs_path}. Error info: {e}")
    
    return imported_functions

def get_folder_name_by_level(folder_path: str, level: int) -> str:
    """
    Get the folder name based on the specified level.
    
    Args:
        folder_path (str): Full folder path.
        level (int): Level to go up. 0 means current folder name, 1 means parent folder name, and so on.
    
    Returns:
        str: Folder name at the specified level.
    """
    path_obj = Path(folder_path)
    if level < 0:
        raise ValueError(f"Level must be a non-negative integer, current value: {level}")
    parts = path_obj.parts
    if level >= len(parts):
        raise ValueError(f"level={level} exceeds path depth, path '{folder_path}' only has {len(parts)} levels")
    return parts[-(level + 1)]

# # model_checkpoint_folder_path = r"/media/capios/20TB/lln/project/deepseek/test/..."
# # bool_qwen3_model = detect_qwen_model_type(model_checkpoint_folder_path)
# # print(f"bool_qwen3_model:{bool_qwen3_model}")


############################ Data Loading Functions ############################
def transform_chem_question_to_nat_question(question, language="en"):
    """
    Transform a chemical question into natural language format.
    
    Args:
        question: String containing the chemical question.
        language: Output language, "zh" for Chinese, "en" for English. Defaults to "zh" (logic maintained, though typically called with "en").
        
    Returns:
        nat_question: Transformed natural language question string.
    """
    # Extract Formula
    formula_match = re.search(r'Formula:(.*?);', question)
    formula = formula_match.group(1) if formula_match else None
    
    # Extract FW (Molecular Weight)
    fw_match = re.search(r'FW:(.*?);', question)
    fw = fw_match.group(1) if fw_match else None
    
    # Extract HSQC info
    hsqc_match = re.search(r'HSQC:(.*?);', question)
    hsqc = hsqc_match.group(1) if hsqc_match else None
    
    # Extract HMBC info
    hmbc_match = re.search(r'HMBC:(.*?);', question)
    hmbc = hmbc_match.group(1) if hmbc_match else None
    
    # Extract COSY info
    cosy_match = re.search(r'COSY:(.*?);', question)
    cosy = cosy_match.group(1) if cosy_match else None
    
    # Extract 13C NMR info
    cnmr_match = re.search(r'13C NMR:\[NCP:(\d+)\](.*?);', question)
    c_nmr_count = cnmr_match.group(1) if cnmr_match else None
    c_nmr_shifts = cnmr_match.group(2).split(',') if cnmr_match and cnmr_match.group(2) else []
    
    # Extract 1H NMR info
    hnmr_match = re.search(r'1H NMR:\[NHP:(\d+)\](.*?)($|;)', question)
    h_nmr_count = hnmr_match.group(1) if hnmr_match else None
    h_nmr_signals_str = hnmr_match.group(2) if hnmr_match else ""
    
    # Check if [MASK] exists in 1H NMR signals
    has_mask_in_hnmr = "[MASK]" in h_nmr_signals_str
    
    # Build natural language output
    result_parts = []
    
    # Select text template based on language
    if language == "zh":
        # Chinese version
        # Merge Formula and FW info
        if formula and fw and formula != '[MASK]' and fw != '[MASK]':
            result_parts.append(f"该分子的分子式为{formula}，分子量为{fw}")
        else:
            if formula == '[MASK]' or formula is None:
                result_parts.append("分子式信息未给出")
            elif formula:
                result_parts.append(f"该分子的分子式为{formula}")
                
            if fw == '[MASK]' or fw is None:
                result_parts.append("分子量信息未给出")
            elif fw:
                result_parts.append(f"分子量为{fw}")
        
        # Add HSQC, HMBC, COSY info
        if hsqc == '[MASK]' and hmbc == '[MASK]' and cosy == '[MASK]':
            result_parts.append("HSQC、HMBC、COSY信息未给出")
        else:
            if hsqc == '[MASK]':
                result_parts.append("HSQC信息未给出")
            elif hsqc:
                result_parts.append(f"HSQC信息为{hsqc}")
                
            if hmbc == '[MASK]':
                result_parts.append("HMBC信息未给出")
            elif hmbc:
                result_parts.append(f"HMBC信息为{hmbc}")
                
            if cosy == '[MASK]':
                result_parts.append("COSY信息未给出")
            elif cosy:
                result_parts.append(f"COSY信息为{cosy}")
        
        # Add 13C NMR info
        if c_nmr_count == '[MASK]' or (c_nmr_count and len(c_nmr_shifts) == 1 and c_nmr_shifts[0] == '[MASK]'):
            result_parts.append("13C NMR信息未给出")
        elif c_nmr_count and c_nmr_shifts:
            result_parts.append(f"13C NMR一共有{c_nmr_count}个化学位移信号，分别为：{','.join(c_nmr_shifts)}")
        
        # Add 1H NMR info
        if h_nmr_count == '[MASK]' or (h_nmr_count and h_nmr_signals_str == '[MASK]'):
            result_parts.append("1H NMR信息未给出")
        elif h_nmr_count:
            if h_nmr_signals_str:
                result_parts.append(f"1H NMR一共有{h_nmr_count}个信号，以(chemical shift,integration,(multiplicity,coupling patterns))的形式分别表示为{h_nmr_signals_str}")
            else:
                result_parts.append(f"1H NMR一共有{h_nmr_count}个信号")
            
            if has_mask_in_hnmr:
                result_parts.append("其中[MASK]表示该信息未给出")
        
        # Add closing statement
        result_parts.append("根据以上信息推理分子的SMILES:")
    
    else:
        # English version - More formal expression
        # Merge Formula and FW info
        if formula and fw and formula != '[MASK]' and fw != '[MASK]':
            result_parts.append(f"The molecular formula is {formula} with molecular weight of {fw}")
        else:
            if formula == '[MASK]' or formula is None:
                result_parts.append("The molecular formula is not available")
            elif formula:
                result_parts.append(f"The molecular formula is {formula}")
                
            if fw == '[MASK]' or fw is None:
                result_parts.append("The molecular weight is not available")
            elif fw:
                result_parts.append(f"The molecular weight is {fw}")
        
        # Add HSQC, HMBC, COSY info
        if hsqc == '[MASK]' and hmbc == '[MASK]' and cosy == '[MASK]':
            result_parts.append("HSQC, HMBC, and COSY data are not available")
        else:
            if hsqc == '[MASK]':
                result_parts.append("HSQC data is not available")
            elif hsqc:
                result_parts.append(f"HSQC data is {hsqc}")
                
            if hmbc == '[MASK]':
                result_parts.append("HMBC data is not available")
            elif hmbc:
                result_parts.append(f"HMBC data is {hmbc}")
                
            if cosy == '[MASK]':
                result_parts.append("COSY data is not available")
            elif cosy:
                result_parts.append(f"COSY data is {cosy}")
        
        # Add 13C NMR info
        if c_nmr_count == '[MASK]' or (c_nmr_count and len(c_nmr_shifts) == 1 and c_nmr_shifts[0] == '[MASK]'):
            result_parts.append("13C NMR data is not available")
        elif c_nmr_count and c_nmr_shifts:
            result_parts.append(f"The 13C NMR spectrum shows {c_nmr_count} chemical shift signals: {','.join(c_nmr_shifts)}")
        
        # Add 1H NMR info
        if h_nmr_count == '[MASK]' or (h_nmr_count and h_nmr_signals_str == '[MASK]'):
            result_parts.append("1H NMR data is not available")
        elif h_nmr_count:
            if h_nmr_signals_str:
                result_parts.append(f"The 1H NMR spectrum exhibits {h_nmr_count} signals in the format (chemical shift, integration, (multiplicity, coupling patterns)): {h_nmr_signals_str}")
            else:
                result_parts.append(f"The 1H NMR spectrum exhibits {h_nmr_count} signals")
            
            if has_mask_in_hnmr:
                result_parts.append("Where [MASK] indicates that the information is not provided")
        
        # Add closing statement
        result_parts.append("Based on the above information, infer the molecular SMILES:")
    
    # Select connector based on language, use period to join all parts
    nat_question = "。".join(result_parts) if language == "zh" else ". ".join(result_parts)
    
    return nat_question

# Load data from arrow data folder
def load_multiple_arrow_files(directory_path, split="train"):
    """
    Load multiple .arrow files from a directory and concatenate them into a single dataset.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory {directory_path} does not exist")
    
    # Get all .arrow files in the directory
    arrow_files = glob.glob(os.path.join(directory_path, "*.arrow"))
    
    if not arrow_files:
        raise FileNotFoundError(f"No .arrow files found in directory {directory_path}")
    
    print(f"Found {len(arrow_files)} .arrow files in {directory_path}")
    
    # Load each file and create a list of datasets
    datasets_list = []
    for file_path in arrow_files:
        print(f"Loading file: {os.path.basename(file_path)}")
        try:
            dataset = load_dataset("arrow", data_files=file_path, split=split)
            datasets_list.append(dataset)
            print(f"Successfully loaded, this file contains {len(dataset)} samples")
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
    
    if not datasets_list:
        raise ValueError("No datasets were successfully loaded")
    
    # Concatenate all datasets
    if len(datasets_list) == 1:
        return datasets_list[0]
    else:
        return concatenate_datasets(datasets_list)

def standardize_shift_decimal_place(question, adjust_decimal_places=False, c_shift_decimal_place=1, h_shift_decimal_place=2):
    """
    Standardize the decimal places of NMR chemical shift data.
    Args:
        question: String containing NMR data.
        c_shift_decimal_place: Decimal places for 13C NMR chemical shift, default 1.
        h_shift_decimal_place: Decimal places for 1H NMR chemical shift, default 2.
    Returns:
        processed_question: The question string after decimal place standardization.
    """
    if not adjust_decimal_places:
        return question
    # print("adjust_decimal_places")
    processed_question = question
    
    # Process 13C NMR data
    # Match content inside brackets and value list after 13C NMR
    c_nmr_pattern = r'(13C NMR:\[NCP:\d+\])([\d.,]+)(;)'
    c_nmr_match = re.search(c_nmr_pattern, processed_question)
    
    if c_nmr_match:
        prefix = c_nmr_match.group(1)  # "13C NMR:[NCP:10]"
        shift_values = c_nmr_match.group(2)  # "20.9,62.6,66.2,74.8,131.5,143.1,149.57,151.3,151.8,170.815"
        suffix = c_nmr_match.group(3)  # ";"
        
        # Split values and format decimal places
        values = shift_values.split(',')
        formatted_values = []
        for value in values:
            try:
                # Convert string to float, then format decimal places
                float_value = float(value.strip())
                formatted_value = f"{float_value:.{c_shift_decimal_place}f}"
                formatted_values.append(formatted_value)
            except ValueError:
                # Keep original value if conversion fails
                formatted_values.append(value.strip())
        
        # Reassemble 13C NMR data
        new_c_nmr = prefix + ','.join(formatted_values) + suffix
        processed_question = processed_question.replace(c_nmr_match.group(0), new_c_nmr)
    
    # Process 1H NMR data
    # Match all tuple data after 1H NMR
    h_nmr_pattern = r'(1H NMR:\[NHP:\d+\])(.*?)(?=;|$)'
    h_nmr_match = re.search(h_nmr_pattern, processed_question)
    
    if h_nmr_match:
        prefix = h_nmr_match.group(1)  # "1H NMR:[NHP:6]"
        tuples_content = h_nmr_match.group(2)  # All tuple content
        
        # Match the first numerical value in each tuple (value, number, [MASK])
        tuple_pattern = r'\((\d+\.?\d*),(\d+),(\[MASK\])\)'
        
        def replace_tuple(match):
            # Get first value and format it
            first_value = float(match.group(1))
            formatted_first = f"{first_value:.{h_shift_decimal_place}f}"
            
            # Reassemble tuple
            return f"({formatted_first},{match.group(2)},{match.group(3)})"
        
        # Replace first numerical value in all tuples
        new_tuples_content = re.sub(tuple_pattern, replace_tuple, tuples_content)
        
        # Reassemble 1H NMR data
        new_h_nmr = prefix + new_tuples_content
        processed_question = processed_question.replace(h_nmr_match.group(0), new_h_nmr)
    
    return processed_question

# Load questions from arrow data folder
def get_list_question_from_arrow_folder_path(arrow_folder_path, bool_nat_question, adjust_decimal_places=False, c_shift_decimal_place=1, h_shift_decimal_place=2):
    arrow_folder_datasets = load_multiple_arrow_files(arrow_folder_path)
    list_question = arrow_folder_datasets['question']
    # Standardize decimal places
    list_question = [standardize_shift_decimal_place(q, adjust_decimal_places=adjust_decimal_places, c_shift_decimal_place=c_shift_decimal_place, h_shift_decimal_place=h_shift_decimal_place) for q in list_question]
    if bool_nat_question:
        list_nat_question = []
        for chem_question in list_question:
            nat_question = transform_chem_question_to_nat_question(chem_question, language="en")
            list_nat_question.append(nat_question)
        list_question = list_nat_question
    return list_question


############################ Question Mapping Section ############################
def mapping_list_qusetion_to_certain_form(list_question, tokenizer, bool_qwen3_model, bool_beam_search):
    """
    Prepare questions for inference (apply templates)
    1. Args:
    (1) list_question: List of questions, where each element is a question (string).
    (2) tokenizer: The tokenizer corresponding to the model.
    (3) bool_qwen3_model: Whether the model used for inference is a Qwen3 model. True for Qwen3, False for Qwen2 (Deepseek R1 distilled small model).
    (4) bool_beam_search: Whether to use beam search.

    2. Returns:
    prompt_texts: Mapped list_question.
    """
    prompt_texts = []
    for question in list_question:
        # Construct prompt containing <think> tag to let the model continue the thinking process
        # Format: <Task>[Synthesis]</Task>[Reactant1]...<Think>
        # This way the model will start generating thinking content from <Think>, and then automatically generate </Think><answer>...</answer>
        
        if bool_qwen3_model:
            # For Qwen3 models, we need to use chat template
            # We need to construct a message containing <think>
            full_prompt = f"{question}<think>"
            messages = [
                {"role": "user", "content": full_prompt}
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_texts.append(text)
        else:
            # For Qwen2/DeepSeek models, directly add <Think> tag
            # The format should be similar to the training format: Question + "<Think>"
            prompt_texts.append(f"{question}<think>")
    
    if bool_beam_search:
        prompt_texts = [{"prompt": p} for p in prompt_texts]
    return prompt_texts


############################ Model Loading Section ############################
# Load model
def load_model_from_model_checkpoint_folder_path(model_checkpoint_folder_path):
    llm = LLM(model=model_checkpoint_folder_path,tensor_parallel_size=tensor_parallel_size)
    return llm

# visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
# tensor_parallel_size = len(visible_gpus)
# llm = LLM(model=model_checkpoint_folder_path,tensor_parallel_size=tensor_parallel_size)
# print(f"llm:{llm},tensor_parallel_size:{tensor_parallel_size}")
#####################################################################




############################ Data Saving Section ############################
# Save vLLM inference results to a JSONL file
def save_vllm_infer_result_to_jsonl_file(vllm_infer_result, save_folder_path, save_file_name):
    """
    Save vLLM inference results to a JSONL file.
    This function automatically detects whether the results were generated by the generate method or beam_search.

    Args:
        vllm_infer_result: List of vLLM inference results (generated by generate or beam_search method).
        save_folder_path: Path of the folder to save to.
        save_file_name: Name of the file to save (without extension).
    
    Returns:
        None: Function returns nothing, results are saved directly to file.
    """
    # Ensure save folder path exists
    os.makedirs(save_folder_path, exist_ok=True)
    
    # Construct full file path
    file_path = os.path.join(save_folder_path, f"{save_file_name}.jsonl")
    
    # Prepare data to write
    jsonl_data = []
    
    # Check if list is empty
    if not vllm_infer_result:
        return
    
    # Get the first element in the list to determine generation method
    first_item = vllm_infer_result[0]
    
    # Determine inference method type based on properties of the first element (only need to check once)
    if hasattr(first_item, 'sequences'):
        # Result generated by beam_search method
        for item in vllm_infer_result:
            # Collect all beam results
            beam_results = []
            for sequence in item.sequences:
                beam_results.append(sequence.text)
            # Save all beam results as one JSON object
            jsonl_data.append({"text": beam_results})
    else:
        # Result generated by generate method
        for item in vllm_infer_result:
            generat_result = []
            for generate_output in item.outputs:
                text = item.prompt + item.outputs[0].text
                generat_result.append(text)
            jsonl_data.append({"text": generat_result})
    
    # Write data to JSONL file
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in jsonl_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

############################ Result Evaluation Section ############################
# def convert_jsonl_to_txt_file(model_infer_result_folder_path):
#     ... (Original code commented out) ...

def convert_jsonl_to_txt_file(model_infer_result_folder_path):
    """
    Convert all .jsonl files in the specified path to .txt files.
    
    Args:
        model_infer_result_folder_path: Folder path containing model training info.
    
    Returns:
        None: Function returns nothing, but creates .txt files in the file system.
    """
    # Get main folder name (i.e., model_train_info)
    print(f"ctime: {time.ctime()}")
    model_train_info = os.path.basename(model_infer_result_folder_path)
    
    # Traverse folder structure
    for root, dirs, files in os.walk(model_infer_result_folder_path):
        for file in files:
            if file.endswith('.jsonl'):
                # Get full path of the file
                jsonl_file_path = os.path.join(root, file)
                
                # Get relative path part to extract {training iteration}
                rel_path = os.path.relpath(root, model_infer_result_folder_path)
                
                # Check if rel_path is valid (not empty string or ".")
                if rel_path == '.' or not rel_path:
                    # Skip if jsonl file is directly in model_infer_result_folder_path
                    continue
                
                # Get the first level subfolder name, i.e., {training iteration}
                train_iter = rel_path.split(os.sep)[0]
                
                # Extract dataset name (remove .jsonl extension)
                dataset_name = os.path.splitext(file)[0]
                
                # Construct output .txt file name
                txt_file_name = f"{model_train_info}_{train_iter}_step_{dataset_name}.txt"
                
                # Construct output .txt file path (in the same directory as .jsonl file)
                txt_file_path = os.path.join(root, txt_file_name)
                
                print(f"Processing {jsonl_file_path} -> {txt_file_path}")
                
                # Process .jsonl file and create .txt file
                with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_in, \
                     open(txt_file_path, 'w', encoding='utf-8') as txt_out:
                    for line_num, line in enumerate(jsonl_in, 1):
                        try:
                            # Parse JSON line
                            data = json.loads(line.strip())
                            
                            # Get value of 'text' field (should be a list of strings)
                            test_field = data.get('text', [])
                            
                            # Check if test_field is a list
                            if not isinstance(test_field, list):
                                print(f"Warning: 'text' field in line {line_num} of file {jsonl_file_path} is not a list")
                                # Write non-list test field to txt file, replacing newline with escape sequence
                                txt_out.write(str(test_field).replace('\n', '\\n') + '\n')
                                continue
                            
                            # If test list is empty
                            if not test_field:
                                print(f"Warning: 'text' list in line {line_num} of file {jsonl_file_path} is empty")
                                # Write empty line
                                txt_out.write('\n')
                                continue
                            
                            # Get the first string in the test list
                            first_string = test_field[0]
                            
                            # Replace newlines in string to ensure one line of JSON corresponds to one line in output file
                            first_string_normalized = first_string.replace('\n', '\\n').replace('\r', '\\r')
                            
                            # Write string to txt file, followed by a newline
                            txt_out.write(first_string_normalized + '\n')
                            
                        except json.JSONDecodeError:
                            # Handle JSON parse error
                            print(f"Error: JSON parse error occurred at line {line_num} of file {jsonl_file_path}: {line.strip()}")
                            # Write empty line to maintain line count
                            txt_out.write('\n')
                
                print(f"Converted {jsonl_file_path} to {txt_file_path}")

def evaluate_infer_result_by_convert_to_txt(parent_checkpoint_folder_path, parent_parent_save_folder_path, list_tuple_py_file_abs_path_function_name, dict_key_test_arrow_folder_path_value_dataset_name, level=0):
    print(f"time.ctime{time.ctime}")
    import_functions(list_tuple_py_file_abs_path_function_name)
    # list_test_arrow_file_path = list(dict_key_test_arrow_file_path_value_dataset_name.keys())

    checkpoint_folder_name = get_folder_name_by_level(parent_checkpoint_folder_path, level)
    save_folder_path = str(Path(parent_parent_save_folder_path) / checkpoint_folder_name)

    convert_jsonl_to_txt_file(save_folder_path)
    # batch_evaluate_multiple_test_files(list_test_arrow_file_path, save_folder_path, dict_key_test_arrow_file_path_value_dataset_name, save_result_as_json_and_csv=True)
    batch_evaluate_multiple_test_files(save_folder_path, dict_key_test_arrow_folder_path_value_dataset_name, save_result_as_json_and_csv=True)

############################ Inference Section ############################
def one_model_vllm_infer_one_dataset(
        model_checkpoint_folder_path, 
        arrow_folder_path, 
        bool_nat_question,
        save_folder_path=".", save_file_name="vllm_infer_result",
        beam_width = 1,
        max_tokens = 5000,
        temperature = 0, top_p = 0.95, top_k = 20,
        c_shift_decimal_place=1, h_shift_decimal_place=2
):
    # 1. Model Loading
    visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    tensor_parallel_size = len(visible_gpus)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_folder_path)
    llm = LLM(model=model_checkpoint_folder_path,tensor_parallel_size=tensor_parallel_size)

    # 2. Data Loading
    # 2.1 Load questions from arrow data folder
    list_question = get_list_question_from_arrow_folder_path(arrow_folder_path, bool_nat_question=bool_nat_question, adjust_decimal_places=adjust_decimal_places, c_shift_decimal_place=c_shift_decimal_place, h_shift_decimal_place=h_shift_decimal_place)
    # 2.2 Detect if it is a qwen3 model
    bool_qwen3_model = detect_qwen_model_type(model_checkpoint_folder_path)
    # 2.3 Check if beam search is needed
    bool_beam_search = True if (beam_width>1) else False
    # 2.4 Map questions
    list_question = mapping_list_qusetion_to_certain_form(list_question, tokenizer, bool_qwen3_model, bool_beam_search)

    # 3. Model Inference
    # 3.1. Prepare parameters
    sampling_params = SamplingParams(
        temperature=temperature, 
        max_tokens=max_tokens,
        top_p=top_p, 
        top_k=top_k,
        )
    
    beam_params = BeamSearchParams(
        beam_width=beam_width,       # Set beam width
        temperature=temperature,      # temperature=0 means greedy generation
        max_tokens=max_tokens,    # Set max generated tokens
        ignore_eos=False,   # Default False, stops on EOS
        length_penalty=1.0,  # Control length preference, 1.0 means neutral
        stop_token_ids=[tokenizer.eos_token_id],   # Can accelerate stopping
    )

    # 3.2 Inference results
    if bool_beam_search:
        outputs = llm.beam_search(list_question, beam_params)
    else:
        outputs = llm.generate(list_question, sampling_params)
    
    # 4. Save Results
    save_vllm_infer_result_to_jsonl_file(outputs, save_folder_path, save_file_name)

    return outputs


def one_model_vllm_infer_multi_dataset(
    model_checkpoint_folder_path,
    list_arrow_folder_path,
    list_save_file_name,
    bool_nat_question,
    save_folder_path=".",
    beam_width = 1,
    max_tokens = 5000,
    temperature = 0, top_p = 0.95, top_k = 20,
    adjust_decimal_places=False, c_shift_decimal_place=1, h_shift_decimal_place=2
):
    """
    Use the same model to infer on multiple datasets to avoid reloading the model.
    
    Args:
        model_checkpoint_folder_path: Path to model checkpoint folder
        list_arrow_folder_path: List of arrow dataset folder paths
        list_save_file_name: List of save file names corresponding to each dataset
        bool_nat_question: Whether it is a natural language question
        save_folder_path: Folder path to save results, defaults to current directory
        beam_width: Width for beam search, defaults to 1
        max_tokens: Maximum generated tokens, defaults to 5000
        temperature: Sampling temperature, defaults to 0
        top_p: top-p sampling parameter, defaults to 0.95
        top_k: top-k sampling parameter, defaults to 20
    
    Returns:
        list_infer_results: List containing inference results for multiple datasets
    """
    # Check input parameters
    if len(list_arrow_folder_path) != len(list_save_file_name):
        raise ValueError(f"Number of test files ({len(list_arrow_folder_path)}) does not match number of save file names ({len(list_save_file_name)})")
    
    print(f"\n{'='*80}")
    print(f"Starting batch inference task")
    print(f"Model path: {model_checkpoint_folder_path}")
    print(f"Number of datasets: {len(list_arrow_folder_path)}")
    print(f"Beam width: {beam_width}")
    print(f"{'='*80}\n")
    
    # 1. Model Loading (Load only once)
    print(f"[Init] Loading model and tokenizer...")
    visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    tensor_parallel_size = len(visible_gpus)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_folder_path)
    llm = LLM(model=model_checkpoint_folder_path, tensor_parallel_size=tensor_parallel_size)
    print(f"[Init] Model loading complete (Using {tensor_parallel_size} GPUs)\n")
    
    # 2. Detect if it is a qwen3 model
    bool_qwen3_model = detect_qwen_model_type(model_checkpoint_folder_path)
    print(f"[Init] Model type: {'Qwen3' if bool_qwen3_model else 'Qwen2'}")
    
    # 3. Check if beam search is needed
    bool_beam_search = True if (beam_width > 1) else False
    print(f"[Init] Inference mode: {'Beam Search' if bool_beam_search else 'Greedy Decoding'}\n")
    
    # 4. Prepare inference parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
    )
    
    beam_params = BeamSearchParams(
        beam_width=beam_width,      # Set beam width
        temperature=temperature,    # temperature=0 means greedy generation
        max_tokens=max_tokens,      # Set max generated tokens
        ignore_eos=False,           # Default False, stops on EOS
        length_penalty=1.0          # Control length preference, 1.0 means neutral
    )
    
    # 5. Loop through each dataset
    list_infer_results = []
    for i in range(len(list_arrow_folder_path)):
        arrow_folder_path = list_arrow_folder_path[i]
        save_file_name = list_save_file_name[i]
        
        print(f"\n{'='*80}")
        print(f"Processing dataset [{i+1}/{len(list_arrow_folder_path)}]: {save_file_name}")
        print(f"Data path: {arrow_folder_path}")
        print(f"{'='*80}\n")
        
        # 5.1 Load questions from arrow data folder
        print(f"[Step 1/4] Loading data...")
        list_question = get_list_question_from_arrow_folder_path(
            arrow_folder_path, 
            bool_nat_question=bool_nat_question, 
            adjust_decimal_places=adjust_decimal_places, 
            c_shift_decimal_place=c_shift_decimal_place, 
            h_shift_decimal_place=h_shift_decimal_place
        )
        print(f"[Step 1/4] ✓ Data loading complete, total {len(list_question)} samples\n")
        
        # 5.2 Map questions
        print(f"[Step 2/4] Mapping question format...")
        list_question = mapping_list_qusetion_to_certain_form(
            list_question, 
            tokenizer, 
            bool_qwen3_model, 
            bool_beam_search
        )
        print(f"[Step 2/4] ✓ Question format mapping complete\n")
        
        # 5.3 Model inference
        print(f"[Step 3/4] Starting model inference...")
        if bool_beam_search:
            print(f"            Using Beam Search (beam_width={beam_width})")
            print(f"            ⚠️  First inference may take 1-3 minutes for compilation optimization, please wait...")
            print(f"            Progress bar will be displayed during inference\n")
            outputs = llm.beam_search(list_question, beam_params)
        else:
            print(f"            Using Greedy Decoding")
            print(f"            Progress bar will be displayed during inference\n")
            outputs = llm.generate(list_question, sampling_params)
        
        print(f"\n[Step 3/4] ✓ Inference complete\n")
        
        # 5.4 Save results
        print(f"[Step 4/4] Saving results...")
        save_vllm_infer_result_to_jsonl_file(outputs, save_folder_path, save_file_name)
        save_path = os.path.join(save_folder_path, f"{save_file_name}.jsonl")
        print(f"[Step 4/4] ✓ Results saved to: {save_path}\n")
        
        # 5.5 Add to results list
        list_infer_results.append(outputs)
        
        print(f"Dataset [{i+1}/{len(list_arrow_folder_path)}] processing complete\n")
    
    print(f"\n{'='*80}")
    print(f"All {len(list_arrow_folder_path)} datasets inference complete!")
    print(f"{'='*80}\n")
    
    return list_infer_results

def multi_model_vllm_infer_multi_dataset(
        list_model_checkpoint_folder_path,
        list_arrow_folder_path,
        list_save_file_name,
        bool_nat_question,
        list_save_folder_path,
        beam_width = 1,
        max_tokens = 5000,
        temperature = 0, top_p = 0.95, top_k = 20,
        adjust_decimal_places=False, c_shift_decimal_place=1, h_shift_decimal_place=2
):
    if len(list_model_checkpoint_folder_path) != len(list_save_folder_path):
        raise ValueError(f"Number of model weights ({len(list_model_checkpoint_folder_path)}) does not match number of save folders ({len(list_save_folder_path)})")

    list_list_one_model_vllm_infer_multi_dataset_result = []
    for i in range(len(list_model_checkpoint_folder_path)):
        list_one_model_vllm_infer_multi_dataset_result = one_model_vllm_infer_multi_dataset(
            list_model_checkpoint_folder_path[i],
            list_arrow_folder_path,
            list_save_file_name,
            bool_nat_question,
            save_folder_path = list_save_folder_path[i],
            beam_width = beam_width,
            max_tokens = max_tokens,
            temperature = temperature, top_p = top_p, top_k = top_k,
            adjust_decimal_places=adjust_decimal_places, c_shift_decimal_place=c_shift_decimal_place, h_shift_decimal_place=h_shift_decimal_place
        )
        list_list_one_model_vllm_infer_multi_dataset_result.append(list_one_model_vllm_infer_multi_dataset_result)
    return list_list_one_model_vllm_infer_multi_dataset_result

def quick_multi_model_vllm_infer_multi_dataset(
        parent_checkpoint_folder_path,
        dict_arrow_folder_path_save_file_name,
        parent_parent_save_folder_path,
        bool_nat_question,
        max_tokens=5000,
        beam_width=1,
        temperature = 0, top_p = 0.95, top_k = 20,
        adjust_decimal_places=False, c_shift_decimal_place=1, h_shift_decimal_place=2,
        level=0
):
    """
    Further encapsulation based on multi_model_vllm_infer_multi_dataset for quick calls.

    Args:
    (1) parent_checkpoint_folder_path: The parent folder of the checkpoint folders, i.e., this folder contains multiple checkpoint-xxxx folders.
    (2) dict_arrow_folder_path_save_file_name: A dictionary.
        Key: Arrow dataset folder path.
        Value: File name to use when saving test results for this dataset.
    (3) parent_parent_save_folder_path: Test results will be saved under this path as follows:
        - parent_parent_save_folder_path
        -- Folder matching parent_checkpoint_folder_path name
        --- Folder named after checkpoint iteration
        ---- .jsonl files named after values in dict_arrow_folder_path_save_file_name
    (4) bool_nat_question: Whether to convert questions to natural language.

    Returns:
    list_list_one_model_vllm_infer_multi_dataset_result: Each result in the list is a list_one_model_vllm_infer_multi_dataset_result object, i.e., a list of results for one model inferring on multiple datasets.
    """
    list_model_checkpoint_folder_path = get_subdirectories_with_certain_prefix(parent_checkpoint_folder_path,str_prefix="checkpoint")
    list_arrow_folder_path = list(dict_arrow_folder_path_save_file_name.keys()) 

    model_train_info = get_folder_name_by_level(parent_checkpoint_folder_path, level)
    list_checkpoint_iter = extract_checkpoint_iters(list_model_checkpoint_folder_path)
    list_save_folder_path = build_save_folder_paths(parent_parent_save_folder_path, model_train_info, list_checkpoint_iter)
    # list_save_file_name = build_save_file_names(model_train_info,list_checkpoint_iter,dict_arrow_folder_path_save_file_name)
    list_save_file_name = list(dict_arrow_folder_path_save_file_name.values())

    list_list_one_model_vllm_infer_multi_dataset_result = multi_model_vllm_infer_multi_dataset(
        list_model_checkpoint_folder_path, 
        list_arrow_folder_path, 
        list_save_file_name,
        bool_nat_question,
        list_save_folder_path = list_save_folder_path, 
        beam_width = beam_width,
        max_tokens = max_tokens,
        temperature = temperature, top_p = top_p, top_k = top_k,
        adjust_decimal_places=adjust_decimal_places, c_shift_decimal_place=c_shift_decimal_place, h_shift_decimal_place=h_shift_decimal_place
    )
    return list_list_one_model_vllm_infer_multi_dataset_result

def vllm_infer_multi_folder_weight(
    list_parent_checkpoint_folder_path: Union[str, List[str]],
    dict_arrow_folder_path_save_file_name: dict,
    parent_parent_save_folder_path: str,
    list_tuple_py_file_abs_path_function_name: list,
    bool_nat_question: bool = False,
    beam_width: int = 1,
    max_tokens: int = 5000,
    temperature: float = 0,
    top_p: float = 0.95,
    top_k: int = -1,
    adjust_decimal_places: bool = True,
    c_shift_decimal_place: int = 1,
    h_shift_decimal_place: int = 2,
    level: Union[int, str] = "auto",
    list_need_level_1: List[str] = None
):
    """
    Complete workflow for inference and evaluation of single or multiple folder weights using VLLM.
    
    Args:
        list_parent_checkpoint_folder_path (Union[str, List[str]]): Single checkpoint folder path (string) or list of multiple paths.
        dict_arrow_folder_path_save_file_name (dict): Mapping dictionary of dataset paths and save file names.
        parent_parent_save_folder_path (str): Parent directory path for saving results.
        list_tuple_py_file_abs_path_function_name (list): List of tuples containing file paths and function names for validation functions.
        bool_nat_question (bool): Whether it is a natural language question, default False.
        beam_width (int): Beam search width, default 1.
        max_tokens (int): Maximum generated tokens, default 5000.
        temperature (float): Sampling temperature, default 0 (greedy search).
        top_p (float): nucleus sampling parameter, default 0.95.
        top_k (int): top-k sampling parameter, default -1 (disabled).
        adjust_decimal_places (bool): Whether to adjust decimal places, default True.
        c_shift_decimal_place (int): Decimal places for Carbon chemical shift, default 1.
        h_shift_decimal_place (int): Decimal places for Hydrogen chemical shift, default 2.
        level (Union[int, str]): Hierarchy level for result folder names. Integer indicates levels up (0 for current, 1 for parent), 
                                 "auto" means automatic determination based on list_need_level_1, default "auto".
        list_need_level_1 (List[str]): List of folder names that trigger level=1 when level="auto". 
                                       Otherwise level=0 is used. Default is ["outputs", "output", "ouput"].
        
    Returns:
        None: Function performs complete inference and evaluation workflow for each model folder, saving results to specified directory.
    """
    
    # Set default value for list_need_level_1
    if list_need_level_1 is None:
        list_need_level_1 = ["outputs", "output", "ouput"]
    
    # Process input parameters: if single string, convert to list
    if isinstance(list_parent_checkpoint_folder_path, str):
        checkpoint_folder_paths = [list_parent_checkpoint_folder_path]
        print(f"Processing single model weight folder: {list_parent_checkpoint_folder_path}")
    else:
        checkpoint_folder_paths = list_parent_checkpoint_folder_path
        print(f"Batch processing {len(checkpoint_folder_paths)} model weight folders")
    
    # Iterate through each model weight folder path
    for i, current_checkpoint_folder_path in enumerate(checkpoint_folder_paths):
        if len(checkpoint_folder_paths) > 1:
            print(f"Processing {i+1}/{len(checkpoint_folder_paths)} model weight folder: {current_checkpoint_folder_path}")
        
        # Decide which level folder name to use based on level parameter
        if level == "auto":
            folder_name_level_0 = Path(current_checkpoint_folder_path).name
            if folder_name_level_0 in list_need_level_1:
                actual_level = 1
            else:
                actual_level = 0
        else:
            actual_level = level
        
        # Execute multi-model VLLM inference
        list_list_one_model_vllm_infer_multi_dataset_result = quick_multi_model_vllm_infer_multi_dataset(
            current_checkpoint_folder_path,
            dict_arrow_folder_path_save_file_name,
            parent_parent_save_folder_path,
            bool_nat_question,
            max_tokens=max_tokens,
            beam_width=beam_width,
            temperature=temperature, 
            top_p=top_p, 
            top_k=top_k,
            adjust_decimal_places=adjust_decimal_places,
            c_shift_decimal_place=c_shift_decimal_place,
            h_shift_decimal_place=h_shift_decimal_place,
            level=actual_level
        )
        
        # Build test result folder path
        result_folder_name = get_folder_name_by_level(current_checkpoint_folder_path, actual_level)
        test_result_folder_path = Path(parent_parent_save_folder_path) / result_folder_name
        
        # Convert JSONL files to TXT files
        convert_jsonl_to_txt_file(test_result_folder_path)
        
        # Evaluate inference results
        evaluate_infer_result_by_convert_to_txt(
            current_checkpoint_folder_path, 
            parent_parent_save_folder_path, 
            list_tuple_py_file_abs_path_function_name, 
            dict_arrow_folder_path_save_file_name,
            level=actual_level
        )
        
        if len(checkpoint_folder_paths) > 1:
            print(f"Model weight folder {i+1} processing complete: {current_checkpoint_folder_path}")
    
    if len(checkpoint_folder_paths) > 1:
        print(f"All {len(checkpoint_folder_paths)} model weight folders processing complete!")
    else:
        print(f"Model weight folder processing complete: {checkpoint_folder_paths[0]}")


def single_checkpoint_infer_pipeline(
    checkpoint_folder_path: str,
    ground_truth_arrow_folder_path: str,
    dataset_name: str,
    output_folder_path: str,
    result_name: str,
    stat_metrics_scription_file_path: str = r"./stat_metrics_scription.py",
    beam_width: int = 1,
    max_tokens: int = 5000,
    temperature: float = 0,
    top_p: float = 0.95,
    top_k: int = 20,
    adjust_decimal_places: bool = False,
    c_shift_decimal_place: int = 1,
    h_shift_decimal_place: int = 2
):
    """
    Complete pipeline for performing inference on a single checkpoint and a single dataset, 
    converting the format, and executing evaluation.

    The generated files (.jsonl and .txt) will be named in the format: {result_name}_{dataset_name}.

    Args:
        checkpoint_folder_path (str): The path to the model checkpoint folder.
        ground_truth_arrow_folder_path (str): The path to the folder containing the Ground Truth dataset.
        dataset_name (str): The name of the dataset. This will be used as a suffix for the output filenames.
        output_folder_path (str): The root directory where output results will be stored.
        result_name (str): The specific result identifier. A subfolder with this name will be created inside `output_folder_path`, 
                           and it will also be used as a prefix for the output filenames.
        stat_metrics_scription_file_path (str, optional): The path to the Python script containing the evaluation function 
                                                          `batch_evaluate_multiple_test_files`. Defaults to "./stat_metrics_scription.py".
        beam_width (int, optional): The beam width for beam search decoding. Defaults to 1 (Greedy Search).
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 5000.
        temperature (float, optional): Sampling temperature. Defaults to 0.
        top_p (float, optional): Nucleus sampling probability. Defaults to 0.95.
        top_k (int, optional): Top-k sampling parameter. Defaults to 20.
        adjust_decimal_places (bool, optional): Whether to standardize decimal places in NMR data. Defaults to False.
        c_shift_decimal_place (int, optional): The number of decimal places for 13C NMR shifts. Defaults to 1.
        h_shift_decimal_place (int, optional): The number of decimal places for 1H NMR shifts. Defaults to 2.

    Returns:
        None: The function saves the inference results and evaluation metrics (CSV/JSON) to the disk.
    """
    
    # 1. Construct the save directory path
    # The results will be saved inside: output_folder_path/result_name
    save_folder_path = os.path.join(output_folder_path, result_name)
    os.makedirs(save_folder_path, exist_ok=True)
    
    # Define the final filename prefix format: {result_name}_{dataset_name}
    full_save_name = f"{result_name}_{dataset_name}"
    
    print(f"Start processing single checkpoint: {checkpoint_folder_path}")
    print(f"Output directory: {save_folder_path}")
    print(f"Target filename prefix: {full_save_name}")

    # 2. Inference: Generate JSONL file
    # Note: 'list_save_file_name' determines the output filename.
    one_model_vllm_infer_multi_dataset(
        model_checkpoint_folder_path=checkpoint_folder_path,
        list_arrow_folder_path=[ground_truth_arrow_folder_path],
        list_save_file_name=[full_save_name],  # Pass the combined name here
        bool_nat_question=False, 
        save_folder_path=save_folder_path,
        beam_width=beam_width,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        adjust_decimal_places=adjust_decimal_places,
        c_shift_decimal_place=c_shift_decimal_place,
        h_shift_decimal_place=h_shift_decimal_place
    )

    # 3. Convert JSONL to TXT
    jsonl_file_path = os.path.join(save_folder_path, f"{full_save_name}.jsonl")
    txt_file_path = os.path.join(save_folder_path, f"{full_save_name}.txt")
    
    print(f"Converting JSONL to TXT: {jsonl_file_path} -> {txt_file_path}")
    
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_in, \
             open(txt_file_path, 'w', encoding='utf-8') as txt_out:
            for line in jsonl_in:
                data = json.loads(line.strip())
                test_field = data.get('text', [])
                
                # Extract the first generation result and handle escape characters
                if isinstance(test_field, list) and test_field:
                    content = test_field[0].replace('\n', '\\n').replace('\r', '\\r')
                    txt_out.write(content + '\n')
                else:
                    txt_out.write('\n')
    except Exception as e:
        print(f"Error during TXT conversion: {e}")

    # 4. Execute Evaluation
    eval_func_name = "batch_evaluate_multiple_test_files"
    
    # Check if the evaluation script exists
    if not os.path.exists(stat_metrics_scription_file_path):
        print(f"Warning: Evaluation script not found at {stat_metrics_scription_file_path}. Skipping evaluation.")
        return

    # Dynamically import the evaluation function
    imported_funcs = import_functions([(stat_metrics_scription_file_path, eval_func_name)])
    eval_func = imported_funcs.get(eval_func_name, globals().get(eval_func_name))
    
    if eval_func:
        print(f"Starting evaluation using {eval_func_name}...")
        
        # Prepare the mapping dictionary: {Ground Truth Arrow Path : Dataset Name (Filename without extension)}
        # CRITICAL: Since the filename is now "{result_name}_{dataset_name}.txt",
        # the evaluation script will interpret "{result_name}_{dataset_name}" as the key to look up.
        # Therefore, we must map the arrow path to `full_save_name`.
        dict_arrow_mapping = {ground_truth_arrow_folder_path: full_save_name}
        
        try:
            # save_result_as_json_and_csv=True triggers the CSV generation
            eval_func(
                save_folder_path, 
                dict_arrow_mapping, 
                save_result_as_json_and_csv=True
            )
            print("Evaluation finished. CSV files should be in the output folder.")
        except Exception as e:
            print(f"Error during evaluation execution: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Warning: Function '{eval_func_name}' could not be loaded.")


if __name__ == "__main__":
    checkpoint_folder_path = r"/home/sunhnayu/lln/project/deepseek/CLReasoner/train/retrain_for_test/outputs/checkpoint-6200"
    ground_truth_arrow_folder_path = r"/home/sunhnayu/lln/project/deepseek/data/xx_benchmark/test_set/xx_nmr_benchmark_train_subset_197983_depth1_197918_yws4090_subset1000_2508040949"
    dataset_name = "test_set"
    output_folder_path = r"/home/sunhnayu/lln/project/deepseek/CLReasoner/train/retrain_for_test/retrain_for_test"
    result_name = "260204_nmr_structure_elucidation_result"
    stat_metrics_scription_file_path = r"/home/sunhnayu/lln/project/deepseek/CLReasoner/test/stat_metrics_scription.py"
    single_checkpoint_infer_pipeline(
        checkpoint_folder_path=checkpoint_folder_path,
        ground_truth_arrow_folder_path=ground_truth_arrow_folder_path,
        dataset_name=dataset_name,
        output_folder_path=output_folder_path,
        result_name=result_name,
        stat_metrics_scription_file_path = stat_metrics_scription_file_path,
        beam_width = 1,
        max_tokens = 5000,
        temperature = 0,
        top_p = 0.95,
        top_k = 20,
        adjust_decimal_places = False,
        c_shift_decimal_place = 1,
        h_shift_decimal_place = 2
    )
