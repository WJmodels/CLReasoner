import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
from typing import List, Any, Tuple, Dict, Optional, Union, Iterable
from itertools import combinations, permutations
from collections import defaultdict, OrderedDict, Counter
from multiprocessing import Pool
from functools import partial

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors

# I. Load Ground Truth and determine task type
########################   Helper Functions   #######################

def get_files_with_suffix(folder_path: str, suffix: str, traverse_sub_folder: bool = False) -> List[str]:
    """
    Scans a directory for files with a specific extension.

    Args:
        folder_path: The root directory to scan.
        suffix: The file extension to look for (e.g., '.arrow').
        traverse_sub_folder: If True, searches recursively through subdirectories.

    Returns:
        A list of absolute or relative paths to the matching files.
    """
    list_target_file_path = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(suffix):
                list_target_file_path.append(os.path.join(root, file))
        if not traverse_sub_folder:
            break
    return list_target_file_path

def get_df_sample_info_from_arrow_file(arrow_file_path: str) -> pd.DataFrame:
    """
    Loads a HuggingFace Dataset arrow file and converts it into a DataFrame.

    Args:
        arrow_file_path: Path to the .arrow file.

    Returns:
        A pandas DataFrame containing the dataset samples.
    """
    dataset = Dataset.from_file(arrow_file_path)
    df = dataset.to_pandas()
    return df

def judge_task_from_answer_by_keyword(
    list_answer: List[Any], 
    dict_key_task_value_list_task_keyword: Dict[str, List[str]] = {"nmr": [], "retrosyn": ["Reactant", "F1"], "syn": ["Product"]}
) -> str:
    """
    Identifies the chemistry task type based on keywords found in the ground truth answer.

    Args:
        list_answer: A list of ground truth answers.
        dict_key_task_value_list_task_keyword: Mapping of task names to list of identifying keywords.

    Returns:
        The task name (e.g., 'nmr', 'retrosyn', or 'syn').

    Raises:
        ValueError: If keywords overlap between tasks or if no task can be matched.
    """
    if not list_answer:
        raise ValueError("list_answer cannot be empty")
    
    answer_text = str(list_answer[0])
    seen_keywords = set()
    
    for task, keywords in dict_key_task_value_list_task_keyword.items():
        for kw in keywords:
            if kw in seen_keywords:
                raise ValueError(f"Keyword '{kw}' is duplicated across tasks.")
            seen_keywords.add(kw)
    
    for task, keywords in dict_key_task_value_list_task_keyword.items():
        for kw in keywords:
            if kw in answer_text:
                return task
    
    # Return default task (one with no keywords)
    for task, keywords in dict_key_task_value_list_task_keyword.items():
        if not keywords:
            return task
    
    raise ValueError("No task matched and no default task defined.")

def load_question_cot_answer_list_and_str_task_from_arrow_folder(
    arrow_folder_path: str,
    dict_key_task_value_list_task_keyword: Dict[str, List[str]] = {"nmr": [], "retrosyn": ["Reactant", "F1"], "syn": ["Product"]}
) -> Tuple[List[str], List[str], List[str], str]:
    """
    Aggregates all arrow files in a folder and determines the task type.

    Args:
        arrow_folder_path: Folder containing .arrow files.
        dict_key_task_value_list_task_keyword: Keywords for task identification.

    Returns:
        A tuple containing (list of questions, list of CoTs, list of answers, task_type).
    """
    list_arrow_file_path = get_files_with_suffix(arrow_folder_path, ".arrow", traverse_sub_folder=False)
    if not list_arrow_file_path:
        raise ValueError(f"No .arrow files found in '{arrow_folder_path}'")
    
    list_arrow_file_path.sort()
    combined_questions = []
    combined_cots = []
    combined_answers = []
    
    for arrow_file_path in list_arrow_file_path:
        print(f"Loading arrow file: {arrow_file_path}")
        df = get_df_sample_info_from_arrow_file(arrow_file_path)
        combined_questions.extend(df['question'].tolist())
        combined_cots.extend(df['cot'].tolist())
        combined_answers.extend(df['answer'].tolist())
    
    str_task = judge_task_from_answer_by_keyword(combined_answers, dict_key_task_value_list_task_keyword)
    
    # Pre-process Ground Truth placeholders
    if str_task == "retrosyn":
        combined_answers = [_replace_reactant_placeholders(ans) for ans in combined_answers]
    elif str_task == "syn":
        combined_answers = [_replace_product_placeholders(ans) for ans in combined_answers]
        
    print(f"Successfully merged {len(list_arrow_file_path)} files. Total samples: {len(combined_questions)}, Task: {str_task}")
    return combined_questions, combined_cots, combined_answers, str_task

# II. Load Inference Results
def read_multi_jsonl_to_list_list_list_infer_cotent(
    list_jsonl_file_path: List[str],
    extract_key: str = "text",
    fix_escaped_backslash: bool = True,
    beamsearch_top_n_involve: Optional[int] = None,
) -> Tuple[List[List[List[str]]], Dict[int, str]]:
    """
    Reads multiple JSONL files (each from a different model) into a nested structure.

    Args:
        list_jsonl_file_path: List of paths to inference result files.
        extract_key: JSON key containing the list of Top-K strings.
        fix_escaped_backslash: If True, converts '\\\\' to '\\' (common in SMILES JSON exports).
        beamsearch_top_n_involve: Truncate Top-K results to this number.

    Returns:
        A tuple of (3D list [model_idx][question_idx][candidate_idx], index_to_filename_map).
    """
    if not isinstance(list_jsonl_file_path, list) or len(list_jsonl_file_path) == 0:
        raise ValueError("list_jsonl_file_path must be a non-empty list.")

    list_list_list_infer_cotent: List[List[List[str]]] = []
    dict_key_model_index_value_model_result_jsonl_file_path: Dict[int, str] = {}

    for model_idx, file_path in enumerate(list_jsonl_file_path):
        dict_key_model_index_value_model_result_jsonl_file_path[model_idx] = file_path
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found for model index {model_idx}: {file_path}")

        model_questions_topk: List[List[str]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                s = line.strip()
                if not s: continue
                try:
                    obj = json.loads(s)
                except Exception as e:
                    raise ValueError(f"Failed to parse JSON: {file_path} Line {line_no}. Error: {e}")

                topk_list = obj[extract_key]
                topk_str_list: List[str] = []
                for elem in topk_list:
                    if elem is None:
                        topk_str_list.append("")
                    else:
                        elem_str = str(elem)
                        if fix_escaped_backslash and '\\\\' in elem_str:
                            elem_str = elem_str.replace('\\\\', '\\')
                        topk_str_list.append(elem_str)

                if beamsearch_top_n_involve is not None and beamsearch_top_n_involve > 0:
                    topk_str_list = topk_str_list[:beamsearch_top_n_involve]
                model_questions_topk.append(topk_str_list)

        list_list_list_infer_cotent.append(model_questions_topk)
    return list_list_list_infer_cotent, dict_key_model_index_value_model_result_jsonl_file_path

# III. Determine Voting Eligibility
def remove_disordered_tags(s: str, tuple_first_second: Tuple[str, str]) -> str:
    """
    Removes closing tags that appear before an opening tag in a string.

    Args:
        s: Input string.
        tuple_first_second: A tuple of (opening_tag, closing_tag).

    Returns:
        The string with disordered tags removed.
    """
    first, second = tuple_first_second
    first_positions = [match.start() for match in re.finditer(re.escape(first), s)]
    second_positions = [match.start() for match in re.finditer(re.escape(second), s)]

    filtered_s = s
    for pos in second_positions:
        if not any(f_pos < pos for f_pos in first_positions):
            filtered_s = filtered_s[:pos] + filtered_s[pos + len(second):]
    return filtered_s

def extract_str_answer(str_generate_result_without_prompt: str) -> str:
    """
    Parses the raw generation to extract the content between <answer> tags.
    If tags are missing, it attempts to use common chemistry pattern fallbacks.

    Args:
        str_generate_result_without_prompt: Raw text generated by the model.

    Returns:
        The extracted answer string or 'F_' if extraction fails.
    """
    # Clean up tags
    str_generate_result_without_prompt = remove_disordered_tags(str_generate_result_without_prompt, ('<answer>', '</answer>'))
    bool_answer_start = '<answer>' in str_generate_result_without_prompt
    bool_answer_end = '</answer>' in str_generate_result_without_prompt
    bool_specific_str = '])' in str_generate_result_without_prompt
    bool_nat_question_str = 'infer the molecular SMILES:' in str_generate_result_without_prompt
    
    if bool_answer_start and bool_answer_end:
        str_answer = str_generate_result_without_prompt.split('</answer>')[-2].split('<answer>')[-1]
    elif bool_answer_start and not bool_answer_end:
        str_answer = str_generate_result_without_prompt.split('<answer>')[1]
    elif not bool_answer_start and bool_answer_end:
        str_answer = "F_"
    else:
        # Fallback for older prompts/formats
        if bool_nat_question_str:
            str_answer = str_generate_result_without_prompt.split("infer the molecular SMILES:")[-1].split('"')[0]
        elif bool_specific_str:
            str_answer = str_generate_result_without_prompt.split("])")[-1].split('"')[0]
        else:
            str_answer = "F_"
    return str_answer.strip()

def list_list_infer_cotent_to_list_list_infer_answer(list_list_infer_cotent: List[List[str]]) -> List[List[str]]:
    """
    Converts a 2D list of full generated texts into a 2D list of extracted answers.

    Args:
        list_list_infer_cotent: Nested list [question][candidate_text].

    Returns:
        Nested list [question][extracted_answer].
    """
    list_list_infer_answer = []
    for row in list_list_infer_cotent:
        list_list_infer_answer.append([extract_str_answer(str(content)) for content in row])
    return list_list_infer_answer

def extract_formula_from_generate_result(str_generate_result_without_prompt: str) -> Optional[str]:
    """
    Extracts molecular formula (e.g., C6H12O6) from the reasoning trace.

    Args:
        str_generate_result_without_prompt: The model's reasoning text.

    Returns:
        Formula string if found, otherwise None.
    """
    pattern = r"Formula:([^;]+);"
    match = re.search(pattern, str_generate_result_without_prompt)
    return match.group(1).strip() if match else None

def normalize_molecular_formula(formula: str) -> str:
    """
    Parses and normalizes a molecular formula string to a standard Hill system order (C, then H, then alphabetical).

    Args:
        formula: A raw formula string (e.g., 'H2O', '(NH4)2SO4').

    Returns:
        Normalized formula string.
    """
    if not formula: return ""
    import re
    from collections import defaultdict

    def parse_segment_counts(segment: str) -> dict[str, int]:
        s, n, i = segment, len(segment), 0
        def parse_int(idx: int):
            if idx >= n or not s[idx].isdigit(): return None, idx
            j = idx
            while j < n and s[j].isdigit(): j += 1
            return int(s[idx:j]), j
        def parse_element(idx: int):
            if idx >= n or not s[idx].isupper(): return None, idx
            j = idx + 1
            if j < n and s[j].islower(): j += 1
            return s[idx:j], j
        def merge(dst, src, mul=1):
            for k, v in src.items(): dst[k] = dst.get(k, 0) + v * mul
        def parse_block(idx: int):
            out = defaultdict(int)
            while idx < n:
                ch = s[idx]
                if ch in "([{":
                    idx += 1
                    inner, idx = parse_block(idx)
                    if idx < n and s[idx] in ")]}": idx += 1
                    mul, idx = parse_int(idx)
                    merge(out, inner, mul if mul is not None else 1)
                    continue
                if ch in ")]}": return out, idx
                el, j = parse_element(idx)
                if el:
                    cnt, j2 = parse_int(j)
                    out[el] += (cnt if cnt is not None else 1)
                    idx = j2
                    continue
                idx += 1
            return out, idx
        counts, _ = parse_block(0)
        return dict(counts)

    # Basic cleanup
    s = re.sub(r"\s+", "", formula).replace("−", "-")
    s = s.replace("+", "").replace("-", "").replace("^", "")
    parts = re.split(r"[·⋅∙\.•]", s)
    total_counts = defaultdict(int)
    for part in parts:
        if not part: continue
        m = re.match(r"^(\d+)(.+)$", part) # Handles '5H2O' prefix
        if m: part = f"({m.group(2)}){m.group(1)}"
        for el, c in parse_segment_counts(part).items(): total_counts[el] += c

    if not total_counts: return ""
    res = []
    # Standard Hill System: C first, then H
    if "C" in total_counts:
        c = total_counts.pop("C")
        res.append(f"C{c if c > 1 else ''}")
    if "H" in total_counts:
        h = total_counts.pop("H")
        res.append(f"H{h if h > 1 else ''}")
    for el in sorted(total_counts.keys()):
        n = total_counts[el]
        res.append(f"{el}{n if n > 1 else ''}")
    return "".join(res)

def is_valid_smiles_with_formula_check(
    smiles: str, 
    target_formula: str, 
    return_canonical: bool = False, 
    check_length: int = 5000,
    return_special_str_when_smiles_false: str | None = "F_",
    check_nmr_formula: bool = True
) -> Tuple[bool, str]:
    """
    Validates a SMILES string and optionally checks if its formula matches the target.

    Args:
        smiles: The candidate SMILES string.
        target_formula: The formula extracted from the model trace.
        return_canonical: If True, returns the RDKit canonical SMILES.
        check_length: Rejects strings exceeding this length.
        return_special_str_when_smiles_false: String to return on failure.
        check_nmr_formula: If True, compares SMILES formula to target_formula.

    Returns:
        Tuple of (is_valid, final_smiles).
    """
    if not isinstance(smiles, str) or not isinstance(target_formula, str):
        return False, return_special_str_when_smiles_false
    
    normalized_target_formula = normalize_molecular_formula(target_formula)
    if check_length is not None and len(smiles) > check_length:
        return False, return_special_str_when_smiles_false

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return False, return_special_str_when_smiles_false
        
        if check_nmr_formula:
            mol_formula = rdMolDescriptors.CalcMolFormula(mol)
            normalized_mol_formula = normalize_molecular_formula(mol_formula)
            if normalized_mol_formula != normalized_target_formula:
                return False, return_special_str_when_smiles_false

        return True, (Chem.MolToSmiles(mol, canonical=True) if return_canonical else smiles)
    except:
        return False, return_special_str_when_smiles_false

def extract_nmr_list_list_question_formula(list_list_infer_cotent: List[List[str]]) -> List[List[Optional[str]]]:
    """
    Extracts formulas from reasoning traces for a whole batch.

    Args:
        list_list_infer_cotent: Nested list [question][candidate_trace].

    Returns:
        Nested list [question][extracted_formula].
    """
    nmr_list_list_question_formula = []
    for row in list_list_infer_cotent:
        nmr_list_list_question_formula.append([extract_formula_from_generate_result(str(c)) for c in row])
    return nmr_list_list_question_formula

def extract_filter_nmr_list_infer_answer(list_list_infer_cotent: List[List[str]], check_nmr_formula: bool = True) -> List[List[str]]:
    """
    Extracts answers and filters them based on SMILES validity and formula consistency (Specific to NMR).

    Args:
        list_list_infer_cotent: Full traces.
        check_nmr_formula: Whether to enforce formula matching.

    Returns:
        Filtered extracted answers.
    """
    list_list_infer_answer = list_list_infer_cotent_to_list_list_infer_answer(list_list_infer_cotent)
    list_list_question_formula = extract_nmr_list_list_question_formula(list_list_infer_cotent)

    filtered_answers = []
    for ans_row, fml_row in zip(list_list_infer_answer, list_list_question_formula):
        new_row = []
        for smiles_candidate, formula in zip(ans_row, fml_row):
            # formula can be None; is_valid_smiles handles that
            ok, out_smiles = is_valid_smiles_with_formula_check(smiles_candidate, formula or "", check_nmr_formula=check_nmr_formula)
            new_row.append(out_smiles)
        filtered_answers.append(new_row)
    return filtered_answers

def _replace_product_placeholders(content: str) -> str:
    """
    Converts '[Product]' or '[Product1][Product2]' to dot-separated SMILES format.

    Args:
        content: String containing product placeholders.

    Returns:
        String with placeholders replaced by dots where appropriate.
    """
    def _product_repl(m):
        idx = m.group(1)
        return "" if not idx or idx == "1" else "."
    return re.sub(r"\[Product(\d*)\]", _product_repl, content)

def _replace_reactant_placeholders(content: str) -> str:
    """
    Converts '[Reactant]' placeholders to dot-separated SMILES format.

    Args:
        content: String containing reactant placeholders.

    Returns:
        Cleaned string.
    """
    def _reactant_repl(m):
        idx = m.group(1)
        return "" if not idx or idx == "1" else "."
    return re.sub(r"\[Reactant(\d*)\]", _reactant_repl, content)

def is_smilesdotsmiles_valid_smiles(
    smilesdotsmiles: str, 
    check_length: int = 5000, 
    return_special_str_when_smilesdotsmiles_false: str = "F_"
) -> Tuple[bool, str]:
    """
    Validates a dot-separated SMILES string (e.g., mixtures or reactants).

    Args:
        smilesdotsmiles: Candidate string.
        check_length: Threshold for length check.
        return_special_str_when_smilesdotsmiles_false: Error string.

    Returns:
        Tuple of (is_valid, processed_string).
    """
    s = str(smilesdotsmiles).strip()
    if not s: return False, return_special_str_when_smilesdotsmiles_false
    parts = s.split(".")
    # Avoid strings like 'C..C'
    if any(not p.strip() for p in parts): return False, return_special_str_when_smilesdotsmiles_false
    for p in parts:
        sp = p.strip()
        if len(sp) >= check_length or Chem.MolFromSmiles(sp) is None:
            return False, return_special_str_when_smilesdotsmiles_false
    return True, s

def extract_filter_list_list_infer_answer_from_list_list_infer_cotent(
    list_list_infer_cotent: List[List[str]],
    str_task: str,
    check_length: int = 5000,
    return_special_str_when_check_result_false: str = "F_",
    check_nmr_formula: bool = True,
) -> List[List[str]]:
    """
    The unified entry point for extracting and filtering candidate answers based on the task type.

    Args:
        list_list_infer_cotent: The model generated traces.
        str_task: 'nmr', 'retrosyn', or 'syn'.
        check_length: Length threshold for validity.
        return_special_str_when_check_result_false: Placeholder for invalid candidates.
        check_nmr_formula: Whether to use trace formulas for NMR filtering.

    Returns:
        A 2D list of cleaned and filtered SMILES strings.
    """
    task = str_task.strip().lower()
    if task == "nmr":
        return extract_filter_nmr_list_infer_answer(list_list_infer_cotent, check_nmr_formula=check_nmr_formula)

    if task in ["retrosyn", "syn"]:
        raw_answers = list_list_infer_cotent_to_list_list_infer_answer(list_list_infer_cotent)
        final_result = []
        for row in raw_answers:
            filtered_inner = []
            for ans in row:
                ans = str(ans or "").strip()
                # Task specific cleanup
                ans = _replace_reactant_placeholders(ans) if task == "retrosyn" else _replace_product_placeholders(ans)
                # Remove spaces and normalize dots
                ans = re.sub(r"\s+", "", ans)
                ans = re.sub(r"\.{2,}", ".", ans).strip(".")
                
                ok, valid_or_placeholder = is_smilesdotsmiles_valid_smiles(ans, check_length, return_special_str_when_check_result_false)
                filtered_inner.append(valid_or_placeholder)
            final_result.append(filtered_inner)
        return final_result
    raise ValueError(f"Unsupported task: {str_task}")

# IV. Voting Logic
def _canonicalize_smiles(smi: str, dechirality: bool = True, check_length: int = 5000) -> Optional[str]:
    """
    Produces a canonical SMILES string using RDKit for voting comparison.

    Args:
        smi: Raw SMILES string.
        dechirality: If True, removes stereochemical information for more relaxed voting.
        check_length: Maximum allowed length.

    Returns:
        Canonicalized SMILES or None if invalid.
    """
    if not smi or (check_length and len(smi) > check_length): return None
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None: return None
        if dechirality: Chem.RemoveStereochemistry(mol)
        # isomericSmiles=False ensures stereochemistry is not in the output string
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=not dechirality)
    except: return None

def multi_model_one_question_smiles_answer_to_index_answer(
    list_dict_key_model_name_value_topk_list_one_question: List[Dict[str, List[str]]],
    list_dict_key_model_name_value_full_content_list_one_question: List[Dict[str, List[str]]], 
    dechirality: bool = True,
    remove_smiles: str = "F_",
    check_length: int = 5000,
    ignore_order: bool = True, 
) -> Tuple[List[List[List[int]]], Dict[int, Dict[str, List[str]]], Dict[Tuple[int, ...], str]]:
    """
    Converts raw SMILES candidates from multiple models into index-based sequences to handle mixtures.
    Also builds mapping dictionaries to reconstruct original strings later.

    Args:
        list_dict_key_model_name_value_topk_list_one_question: List of candidate answers per model.
        list_dict_key_model_name_value_full_content_list_one_question: List of full generated content per model.
        dechirality: Remove chirality for indexing.
        remove_smiles: Invalid answer placeholder to ignore.
        ignore_order: If True, 'A.B' is indexed same as 'B.A'.

    Returns:
        - Nested list of indices [model][candidate_idx][sequence_of_ids].
        - Mapping from ID to original model-specific string.
        - Mapping from ID sequence (tuple) to full reasoning trace content.
    """
    global_smi2idx: Dict[str, int] = {}
    idx2model2originals = defaultdict(lambda: defaultdict(list))
    tuple_indices_to_full_content: Dict[Tuple[int, ...], str] = {}
    list_multi_model_result_indices = []

    for m_idx, (model_dict, full_dict) in enumerate(zip(list_dict_key_model_name_value_topk_list_one_question, list_dict_key_model_name_value_full_content_list_one_question)):
        for model_name, answers in model_dict.items():
            full_contents = full_dict[model_name]
            model_indexed_answers = []
            for ans_idx, ans_str in enumerate(answers):
                if not ans_str:
                    model_indexed_answers.append([]); continue
                # Split mixtures
                parts = [p.strip() for p in str(ans_str).split(".")]
                parts = [p for p in parts if p and p != remove_smiles]
                one_answer_indices = []
                for raw_piece in parts:
                    std_smi = _canonicalize_smiles(raw_piece, dechirality, check_length)
                    if std_smi is None: continue
                    if std_smi not in global_smi2idx: global_smi2idx[std_smi] = len(global_smi2idx)
                    idx = global_smi2idx[std_smi]
                    one_answer_indices.append(idx)
                    idx2model2originals[idx][model_name].append(raw_piece)
                model_indexed_answers.append(one_answer_indices)
                if one_answer_indices:
                    k = tuple(sorted(one_answer_indices)) if ignore_order else tuple(one_answer_indices)
                    # Cache the first model's full content for this chemical identity
                    if k not in tuple_indices_to_full_content:
                        tuple_indices_to_full_content[k] = full_contents[ans_idx]
            list_multi_model_result_indices.append(model_indexed_answers)
    return list_multi_model_result_indices, {k: dict(v) for k, v in idx2model2originals.items()}, tuple_indices_to_full_content

def vote_list_multi_model_result_list_one_model_result_list_smilesdotsmiles_smiles_index_to_topk_list_list_smiles_index(
    list_multi_model_result_indices: List[List[List[int]]],
    ignore_order: bool,
    force_to_topk: Optional[int] = None,
    add_special_str_when_lack_result_to_fit_topk: str = "F_",
) -> Tuple[List[Union[List[int], str]], Dict[Tuple[int, ...], List[int]]]:
    """
    The core voting engine. Counts occurrences of chemical identities across models.

    Args:
        list_multi_model_result_indices: Nested list of IDs.
        ignore_order: Whether mixture order matters.
        force_to_topk: If set, pad the results to this length.
        add_special_str_when_lack_result_to_fit_topk: Padding string.

    Returns:
        - Sorted list of ID sequences (Top candidates).
        - Mapping from ID sequence to list of model indices that voted for it.
    """
    def key_of(ans): return tuple(sorted(ans)) if ignore_order else tuple(ans)
    counts = defaultdict(int)
    first_pos = {} # Tie breaker: (model_index, candidate_index)
    voters_in_order = defaultdict(list)
    voters_seen = defaultdict(set)

    for m_idx, model_answers in enumerate(list_multi_model_result_indices):
        for a_idx, ans in enumerate(model_answers):
            if not ans: continue
            k = key_of(ans)
            counts[k] += 1
            # Track earliest appearance for tie-breaking
            if k not in first_pos or (m_idx, a_idx) < first_pos[k]: first_pos[k] = (m_idx, a_idx)
            # Track which models voted for this
            if m_idx not in voters_seen[k]:
                voters_seen[k].add(m_idx)
                voters_in_order[k].append(m_idx)

    if not counts:
        res = []
        if force_to_topk: res.extend([add_special_str_when_lack_result_to_fit_topk] * force_to_topk)
        return res, OrderedDict()

    # Sort criteria: 1. Max votes, 2. Earliest model, 3. Earliest position in model
    uniq_keys = sorted(counts.keys(), key=lambda k: (-counts[k], first_pos[k][0], first_pos[k][1]))
    
    topk_list = [list(k) for k in uniq_keys]
    if force_to_topk:
        if len(topk_list) > force_to_topk: topk_list = topk_list[:force_to_topk]
        else: topk_list.extend([add_special_str_when_lack_result_to_fit_topk] * (force_to_topk - len(topk_list)))
    
    mapping = OrderedDict((k, voters_in_order[k]) for k in uniq_keys)
    return topk_list, mapping

def convert_dict_key_tuple_smiles_index_value_vote_this_answer_model_index_to_dict_smiles_value_vote_this_answer_model_index(
    list_multi_model_result_indices: List[List[List[int]]], 
    idx2model2originals: Dict[int, Dict[str, List[str]]], 
    ordered_mapping: Dict[Tuple[int, ...], List[int]], 
    tuple_indices_to_full_content: Dict[Tuple[int, ...], str], 
    ignore_order: bool = True
) -> OrderedDict:
    """
    Reconstructs original reasoning traces or SMILES strings from the integer-based voting results.

    Args:
        list_multi_model_result_indices: Trace ID sequences.
        idx2model2originals: ID to raw string map.
        ordered_mapping: Winning sequences and their voters.
        tuple_indices_to_full_content: Winning sequences and their Reasoning Traces.
        ignore_order: Mixture order flag.

    Returns:
        Ordered dictionary: {ReconstructedContent: {voters: [...], reconstructed_smiles: "..."}}
    """
    # Identify model names from the map
    all_names = sorted({n for idx_d in idx2model2originals.values() for n in idx_d.keys()})
    if not all_names: all_names = ["model0"]
    model_index_to_name = {i: n for i, n in enumerate(all_names)}
    
    def _first_seq(m_idx, key_tuple):
        model_answers = list_multi_model_result_indices[m_idx]
        sorted_target = tuple(sorted(key_tuple)) if ignore_order else key_tuple
        for ans in model_answers:
            current = tuple(sorted(ans)) if ignore_order else tuple(ans)
            if current == sorted_target: return list(ans)
        return list(key_tuple)

    usage_ptr = defaultdict(int)
    out = OrderedDict()
    for key_tuple, voter_indices in ordered_mapping.items():
        search_key = tuple(sorted(key_tuple)) if ignore_order else key_tuple
        # Extract full reasoning trace if available
        full_content = tuple_indices_to_full_content.get(search_key)
        
        # Determine winning SMILES by looking at the first voter's string
        lead_m_idx = voter_indices[0]
        lead_m_name = model_index_to_name.get(lead_m_idx, all_names[0])
        seq_indices = _first_seq(lead_m_idx, key_tuple)
        
        raw_pieces = []
        for idx in seq_indices:
            name2list = idx2model2originals.get(idx, {})
            lst = name2list.get(lead_m_name, [])
            if lst:
                pos = usage_ptr[(idx, lead_m_name)]
                raw_pieces.append(lst[pos] if pos < len(lst) else lst[0])
                usage_ptr[(idx, lead_m_name)] += 1
            else:
                # Fallback to any model's string for this ID
                raw_pieces.append(next(iter(name2list.values()))[0] if name2list else "")
        
        reconstructed = ".".join(raw_pieces)
        # Use Reasoning Trace as Key, or fallback to cleaned SMILES
        out[full_content or reconstructed] = {"votes": voter_indices, "reconstructed": reconstructed}
    return out

def _process_one_question_for_sc(
    question_idx: int, 
    per_model_ans_strict: List[List[str]], 
    per_model_ans_relaxed: List[List[str]], 
    per_model_full: List[List[str]],
    ignore_order: bool, 
    dechirality: bool, 
    force_to_topk: Optional[int], 
    add_special_str: str, 
    check_length: int
) -> OrderedDict:
    """
    Orchestrates the Self-Consistency logic for a single question.
    It first attempts a 'strict' vote (e.g., formula match); 
    if no valid candidate survives, it falls back to 'relaxed' candidates.

    Args:
        question_idx: Question ID.
        per_model_ans_strict: Candidate answers that passed strict filters.
        per_model_ans_relaxed: Candidate answers that passed basic filters.
        per_model_full: Full model generation traces.
        ignore_order: Mixture order flag.
        dechirality: Stereochemical flag for voting.
        force_to_topk: Results padding count.
        add_special_str: Padding value.
        check_length: Max SMILES length.

    Returns:
        Ordered dictionary of results.
    """
    def _run_voting(candidate_answers):
        list_dict_ans = [{f"model{i}": topk} for i, topk in enumerate(candidate_answers)]
        list_dict_full = [{f"model{i}": full} for i, full in enumerate(per_model_full)]
        list_indices, idx2originals, tuple2full = multi_model_one_question_smiles_answer_to_index_answer(
            list_dict_ans, list_dict_full, dechirality, add_special_str, check_length, ignore_order
        )
        _, ordered_mapping = vote_list_multi_model_result_list_one_model_result_list_smilesdotsmiles_smiles_index_to_topk_list_list_smiles_index(
            list_indices, ignore_order, force_to_topk, add_special_str
        )
        return convert_dict_key_tuple_smiles_index_value_vote_this_answer_model_index_to_dict_smiles_value_vote_this_answer_model_index(
            list_indices, idx2originals, ordered_mapping, tuple2full, ignore_order
        )

    # First attempt: Strict voting
    strict_res = _run_voting(per_model_ans_strict)
    top1 = next(iter(strict_res)) if strict_res else add_special_str
    
    # Fallback condition: If Top-1 is invalid (starts with F_), use relaxed filter candidates
    if top1 == add_special_str or top1.startswith(add_special_str):
        return _run_voting(per_model_ans_relaxed)
    return strict_res

def sc_multi_topk_answer_to_sc_topk_answer(
    list_list_list_infer_cotent: List[List[List[str]]], 
    task: str, 
    save_folder_path: str, 
    save_file_name: str,
    ignore_order: bool = True, 
    dechirality: bool = True, 
    force_to_topk: Optional[int] = None, 
    check_length: int = 5000,
    add_special_str_when_lack_result_to_fit_topk: str = "F_", 
    num_process: int = 16, 
    check_nmr_formula: bool = True
) -> Tuple[str, List[OrderedDict]]:
    """
    Runs the multi-model Self-Consistency pipeline in parallel for all questions.

    Args:
        list_list_list_infer_cotent: Input traces [model][question][candidate].
        task: Task type.
        save_folder_path: Output dir.
        save_file_name: Output filename.
        ignore_order: Mixture order flag.
        dechirality: Stereochemical flag.
        force_to_topk: Top candidates to keep.
        check_length: Validity threshold.
        add_special_str_when_lack_result_to_fit_topk: Placeholder.
        num_process: Parallel workers.
        check_nmr_formula: NMR specific filtering.

    Returns:
        - Path to saved results.
        - List of voting result dictionaries per question.
    """
    num_models, num_questions = len(list_list_list_infer_cotent), len(list_list_list_infer_cotent[0])
    list_list_list_strict = []
    list_list_list_relaxed = []
    
    print("Extracting and filtering answers (Strict & Relaxed)...")
    for m_idx in range(num_models):
        # Generate strict list (formula check enabled if task is NMR)
        strict = extract_filter_list_list_infer_answer_from_list_list_infer_cotent(
            list_list_list_infer_cotent[m_idx], task, check_length, add_special_str_when_lack_result_to_fit_topk, check_nmr_formula
        )
        list_list_list_strict.append(strict)
        
        # Generate relaxed list (formula check disabled)
        relaxed = extract_filter_list_list_infer_answer_from_list_list_infer_cotent(
            list_list_list_infer_cotent[m_idx], task, check_length, add_special_str_when_lack_result_to_fit_topk, False
        ) if check_nmr_formula else strict
        list_list_list_relaxed.append(relaxed)

    # Package arguments for parallel map
    per_q_inputs = []
    for q_idx in range(num_questions):
        per_q_inputs.append((
            q_idx, 
            [m[q_idx] for m in list_list_list_strict], 
            [m[q_idx] for m in list_list_list_relaxed], 
            [m[q_idx] for m in list_list_list_infer_cotent], 
            ignore_order, dechirality, 
            force_to_topk, add_special_str_when_lack_result_to_fit_topk, check_length
        ))

    results = []
    # Use ProcessPool to bypass GIL for RDKit/Regex intensive tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_process) as ex:
        futures = [ex.submit(_process_one_question_for_sc, *inp) for inp in per_q_inputs]
        for fut in tqdm(futures, desc="Self-Consistency Parallel Execution"): 
            results.append(fut.result())

    os.makedirs(save_folder_path, exist_ok=True)
    out_path = os.path.join(save_folder_path, f"{save_file_name}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for od in results:
            # Save raw mapping of Voter Counts
            f.write(json.dumps({k: v['votes'] for k, v in od.items()}, ensure_ascii=False) + "\n")
    return out_path, results

# V. Accuracy and Metric Calculations
def _get_target_smiles_from_gt(gt_text: str, task: str) -> str:
    """
    Parses Ground Truth strings to extract only the target SMILES.

    Args:
        gt_text: Raw GT answer.
        task: 'nmr', 'retrosyn', or 'syn'.

    Returns:
        Target SMILES string.
    """
    gt_text = str(gt_text).strip()
    if task == "nmr": return gt_text
    if ">>" in gt_text:
        parts = gt_text.split(">>")
        # retrosyn: Reactant is before '>>' usually, syn: Product is after
        # Note: adjust split logic based on your specific dataset format if needed
        return parts[0].strip() if task == "retrosyn" else parts[-1].strip()
    return gt_text

def _compute_metrics_single_row(args: Tuple) -> Dict[str, Any]:
    """
    Calculates Accuracy and Tanimoto Similarity for a single question.

    Args:
        args: (list_of_predictions, gt_text, task, check_length, max_k).

    Returns:
        Dictionary containing metric results for the row.
    """
    list_preds, gt_text, task, check_length, max_k = args
    res = {"valid": False, "tanimoto": 0.0, "exact_match": False, "top_hits": [False] * max_k}
    if not list_preds: return res
    try:
        gt_target = _get_target_smiles_from_gt(gt_text, task)
        mol_gt = Chem.MolFromSmiles(gt_target)
        if not mol_gt: return res
        
        # Canonical SMILES for exact match (ignoring chirality for metrics)
        can_gt = Chem.MolToSmiles(mol_gt, canonical=True, isomericSmiles=False)
        fp_gt = AllChem.GetMorganFingerprintAsBitVect(mol_gt, 2, nBits=1024)
        
        found_match = False
        for i, pred in enumerate(list_preds[:max_k]):
            if not pred or pred.startswith("F_"): continue
            m = Chem.MolFromSmiles(pred)
            if not m: continue
            
            # Top-1 Validity and Tanimoto
            if i == 0:
                res["valid"] = True
                fp_p = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
                res["tanimoto"] = DataStructs.FingerprintSimilarity(fp_p, fp_gt)
            
            can_p = Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
            if can_p == can_gt:
                found_match = True
                if i == 0: res["exact_match"] = True
            
            # If a match is found at index i, it counts for all Top-K where K > i
            if found_match: 
                for k in range(i, max_k):
                    res["top_hits"][k] = True
                break # Found earliest correct answer, move on
    except: pass
    return res

def calculate_metrics_standard(
    list_list_pred: List[List[str]], 
    list_gt: List[str], 
    task: str, 
    check_length: int = 5000, 
    num_process: int = 16, 
    calc_topk: int = 10
) -> Dict[str, float]:
    """
    Calculates batch-level metrics including Validity Rate, Accuracy, and Tanimoto.

    Args:
        list_list_pred: Predictions per question.
        list_gt: Ground truths.
        task: Task type.
        check_length: Threshold for metrics validation.
        num_process: Parallel workers.
        calc_topk: K value for Top-K metrics.

    Returns:
        Final averaged metrics.
    """
    args = [(p, g, task, check_length, calc_topk) for p, g in zip(list_list_pred, list_gt)]
    with Pool(num_process) as pool: 
        results = pool.map(_compute_metrics_single_row, args)
    
    total = len(results)
    if not total: return {}
    metrics = {
        "prediction_validity_rate": sum(1 for r in results if r["valid"]) / total,
        "accuracy": sum(1 for r in results if r["exact_match"]) / total,
        "average_tanimoto": sum(r["tanimoto"] for r in results) / total
    }
    for k in range(calc_topk): 
        metrics[f"top{k+1}"] = sum(1 for r in results if r["top_hits"][k]) / total
    return metrics

# VI. Pipeline and Save Results
def save_results_aligned(dict_results: Dict[str, Any], save_folder_path: str, save_filename_prefix: str):
    """
    Saves experimental results into a structured JSON and various CSV formats.

    Args:
        dict_results: The nested result dictionary.
        save_folder_path: Dir to save.
        save_filename_prefix: Filename prefix.
    """
    import csv
    json_path = os.path.join(save_folder_path, f"{save_filename_prefix}.json")
    with open(json_path, 'w', encoding='utf-8') as f: 
        json.dump(dict_results, f, indent=4)
    
    # Sort steps numerically if possible
    steps = sorted(dict_results.keys(), key=lambda x: int(x) if x.isdigit() else x)
    all_datasets = sorted({ds for s in steps for ds in dict_results[s].keys()})
    core_metrics = ["accuracy", "average_tanimoto", "prediction_validity_rate"]

    # Wide Formatted CSV (Hierarchical structure for spreadsheets)
    fmt_path = os.path.join(save_folder_path, f"{save_filename_prefix}_formatted.csv")
    with open(fmt_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([""] + [ds for ds in all_datasets for _ in core_metrics])
        writer.writerow(["metric"] + core_metrics * len(all_datasets))
        writer.writerow(["train_step"] + [""] * (len(all_datasets) * len(core_metrics)))
        for s in steps:
            row = [s]
            for ds in all_datasets:
                data = dict_results[s].get(ds, {})
                for m in core_metrics:
                    val = data.get(m, 0)
                    row.append(f"{val:.4f}" if isinstance(val, (int, float)) else str(val))
            writer.writerow(row)

    # Standard Accuracy Only CSV
    acc_path = os.path.join(save_folder_path, f"{save_filename_prefix}_accuracy.csv")
    with open(acc_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["train_step"] + all_datasets)
        for s in steps:
            row = [s]
            for ds in all_datasets:
                acc = dict_results[s].get(ds, {}).get('accuracy', 0)
                row.append(f"{acc:.4f}")
            writer.writerow(row)
    print(f"Results saved to {save_folder_path} with prefix {save_filename_prefix}")

def cot_sc_multi_jsonl_topk_result_to_dict_key_str_topk_value_float_acc(
    arrow_folder_path: str,
    list_jsonl_file_path: List[str],
    save_folder_path: str,
    save_folder_name: str,
    topk: int,
    ignore_order: bool = True,
    dechirality: bool = True,
    force_to_topk: Optional[int] = None,
    check_length: int = 5000,
    num_process: int = 8,
    extract_key: str = "text",
    fix_escaped_backslash: bool = True,
    add_special_str_when_lack_result_to_fit_topk: str = "F_",
    check_nmr_formula: bool = True
) -> Dict[str, float]:
    """
    The main high-level API to execute multi-model Self-Consistency voting and calculate Top-K accuracy.

    Args:
        arrow_folder_path: Ground truth path.
        list_jsonl_file_path: Inference result paths from multiple models.
        save_folder_path: Root output directory.
        save_folder_name: Sub-folder for this experiment.
        topk: K for Top-K accuracy.
        ignore_order: Mixture order flag for voting.
        dechirality: Stereochemical flag for voting.
        force_to_topk: Candidate count per question.
        check_length: Max SMILES length.
        num_process: Worker count.
        extract_key: JSON key for candidates.
        fix_escaped_backslash: JSON escape fix.
        add_special_str_when_lack_result_to_fit_topk: Fail placeholder.
        check_nmr_formula: NMR formula trace check.

    Returns:
        Summary metrics dictionary.
    """
    target_dir = os.path.join(save_folder_path, save_folder_name)
    os.makedirs(target_dir, exist_ok=True)
    
    print("="*80 + "\nStarting Multi-Model Self-Consistency Voting Pipeline\n" + "="*80)
    
    # 1. Load Ground Truth
    _, _, list_gt, str_task = load_question_cot_answer_list_and_str_task_from_arrow_folder(arrow_folder_path)
    
    # 2. Load Multi-Model Content
    list_content, _ = read_multi_jsonl_to_list_list_list_infer_cotent(
        list_jsonl_file_path, extract_key, fix_escaped_backslash
    )
    
    # 3. Perform Voting
    _, voting_results = sc_multi_topk_answer_to_sc_topk_answer(
        list_content, 
        str_task, 
        target_dir, 
        "sc_results", 
        ignore_order, 
        dechirality, 
        force_to_topk, 
        check_length, 
        add_special_str_when_lack_result_to_fit_topk, 
        num_process, 
        check_nmr_formula
    )

    # 4. Prepare for Metric Calculation
    sc_preds = []
    for q_res in voting_results:
        # Extract reconstructed SMILES for each surviving candidate
        row = [v['reconstructed'] for k, v in q_res.items() if not (k.startswith("F_") and len(k) < 5)]
        sc_preds.append(row if row else ["F_"])

    # 5. Finalize Metrics
    metrics = calculate_metrics_standard(sc_preds, list_gt, str_task, check_length, num_process, topk)
    print("\nFinal Averaged Metrics:")
    for k, v in metrics.items(): 
        print(f"  {k}: {v:.4f}")
    
    with open(os.path.join(target_dir, "summary_metrics.json"), "w") as f: 
        json.dump(metrics, f, indent=2)
        
    return metrics


if __name__ == "__main__":
    # Example code for chemistry_aware_self_consistency
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Path to the Ground Truth Arrow dataset folder
    gt_path = os.path.abspath(os.path.join(SCRIPT_DIR, r"../sample/chemistry_aware_self_consistency_sample_ground_truth/RP_test_arrow"))

    # Inference results (JSONL files) used for chemistry-aware self-consistency
    # Note: The list order represents priority during tie-breaking; 
    # models appearing earlier in the list are given higher priority.
    raw_inference_paths = [
        r"../sample/chemistry_aware_self_consistency_sample_infer_result_for_sc/RP-Multi-CoT-4_weight.jsonl",
        r"../sample/chemistry_aware_self_consistency_sample_infer_result_for_sc/RP-Multi-CoT-3_weight.jsonl",
        r"../sample/chemistry_aware_self_consistency_sample_infer_result_for_sc/RP-Multi-CoT-5_weight.jsonl",
        r"../sample/chemistry_aware_self_consistency_sample_infer_result_for_sc/RP-Multi-CoT-2_weight.jsonl",
        r"../sample/chemistry_aware_self_consistency_sample_infer_result_for_sc/RP-Multi-CoT-1_weight.jsonl",
        r"../sample/chemistry_aware_self_consistency_sample_infer_result_for_sc/RP-Single-CoT-7_weight.jsonl",
        r"../sample/chemistry_aware_self_consistency_sample_infer_result_for_sc/RP-Multi-CoT-6_weight.jsonl",
        r"../sample/chemistry_aware_self_consistency_sample_infer_result_for_sc/RP-Single-CoT-4_weight.jsonl",
        r"../sample/chemistry_aware_self_consistency_sample_infer_result_for_sc/RP-Single-CoT-5_weight.jsonl",
        r"../sample/chemistry_aware_self_consistency_sample_infer_result_for_sc/RP-Single-CoT-2_weight.jsonl",
        r"../sample/chemistry_aware_self_consistency_sample_infer_result_for_sc/RP-Multi-CoT-7_weight.jsonl",
        r"../sample/chemistry_aware_self_consistency_sample_infer_result_for_sc/RP-Single-CoT-3_weight.jsonl",
        r"../sample/chemistry_aware_self_consistency_sample_infer_result_for_sc/RP-Single-CoT-1_weight.jsonl",
        r"../sample/chemistry_aware_self_consistency_sample_infer_result_for_sc/RP-Single-CoT-6_weight.jsonl",
    ]
    inference_files = [os.path.abspath(os.path.join(SCRIPT_DIR, p)) for p in raw_inference_paths]
    
    # Path to the output root directory
    results_root = os.path.abspath(os.path.join(SCRIPT_DIR, r"../sample/chemistry_aware_self_consistency_sample_output"))
    # Name of the output experiment/result subfolder
    experiment_name = "chemistry_aware_self_consistency_sample_output"

    # Execution
    final_metrics = cot_sc_multi_jsonl_topk_result_to_dict_key_str_topk_value_float_acc(
        arrow_folder_path=gt_path,
        list_jsonl_file_path=inference_files,
        save_folder_path=results_root,
        save_folder_name=experiment_name,
        topk=10,
        num_process=16,
        check_nmr_formula=True # Set to False if formula consistency check is not needed (e.g., for synthesis tasks)
    )