###TAKEN FROM HERE
### https://github.com/princeton-pli/LongProc/blob/main/longproc


from typing import Tuple, List, Dict, Callable

import json
import os
import yaml
import re
from countdown_evaluator import (
    build_countdown_demonstration,
    evaluate_countdown_final_solution,
    evaluate_countdown_search_procedure
)


def _extract_with_tag(response: str, tag: str):
    start = response.find(f"<{tag}>")
    end = response.find(f"</{tag}>")
    if start == -1 or end == -1:
        return None
    return response[start+len(tag)+2:end].strip()



def eval_countdown(prediction: str, example = None, data_item = None):
    """
    Returns: metrics (dict) and additional info to update the original sample with (dict)
    """
    if data_item is None:
        data_item = example["item"]
    pred_solution = _extract_with_tag(prediction, "Solution")
    nums = data_item["nums"]
    target = data_item["target"]
    if pred_solution is not None and evaluate_countdown_final_solution(nums, target, pred_solution):
            return {"accuracy": 1.0, "partial_accuracy": 1.0, "extraction_rate": 1.0}, {"parsed_output": pred_solution,}

    extraction_rate = 1.0 if pred_solution is not None else 0.0
    # handle probably unclosed search procedure
    if "# Search Procedure" not in prediction:
        return {"accuracy": 0.0, "partial_accuracy": 0.0, "extraction_rate": extraction_rate}, {"parsed_output": None,}

    pred_procedure = prediction.split("# Search Procedure")[-1].strip()

    ground_truth_procedure = data_item["reference_output"]
    # evaluate the procedure
    gt_procedure = ground_truth_procedure.split("# Search Procedure")[-1].split("Now we have found the target")[0].strip()

    partial_accuracy, error_report = evaluate_countdown_search_procedure(nums, target, pred_procedure, gt_procedure)
    return {"accuracy": 0.0, "partial_accuracy": partial_accuracy, "extraction_rate": extraction_rate}, {"parsed_output": pred_solution, "error_report": error_report}


def _load_countdown_data(dataset_name: str, path: str=None) -> Tuple[Dict, Callable]:
    assert dataset_name in ["countdown_0.5k", "countdown_2k", "countdown_8k"]

    if path is None: path = "longproc_data"

    path = os.path.join(path, "countdown")

    data_file = os.path.join(path, dataset_name + ".json")
    with open(data_file, "r") as f:
        data = json.load(f)

    with open(os.path.join(path, "prompts.yaml"), "r") as f:
        prompt = yaml.safe_load(f)
        user_prompt = prompt['USER_PROMPT']

    def build_icl_demonstration():
        _DEMO_SET = [
            {"nums": [40, 19, 23, 7], "target": 29,},
            # {"nums": [9, 16, 6, 18], "target": 12,},
        ]
        examples = []
        for demo in _DEMO_SET:
            _, demonstration = build_countdown_demonstration(demo["nums"], demo["target"])
            examples.append(f"# Example\nNumbers: {demo['nums']}\nTarget: {demo['target']}\n\n{demonstration}")
        examples = "\n\n".join(examples)
        return examples

    # partially fill the user prompt
    user_prompt = user_prompt.format(demonstration=build_icl_demonstration(), nums="{nums}", target="{target}")

    data_purged = []
    for d in data:
        nums = d["nums"]
        target = d["target"]
        solution, demonstration = build_countdown_demonstration(nums[:], target)
        solution_str = "\n".join(solution)
        assert evaluate_countdown_final_solution(nums, target, solution_str), f"Failed to evaluate solution {solution_str}"
        data_purged.append({
            "nums": nums,
            "target": target,
            "solution": solution,
            "reference_output": demonstration,
        })


    return {
        "data": data_purged,
        "prompt_template": user_prompt,
    }, eval_countdown


def load_longproc_data(dataset_name: str, path: str=None) -> Tuple[List, Callable]:
    """
    Load the dataset and evaluation function given the dataset name and path.
    returns: list of data, evaluation function
    the data list will contain {"input_prompt", "reference_output", and "item"} for each data point
    """

    dataset_basename = dataset_name.rsplit("_", 1)[0]

    dataset_loaders = {
        "countdown": _load_countdown_data,
    }
    
    if dataset_basename in dataset_loaders:
        dataset_loading_func = dataset_loaders[dataset_basename]
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    packed_data, eval_func = dataset_loading_func(dataset_name, path)

    template = packed_data["prompt_template"]
    data = packed_data["data"]

    upacked_data = []
    for d in data:
        upacked_data.append({
            "input_prompt": template.format(**d),
            "reference_output": d["reference_output"],
            "item": d
        })

    assert all(["input_prompt" in d and "reference_output" in d and "item" in d for d in upacked_data])

    return upacked_data, eval_func
