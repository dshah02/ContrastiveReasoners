from unsloth import FastLanguageModel
import torch
import re
from trl import GRPOConfig, GRPOTrainer

model_name = "meta-llama/meta-Llama-3.1-8B-Instruct"
max_seq_length = 2048 # Can increase for longer reasoning traces
lora_rank = 32 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)




# Load and prep dataset
SYSTEM_PROMPT = """
We will be playing the game 24. In this game you will start with 4 numbers from 1 to 10 and aim to generate a target number.
To generate the target number, you will combine the 4 starting numbers using +, -, *, and /. 
Your reasoning may consist of many steps where you combine different numbers, after each combination, represent the current state between <state> and </state> tags.
Put your answer in between <answer> and </answer> tags:

For example, if the 4 numbers are 3, 3, 5, 5, and your target was 30, you would go about this with

1. 3 * 5 = 15, so we can update state to <state> 15, 3, 5 </state>
2. 15 * 5 = 75, so we can update state to <state> 75, 3 </state>
3. This seems larger than our goal of 30, however we could combine with 5 with 3 to get <state> 15, 15 </state>
4. Ah and 15 + 15 = 30, so we can reach <state> 30 </state>

<answer>
(3 * 5) + (3 * 5) = 30
</answer>
"""

XML_COT_FORMAT = """
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def get_questions():
    import json
    with open('dataset.json', 'r') as f:
        data = json.load(f)
    
    processed_data = []
    for item in data['dataset']:
        numbers_str = ', '.join(map(str, item['numbers']))
        question = f"Given the numbers {numbers_str}, reach the target number {item['goal']} using +, -, *, and / operations."
        processed_data.append({
            'prompt': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': question}
            ],
            'answer': item['solution']
        })
    
    return processed_data

dataset = get_questions()

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    # For the 24 game, we'll consider the answer correct if it matches the solution
    return [2.0 if r.strip() == a.strip() else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # For the 24 game, we want to reward valid mathematical expressions
    return [0.5 if any(op in r for op in ['+', '-', '*', '/']) else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


max_prompt_length = 512

import random
# Save the trained model with unique ID
run_id = random.randint(1000, 9999)

training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 6, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = f"outputs_{run_id}",
)


trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.train()


print(f"Saving trained model with ID {run_id}...")
trainer.model.save_pretrained(f"./outputs_{run_id}/final_model")
print("Training completed successfully!")