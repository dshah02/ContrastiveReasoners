from unsloth import FastLanguageModel
import torch
import re
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import json
import os
import logging
import yaml
import argparse

# Set environment variables for offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

parser = argparse.ArgumentParser(description='Run the experimental model with configuration')
parser.add_argument('--config', type=str, default='config_0.yaml', help='Path to the configuration file')
args = parser.parse_args()

# Increase logging to debug issues
logging.basicConfig(level=logging.INFO)

# Set caching and model parameters
cache_dir = "./cache"  # Local cache directory

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Log configuration values
logging.info(f"Configuration loaded from {args.config}:")
logging.info(f"max_seq_length: {config['max_seq_length']}")
logging.info(f"lora_rank: {config['lora_rank']}")
logging.info(f"easy_dataset: {config['easy_dataset']}")


max_seq_length = int(config['max_seq_length'])
lora_rank = int(config['lora_rank'])
use_easy_dataset = bool(config['easy_dataset'])

# Configure to skip VLLM for offline use
# The key is to bypass VLLM's auto-detection which tries to access HF Hub
print("Loading model from local cache...")
try:
    # Load the cached model and tokenizer from disk directly
    # We'll specify additional parameters to avoid online lookups
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cache_dir,  # Using the local folder 
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,  # This might trigger VLLM
        prefer_vllm=False,    # Explicitly avoid VLLM which requires HF API calls
        tokenizer_path=cache_dir,  # Specify explicit tokenizer path
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.8,
        local_files_only=True,
        trust_remote_code=True,  # Trust code from cache
        use_safetensors=True,    # Use safetensors when available
    )
except RuntimeError as e:
    print(f"Error loading with fast_inference: {e}")
    print("Trying alternative loading method...")
    # Fallback method without fast_inference which might use VLLM
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cache_dir,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,  # Disable fast inference to avoid VLLM
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,
        local_files_only=True,
        trust_remote_code=True,
    )

# Re-apply PEFT modifications
print("Applying PEFT modifications...")
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


from strings import SYSTEM_PROMPT, XML_COT_FORMAT, MINI_SYS_PROMPT

def get_questions():
    import json
    with open('dataset_hard.json' if not use_easy_dataset else 'dataset_easy.json', 'r') as f:
        data = json.load(f)
    
    processed_data = []
    for item in data:
        numbers_str = ', '.join(map(str, item['numbers']))
        question = f"Given the numbers {numbers_str}, reach the target number {item['target']} using +, -, *, and / operations and using each number once."
        processed_data.append({
            'prompt': [
                {'role': 'system', 'content': MINI_SYS_PROMPT},
                {'role': 'user', 'content': question}
            ],
            'answer': item['target']
        })
    
    return processed_data

dataset = get_questions()
dataset = Dataset.from_list(dataset)

# Helper functions for reward calculation
def extract_xml_answer(text: str) -> str:
    if "<answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()
    

from strings import evaluate_expression, is_valid_expression, extract_states

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]: 
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    resp = [extract_xml_answer(r) for r in responses]
    user_messages = [p[1]['content'] for p in prompts]    
    numbers_parts = [um.split("Given the numbers ")[1].split(", reach the target")[0] for um in user_messages]
    numbers_list = [[int(num.strip()) for num in nm.split(",")] for nm in numbers_parts]
    targets = [int(um.split("target number ")[1].split(" using")[0]) for um in user_messages]
    
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{resp[0]}")
    output =  [10.0 if evaluate_expression(res, inp, ans) else 0 for res, inp, ans in zip(resp, numbers_list, targets)] 
    # Print successful cases where output is 10.0
    for i, (out, res, inp) in enumerate(zip(output, resp, numbers_list)):
        if out == 10.0:
            print(f"Successful case {i}:")
            print(f"Input numbers: {inp}")
            print(f"Valid solution: {res}")
    
    return output

    
def format_reward_func(completions, **kwargs) -> list[float]: 
    responses = [completion[0]['content'] for completion in completions]
    resp = [extract_xml_answer(r) for r in responses]

    return [0.5 if is_valid_expression(res) else 0 for res in resp] 
    
from verify_states import is_valid_state

def validation_reward_func(completions, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    state_list = [extract_states(res) for res in responses]
    print("STATE LIST", state_list)
    outputs = []
    for sl in state_list:
        broke = False
        if len(state_list) < 3: #less than 3 states means wrong almost surely
            outputs.append(-2)
        else:
            for l in range(1, len(sl)-1):
                if not is_valid_state(sl[l], sl[:l]):
                    broke = True
            outputs.append([-1 if broke else 0])
    return outputs
    
# Configure training
max_prompt_length = 400
import random
# Save the trained model with unique ID
run_id = random.randint(1000, 9999)

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=2, #before was 6
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    output_dir=f"models/outputs_{run_id}",
    # Add these parameters to ensure offline mode
    hub_model_id=None,  # Disable Hugging Face Hub integration
    push_to_hub=False,  # Don't try to push to Hub
)

# Verify model loaded correctly before training
print(f"Model type: {type(model).__name__}")
print(f"Tokenizer type: {type(tokenizer).__name__}")

# Create and start the trainer with error handling
print("Starting training...")
try:
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            # xmlcount_reward_func,
            # soft_format_reward_func,
            # strict_format_reward_func,
            # int_reward_func,
            correctness_reward_func,
            format_reward_func,
            validation_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    
    
    print(f"Saving trained model with ID {run_id}...")
    trainer.model.save_pretrained(f"./models/outputs_{run_id}/final_model")
    print("Training completed successfully!")
    
except Exception as e:
    print(f"Error during training: {e}")
    import traceback
    traceback.print_exc()