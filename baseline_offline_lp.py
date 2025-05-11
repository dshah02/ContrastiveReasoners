###MODIFIED FROM HERE: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb

from unsloth import FastLanguageModel
import torch
import re
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import json
import os
import logging
import yaml
import random
import argparse
from longproc_data import load_longproc_data
from rewards import get_reward_func, get_partial_acc_func, get_extraction_rew_func


os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config', type=str, default='configs/config_0.yaml')
parser.add_argument('--model', type=str, default = "llama", required=False)

args = parser.parse_args()
model_name = args.model

logging.basicConfig(level=logging.INFO)

store_dir = "../../../scratch/gpfs/ds6237/"

if "gemma" == model_name: #unsupported by unsloth for now
    model_name = "google/gemma-3-4b-it"
    cache_dir = f"{store_dir}/cache/gemma-4b"
elif "qwen" == model_name:
    model_name = "Qwen/Qwen2.5-7B"
    cache_dir = f"{store_dir}/cache/qwen-2-5-7b"
elif "r1-qwen" == model_name:
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" #works but says 5b
    cache_dir =  f"{store_dir}/cache/r1-qwen"
else:
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct"
    cache_dir =  f"{store_dir}/cache/llama-3-1-8b"

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

logging.info(f"Configuration loaded from {args.config}:")
logging.info(f"max_seq_length: {config['max_seq_length']}")
logging.info(f"lora_rank: {config['lora_rank']}")
logging.info(f"easy_dataset: {config['easy_dataset']}")
logging.info(f"steps: {config['steps']}")
logging.info(f"state_format_loss: {config['state_format_loss']}")


max_input_len = 1200
max_output_len = 1024
lora_rank = int(config['lora_rank'])
use_easy_dataset = bool(config['easy_dataset'])
steps = int(config['steps'])
state_format_loss = bool(config['state_format_loss'])

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cache_dir, 
        max_seq_length=max_input_len + max_output_len,
        load_in_4bit=True,
        fast_inference=True, 
        prefer_vllm=False,  
        tokenizer_path=cache_dir, 
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.8,
        local_files_only=True,
        trust_remote_code=True,
        use_safetensors=True,  
    )
except RuntimeError as e:
    print(f"fast inmference error {e}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cache_dir,
        max_seq_length=max_input_len + max_output_len,
        load_in_4bit=True,
        fast_inference=False,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,
        local_files_only=True,
        trust_remote_code=True,
    )

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

dataset, eval_func = load_longproc_data('countdown_0.5k', ".")
random.shuffle(dataset)

def truncate_prompt(prompt, max_length):
    tokens = tokenizer.encode(prompt)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        return tokenizer.decode(tokens)
    return prompt

for item in dataset:
    prompt = item.pop("input_prompt")
    trunc_prompt = truncate_prompt(prompt, max_input_len)
    if trunc_prompt != prompt:
        print("TRUNACATED :(", prompt, trunc_prompt)
        print('-'*20)
    item["prompt"] = prompt

dataset = dataset[:steps]
print("Example", dataset[0]['item']['nums'], dataset[0]['item']['target'], dataset[0]['prompt'])

import random
random.seed()
run_id = random.randint(1000, 9999)
output_dir = f"CR/baseline/outputs_{run_id}_{model_name}"

reward_func = get_reward_func(eval_func)
partial_acc_func = get_partial_acc_func(eval_func)
extraction_rew = get_extraction_rew_func(eval_func)

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
    num_generations=4, #proportional to speed
    max_prompt_length=max_input_len, 
    max_completion_length=max_output_len,  
    max_steps=steps,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="none",
    output_dir=f"{store_dir}/models/{output_dir}",
    hub_model_id=None,
    push_to_hub=False,
)

print("training started")
try:
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_func,
            # partial_acc_func,
            extraction_rew
        ],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()    
    print(f"Saving with ID {run_id}...")
    trainer.model.save_pretrained(output_dir=f"{store_dir}/models/{output_dir}/final_model")

    
except Exception as e: #in case something crashes
    print(f"Error during training: {e}")
    