###MODIFIED FROM HERE: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb

from unsloth import FastLanguageModel
import torch
import re
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import json
import os
import logging
import yaml
import random
import argparse
from critics.critic import ContrastiveCritic
from critics.critic6 import ContrastiveCritic6

from longproc_data import load_longproc_data

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config', type=str, default='configs/config_0.yaml')
parser.add_argument('--model', type=str, default = "llama", required=False)
parser.add_argument('--critic', type=str, default = "", required=False)
parser.add_argument('--steps', type=int, default = 100, required=False)

args = parser.parse_args()
model_name = args.model
critic_name = args.critic
steps = args.steps
logging.basicConfig(level=logging.INFO)

store_dir = "../../../scratch/gpfs/ds6237/"

if "gemma" == model_name:
    model_name = "google/gemma-3-4b-it"
    cache_dir = f"{store_dir}/cache/gemma-4b"
elif "qwen" == model_name:
    model_name = "Qwen/Qwen2.5-7B"
    cache_dir = f"{store_dir}/cache/qwen-2-5-7b"
elif "r1-qwen" == model_name:
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
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
max_output_len = 2048
max_seq_length = max_input_len + max_output_len
lora_rank =  32
use_easy_dataset = bool(config['easy_dataset'])

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cache_dir,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
        prefer_vllm=False,
        tokenizer_path=cache_dir,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,
        local_files_only=True,
        trust_remote_code=True,
        use_safetensors=True,
    )
except RuntimeError as e:
    print(f"Error loading with fast_inference: {e}")
    print("Trying alternative loading method...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cache_dir,
        max_seq_length=max_seq_length,
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


EOS_TOKEN = tokenizer.eos_token

old_dataset, eval_func = load_longproc_data('countdown_2k', ".")
formatted_data = []
for item in old_dataset[:steps]:
    prompt = item['input_prompt']
    completion = item['reference_output']
    formatted_data.append({
        "text": f"{prompt}{completion}{EOS_TOKEN}"
    })
dataset = Dataset.from_list(formatted_data)

import random
random.seed()
run_id = random.randint(1000, 9999)
output_dir = f"outputs_{run_id}_{model_name}"

os.makedirs(f"CR/models/finetune/{output_dir}", exist_ok=True)

try:
    os.makedirs(f"{store_dir}/models/finetune/{output_dir}", exist_ok=True)
    print(f"Created output directory: {store_dir}/models/finetune/{output_dir}")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=steps,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=f"{store_dir}/models/finetune/{output_dir}",
            report_to="none",
        ),
    )

    trainer.train()
    
    print(f"Saving to{run_id}...")
    trainer.model.save_pretrained(save_directory=f"{store_dir}/models/finetune/{output_dir}/final_model")
    
except Exception as e:
    pass