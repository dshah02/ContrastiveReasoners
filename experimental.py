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
from rewards import get_extraction_rew_func
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

args = parser.parse_args()
model_name = args.model
critic_name = args.critic
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

##BASICALLY NONE OF THE CONFIG IS USED ANYMORE 

max_input_len = 1200 #int(config['max_seq_length'])
max_output_len = 1024
lora_rank =  32 #int(config['lora_rank'])
use_easy_dataset = bool(config['easy_dataset'])
steps = int(config['steps'])
# state_format_loss = bool(config['state_format_loss'])
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cache_dir,
        max_seq_length=max_input_len + max_output_len,
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

def reward_func(completions, item, **kwargs):
    reward = []
    correct = []
    print(f"Input numbers: {item[0]['nums']}")
    print(f"Target: {item[0]['target']}")
    print("-" * 40)
    for pred, it in zip(completions, item):
        res, pred_sol = eval_func(pred, data_item = it)
        acc, partial_acc, extraction_rate = res["accuracy"], res["partial_accuracy"], res["extraction_rate"]
        correct.append(acc)
       
        print(f"Parsed output: {pred_sol['parsed_output']}")
        print("-" * 40)
        reward.append(4 * acc)
    print(f"Correct: {sum(correct)/len(correct)}")
    return reward

extraction_rew = get_extraction_rew_func(eval_func)

def extraction_rew(completions, item, **kwargs):
    reward = []
    correct = []
    for pred, it in zip(completions, item):
        res, pred_sol = eval_func(pred, data_item = it)
        acc, partial_acc, extraction_rate = res["accuracy"], res["partial_accuracy"], res["extraction_rate"]
        reward.append(extraction_rate)
    return reward

#__________________________________________________________________________________________________________________

from utils import extract_search_procedure, sag_extract

buffer = []
bsz = 16
norm  =1000
if critic_name in ['rho']:
    critic = ContrastiveCritic6('microsoft/rho-math-1b-interpreter-v0.1')
    norm = 0.01
elif critic_name in ['phi']:
    critic = ContrastiveCritic6('microsoft/Phi-4-mini-reasoning')
    norm = 0.01
else:
    critic = ContrastiveCritic()
    norm = 100

critic.to('cpu')
optimizer = torch.optim.Adam(critic.parameters(), lr=1e-5)

import random
random.seed()
random_id = random.randint(1000, 9999)
print(f"Generated random ID for critic: {random_id}")

counter= [0]
losses = []
q_values = []
def critic_func(completions, item, **kwargs):
    counter[0] += 1
    if counter[0] % 100 == 0 and args.critic != "":
        os.makedirs(f"/scratch/gpfs/ds6237/critic_checkpoints/{args.critic}", exist_ok=True)
        torch.save(critic.action_mlp.state_dict(), f"/scratch/gpfs/ds6237/critic_checkpoints/{args.critic}/action_mlp_{counter[0]}_{random_id}.pt")
        torch.save(critic.goal_mlp.state_dict(),  f"/scratch/gpfs/ds6237/critic_checkpoints/{args.critic}/goal_mlp_{counter[0]}_{random_id}.pt")
        print(f"Saved critic checkpoints at step {counter[0]}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    reward = []
    for pred, it in zip(completions, item):
        res, pred_sol = eval_func(pred, data_item = it)
        acc, partial_acc, extraction_rate = res["accuracy"], res["partial_accuracy"], res["extraction_rate"]
        search_proc, indentation = extract_search_procedure(pred)
        if not search_proc:
            reward.append(0)
            continue
        sag_tuples = sag_extract(search_proc, indentation)        
        

        if sag_tuples:
            for tup in sag_tuples:
                buffer.append((tup, it['target']))
        critic.to(device)    
        
        
        sample_size = min(bsz, len(buffer))
        tuples = random.sample(buffer, sample_size)
        
        states = []
        actions = []
        goals = []
        
        for sag, target in tuples:
            s,a,g = sag
            states.append(s)
            actions.append(a)
            goals.append(g)
        loss = None
        if len(states) > 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            loss = critic.train_step(states, actions, goals, optimizer)
            print(f"Critic training loss: {loss}")
            losses.append(loss)
    
        critic.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
        avg_q_value = 0
        if sag_tuples:
            eval_states = []
            eval_actions = []
            eval_goals = []
            target = it['target']
            
            for s,a,g in sag_tuples:
                eval_states.append(s)
                eval_actions.append(a)
                eval_goals.append(f'= {target}. Evaluate {target} == {target}, target found!')
            
            critic.to(device)
            
            with torch.no_grad():
                critic.eval()
                Q_values = critic.forward(eval_states, eval_actions, eval_goals)
                critic.train()
                
                diag_values = torch.diag(Q_values)
                avg_q_value = diag_values.mean().item()
                print(f"Average Q value for extracted tuples: {avg_q_value:.4f}")
                reward.append(Q_values.mean().item() * min(counter[0]/(100 * norm), 1/norm))
                q_values.append(Q_values.mean().item())
            
            critic.to('cpu')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
                reward.append(0)
        
        if counter[0] % 10 == 0 and args.critic != "":
            try:
                metrics_file = f"/scratch/gpfs/ds6237/critic_checkpoints/{args.critic}/metrics_{counter[0]}_{random_id}.json"
                metrics = {
                    "step": counter[0],
                    "q_values": q_values,
                    "critic_loss": losses,
                    "buffer_size": len(buffer)
                }
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f)
                print(f"Saved critic metrics at step {counter[0]}")
            except Exception as e:
                print(f"Error saving critic metrics at step {counter[0]}: {str(e)}")
    
    return reward

import random
random.seed()
run_id = random.randint(1000, 9999)
output_dir = f"CR/exp/outputs_{run_id}_{model_name}"

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
    num_generations=4 if args.critic != "phi" else 2,
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

try:
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_func,
            extraction_rew,
            critic_func
        ],
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    
    
    print(f"Saving model with ID {run_id}...")
    trainer.model.save_pretrained(output_dir=f"{store_dir}/models/{output_dir}/final_model")
    
except Exception as e:
    print(f"Error: {e}")
