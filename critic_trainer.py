import torch
import json
import argparse
from critics.critic6 import ContrastiveCritic6
from critics.critic7 import ContrastiveCritic7
import os
from tqdm import tqdm

import random
import re
import math

def extract_search_procedure(text):
    lines = text.split('\n')
    start_idx = None
    end_idx = None

    non_empty_lines = [line for line in lines if line.strip()]
    lines = non_empty_lines
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('|-') or line.startswith(' |-'):
            start_idx = i
            break
    
    if start_idx is not None:
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            if not (line.startswith('|-') or line.startswith(' |-')):
                end_idx = i
                break
    
        if end_idx is not None:
            search_procedure = '\n'.join(lines[start_idx:end_idx])
            indentation_levels = []
            for i in range(start_idx, end_idx):
                
                spaces = len(lines[i]) - len(lines[i].lstrip())
                level = spaces  # // 2
                indentation_levels.append(level)
        else:

            search_procedure = '\n'.join(lines[start_idx:-1])
            indentation_levels = []
            for i in range(start_idx, len(lines)):
                spaces = len(lines[i]) - len(lines[i].lstrip())
                level = spaces  # // 2
                indentation_levels.append(level)
        
        return search_procedure
    
    return None, None

def sag_extract(trajectory):

    lines = [line for line in trajectory.strip().split('\n') if line.strip()]
    #NEED AT LEAST 2 LINES

    state_end_idx = random.randint(0, len(lines) - 2)
    state = '\n'.join(lines[:state_end_idx + 1])
    action = lines[state_end_idx + 1]
    remaining_lines = len(lines) - (state_end_idx + 2)

    if remaining_lines <= 0:
        goal_idx = len(lines) - 1
    else:
        #sampling from geometric distribution (using Kevin's log transform)
        p = 0.2 
        steps_forward = min(
            int(math.log(random.random()) / math.log(1-p)) + 1 - 1,  # the +1 -1 are used for tweaking
            remaining_lines
        )
        goal_idx = state_end_idx + 2 + steps_forward
        
    goal_idx = min(goal_idx, len(lines) - 1)
    
    # We only care about the "### Current State" part from the goal line
    goal_line = lines[goal_idx]
    goal_match = re.search(r'### Current State: (\[.*?\])', goal_line)
    
    if goal_match:
        goal = goal_match.group(1)
    else:
        goal = goal_line
    
    return state, action, goal

def get_sag(dataset_path):
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    sag_pairs = []
    for i, data in enumerate(dataset):
        print(i)
        prompt, completion, nums, target = data['prompt'], data['completion'], data['nums'], data['target']
        trajectory = extract_search_procedure(completion)
        for i in range(2):  # how many SAG pairs we get per trajectory
            state, action, goal = sag_extract(trajectory)
            sag_pairs.append({
                'prompt': prompt,
                'completion': completion,
                'nums': nums,
                'target': target,
                'state': state,
                'action': action,
                'goal': goal
            })
    
    return sag_pairs


# def main2():
#     dataset_path = '/scratch/gpfs/dy2617/claude_responses/all_responses.json'
#     lolz = get_sag(dataset_path)
#     import pickle
#     with open('sag_pairs.pkl', 'wb') as f:
#         pickle.dump(lolz, f)

#     print(lolz)



def train_critic(model_path, dataset_path, output_dir, batch_size=16, num_epochs=10, learning_rate=1e-5):
    critic = ContrastiveCritic7(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic.to(device)
    
    optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    sag_pairs = get_sag(dataset_path)

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        import random
        random.shuffle(sag_pairs)

        #batching
        for i in tqdm(range(0, len(sag_pairs), batch_size), desc=f"Epoch {epoch + 1}/{num_epochs}"):
            batch = sag_pairs[i:i + batch_size]
        
            states = [item['state'] for item in batch]
            actions = [item['action'] for item in batch]
            goals = [item['goal'] for item in batch]
            
            loss = critic.train_step(states, actions, goals, optimizer)

            print(loss)
            total_loss += loss
            num_batches += 1
            
            if (i + batch_size) % (batch_size * 10) == 0:
                print(f"Batch {i//batch_size + 1}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1} done. Loss: {avg_loss:.4f}")
    
        os.makedirs(output_dir, exist_ok=True)
        torch.save(critic.action_mlp.state_dict(), os.path.join(output_dir, f"action_mlp_epoch_{epoch+1}.pt"))
        torch.save(critic.goal_mlp.state_dict(), os.path.join(output_dir, f"goal_mlp_epoch_{epoch+1}.pt"))
        
        metrics = {
            "epoch": epoch + 1,
            "average_loss": avg_loss,
            "num_samples": len(sag_pairs)
        }
        with open(os.path.join(output_dir, f"metrics_epoch_{epoch+1}.json"), 'w') as f:
            json.dump(metrics, f)

def main():
   
    # model_path = 'microsoft/rho-math-1b-interpreter-v0.1'
    # model_path = '/scratch/gpfs/dy2617/finetune'
    model_path = 'Qwen/Qwen2.5-7B-Instruct'
    dataset_path = '/scratch/gpfs/dy2617/claude_responses/all_responses.json'
    output_dir = 'critic_checkpoints'
    batch_size = 16
    num_epochs = 10
    learning_rate = 1e-1
    
    train_critic(
        model_path=model_path,
        dataset_path=dataset_path,
        output_dir=output_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )

if __name__ == "__main__":
    main()