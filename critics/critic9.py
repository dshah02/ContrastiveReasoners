import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import re

class ContrastiveCritic9(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2-0.5B", temperature=1.0, projection_dim=256, max_goal_id=1000, num_embedding_dim=64, max_array_length=4):
        super(ContrastiveCritic9, self).__init__()
        
        self.max_array_length = max_array_length
        self.num_embedding_dim = num_embedding_dim
        
        self.number_embedding= nn.Embedding(max_goal_id + 1, num_embedding_dim)

        # nn.init.xavier_uniform_(self.number_embedding.weight)
        
        self.blank_embedding = nn.Parameter(torch.randn(num_embedding_dim))
        # nn.init.xavier_uniform_(self.blank_embedding.view(1, -1))
        
        input_dim = num_embedding_dim * max_array_length
        intermediate_dim = (input_dim + projection_dim) // 2
        
        self.action_mlp = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.LeakyReLU(),
            nn.Linear(intermediate_dim, projection_dim)
        )
             
        self.goal_embedding = nn.Embedding(max_goal_id + 1, num_embedding_dim)
        
        self.goal_mlp = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.LeakyReLU(),
            nn.Linear(intermediate_dim, projection_dim)
        )
        
        self.blank_goal_embedding = nn.Parameter(torch.randn(num_embedding_dim))
        
        self.temperature = temperature
        self.max_goal_id = max_goal_id

    #regex to extract after ###Current State and conver tot an array
    def extract_state_array(self, state_text):
        match = re.search(r'### Current State: \[([\d, ]+)\]', state_text)
        if match:
            numbers_str = match.group(1)
            numbers = []
            for num in numbers_str.split(','):
                num = num.strip()
                try:
                    numbers.append(int(num))
                except ValueError:
                    if num and not num.isdigit():
                        try:
                            float_val = float(num)
                            numbers.append(500 + int(float_val))
                        except ValueError:
                            numbers.append(0)
                    elif num.isdigit():
                        numbers.append(int(num))
            return numbers
        return []

    def encode_action(self, s_texts, a_texts):
        batch_size = len(s_texts)
        device = next(self.parameters()).device
        
        all_embeddings = torch.zeros(batch_size, self.max_array_length * self.num_embedding_dim, device=device)
        
        for i, a_text in enumerate(a_texts):
            numbers = self.extract_state_array(a_text)
            numbers.sort(reverse=True) #sort so order is consistnet
            
            embeddings = []
            for j in range(self.max_array_length):
                if j < len(numbers):
                    num = min(numbers[j], self.max_goal_id)
                    num_tensor = torch.tensor([num], device=device)
                    embedding = self.number_embedding(num_tensor).squeeze(0)
                else:
                    embedding = self.blank_embedding
                
                embeddings.append(embedding)

            state_embedding = torch.cat(embeddings)
            all_embeddings[i] = state_embedding
        
        projected_embeddings = self.action_mlp(all_embeddings)
        return F.normalize(projected_embeddings, p=2, dim=1) 

    def extract_goal_ids(self, g_texts):
        numbers = []
        match = re.search(r'\[([\d, ]+)\]', g_texts)
        if match:
            numbers_str = match.group(1)
            for num in numbers_str.split(','):
                num = num.strip()
                try:
                    numbers.append(int(num))
                except ValueError: #handles decimal case, we add 500 to distinguish
                    if num and not num.isdigit():
                        try:
                            float_val = float(num)
                            numbers.append(500 + int(float_val))
                        except ValueError:
                            numbers.append(0)
                    elif num.isdigit():
                        numbers.append(int(num))
        return numbers

    def encode_goal(self, g_texts):
        batch_size = len(g_texts)
        device = next(self.parameters()).device
        all_embeddings = torch.zeros(batch_size, self.max_array_length * self.num_embedding_dim, device=device)
        
        for i, g_text in enumerate(g_texts):
            numbers = self.extract_goal_ids(g_text)
            numbers.sort(reverse=True)
            embeddings = []

            #might be a bug here and in above
            for j in range(self.max_array_length):
                if j < len(numbers):
                    num = min(numbers[j], self.max_goal_id)
                    num_tensor = torch.tensor([num], device=device)
                    embedding = self.goal_embedding(num_tensor).squeeze(0)
                else:
                    embedding = self.blank_goal_embedding
                
                embeddings.append(embedding)
            
            goal_embedding = torch.cat(embeddings)
            all_embeddings[i] = goal_embedding
        
        projected_embeddings = self.goal_mlp(all_embeddings)
        return F.normalize(projected_embeddings, p=2, dim=1)  

    def forward(self, s_texts = None, a_texts = None, g_texts=None):

        emb_action = self.encode_action(s_texts, a_texts)  
        emb_goal = self.encode_goal(g_texts)               
        Q = torch.matmul(emb_action, emb_goal.t())
        return Q

    def train_step(self, s_texts = None, a_texts = None, g_texts = None, optimizer = None):
        self.train()
        optimizer.zero_grad()
        Q = self.forward(s_texts, a_texts, g_texts)
        batch_size = Q.size(0)
        targets = torch.arange(batch_size, device=Q.device)
        loss = F.cross_entropy(Q / self.temperature, targets)
        loss.backward()
        optimizer.step()
        return loss.item()


if __name__ == "__main__":

    critic = ContrastiveCritic9()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic.to(device)

    # Dummy batch data
    s_texts = ["### Current State: [21, 16, 17, 26]", "### Current State: [37, 17, 26]", "### Current State: [20, 26]"]
    a_texts = ["action text 1", "action text 2", "action text 3"]
    g_texts = ["[28]", "[45]", "[7]"]  # Goals in the format [number]

    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-5)

    with torch.no_grad():
        Q_values = critic.forward(s_texts, a_texts, g_texts)
        print("Pairwise Q:\n", Q_values)

    loss = critic.train_step(s_texts, a_texts, g_texts, optimizer)
    print("Training loss:", loss)
