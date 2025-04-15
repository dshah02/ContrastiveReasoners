import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class ContrastiveCritic(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2-0.5B", temperature=1.0):
        """
        Initializes the contrastive critic.
        Args:
            model_name: Name/path of the pretrained transformer model.
            temperature: Scaling factor for the logits in the contrastive loss.
        """
        super(ContrastiveCritic, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.pad_token or self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.action_encoder = AutoModel.from_pretrained(model_name)
        self.goal_encoder = AutoModel.from_pretrained(model_name)
        self.temperature = temperature

    def encode_action(self, s_texts, a_texts):
        """
        Encodes the (s, a) pair texts.
        The two strings are concatenated (with a space separator) and then tokenized.
        The hidden state at the last non-padding token is returned.
        Args:
            s_texts: List of state strings.
            a_texts: List of action strings.
        Returns:
            Tensor of shape (batch_size, hidden_dim) representing the embeddings.
        """
        combined_texts = [s + " " + a for s, a in zip(s_texts, a_texts)]
        encoded = self.tokenizer(combined_texts, padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(next(self.parameters()).device) for k, v in encoded.items()}
        outputs = self.action_encoder(**encoded)
        last_hidden_states = outputs.last_hidden_state  # (B, L, D)

        attention_mask = encoded["attention_mask"]
        lengths = attention_mask.sum(dim=1) - 1  # (B,)
        batch_size = last_hidden_states.size(0)

        final_embeddings = last_hidden_states[torch.arange(batch_size), lengths, :]
        return final_embeddings

    def encode_goal(self, g_texts):
        """
        Encodes the goal texts.
        Args:
            g_texts: List of goal strings.
        Returns:
            Tensor of shape (batch_size, hidden_dim) representing the goal embeddings.
        """
        encoded = self.tokenizer(g_texts, padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(next(self.parameters()).device) for k, v in encoded.items()}
        outputs = self.goal_encoder(**encoded)
        last_hidden_states = outputs.last_hidden_state  
        attention_mask = encoded["attention_mask"]
        lengths = attention_mask.sum(dim=1) - 1  
        batch_size = last_hidden_states.size(0)
        final_embeddings = last_hidden_states[torch.arange(batch_size), lengths, :]
        return final_embeddings

    def forward(self, s_texts, a_texts, g_texts):
        """
        Forward pass to compute Q(s, a, g) values.
        Args:
            s_texts: List of state strings.
            a_texts: List of action strings.
            g_texts: List of goal strings.
        Returns:
            Q: Tensor of shape (batch_size, batch_size), where entry Q[i, j] is 
               the dot-product between φ(s_texts[i], a_texts[i]) and Ψ(g_texts[j]).
        """
        emb_action = self.encode_action(s_texts, a_texts)  
        emb_goal = self.encode_goal(g_texts)               
        Q = torch.matmul(emb_action, emb_goal.t())
        return Q

    def train_step(self, s_texts, a_texts, g_texts, optimizer):
        """
        Performs a training step on a batch of examples using the InfoNCE contrastive loss.
        For a batch of size N, the target labels are the diagonal elements (i.e. index i is 
        the correct match for row i).
        Args:
            s_texts: List of state strings.
            a_texts: List of action strings.
            g_texts: List of goal strings.
            optimizer: Optimizer for the model parameters.
        Returns:
            loss_value: The scalar loss value for the training step.
        """
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

    critic = ContrastiveCritic()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic.to(device)

    # Dummy batch data
    s_texts = ["state text 1", "state text 2", "state text 3"]
    a_texts = ["action text 1", "action text 2", "action text 3"]
    g_texts = ["goal text 1", "goal text 2", "goal text 3"]

    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-5)

    with torch.no_grad():
        Q_values = critic.forward(s_texts, a_texts, g_texts)
        print("Pairwise Q values:\n", Q_values)

    #FOR DEVAN: EXAMPLE TRAINING STEP
    loss = critic.train_step(s_texts, a_texts, g_texts, optimizer)
    print("Training loss:", loss)
