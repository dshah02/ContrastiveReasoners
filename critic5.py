import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional

class ContrastiveCritic(nn.Module):
    """
    Contrastive Critic model Q(s, a, g) that determines whether (s, a) and g are on the same trajectory.
    
    The model uses two LLaMA 7B transformers as encoders:
    - φ(s, a): Encodes the state-action pair
    - Ψ(g): Encodes the goal
    
    Q(s, a, g) is computed as the dot product of these encodings.
    """
    def __init__(self, model_name = "meta-llama/Llama-3.1-8B", device="cuda"):
        super(ContrastiveCritic, self).__init__()
        
        self.device = device
        
        # Initialize the state-action encoder φ(s, a)
        self.sa_encoder = LlamaModel.from_pretrained(model_name)
        
        # Initialize the goal encoder Ψ(g)
        self.g_encoder = LlamaModel.from_pretrained(model_name)
        
        # Project hidden states to a common dimension for comparison
        hidden_size = self.sa_encoder.config.hidden_size
        self.proj_dim = 512
        
        self.sa_projector = nn.Linear(hidden_size, self.proj_dim)
        self.g_projector = nn.Linear(hidden_size, self.proj_dim)
        
        # Move to device
        self.to(device)
    
    def encode_sa(self, states: List[str], actions: List[str]) -> torch.Tensor:
        """
        Encode state-action pairs using φ(s, a)
        
        Args:
            states: List of state text strings
            actions: List of action text strings
            
        Returns:
            Encoded state-action embeddings
        """
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        
        # Concatenate state and action with a separator
        sa_texts = [f"State: {s} Action: {a}" for s, a in zip(states, actions)]
        
        # Tokenize inputs
        inputs = tokenizer(sa_texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get transformer outputs
        outputs = self.sa_encoder(**inputs)
        
        # Use the last token embedding as the representation
        # Get attention mask to identify non-padding tokens
        attention_mask = inputs['attention_mask']
        # Get indices of last non-padding tokens
        last_token_indices = attention_mask.sum(dim=1) - 1
        # Get embeddings of last tokens
        embeddings = outputs.last_hidden_state[torch.arange(len(last_token_indices), device=self.device), last_token_indices]
        
        # Project to common space
        return self.sa_projector(embeddings)
    
    def encode_g(self, goals: List[str]) -> torch.Tensor:
        """
        Encode goals using Ψ(g)
        
        Args:
            goals: List of goal text strings
            
        Returns:
            Encoded goal embeddings
        """
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
        
        # Tokenize inputs
        inputs = tokenizer(goals, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get transformer outputs
        outputs = self.g_encoder(**inputs)
        
        # Use the last token embedding as the representation
        # Get attention mask to identify non-padding tokens
        attention_mask = inputs['attention_mask']
        # Get indices of last non-padding tokens
        last_token_indices = attention_mask.sum(dim=1) - 1
        # Get embeddings of last tokens
        embeddings = outputs.last_hidden_state[torch.arange(len(last_token_indices), device=self.device), last_token_indices]
        
        # Project to common space
        return self.g_projector(embeddings)
    
    def forward(self, states: List[str], actions: List[str], goals: List[str]) -> torch.Tensor:
        """
        Compute Q(s, a, g) for all combinations in the batch.
        
        Args:
            states: List of state text strings
            actions: List of action text strings
            goals: List of goal text strings
            
        Returns:
            NxN matrix of pair-wise dot products
        """
        # Encode state-action pairs φ(s, a)
        sa_embeddings = self.encode_sa(states, actions)  # [N, proj_dim]
        
        # Encode goals Ψ(g)
        g_embeddings = self.encode_g(goals)  # [N, proj_dim]
        
        # Normalize embeddings to ensure dot products are bounded [-1, 1]
        sa_embeddings = F.normalize(sa_embeddings, p=2, dim=1)
        g_embeddings = F.normalize(g_embeddings, p=2, dim=1)
        
        # Compute NxN matrix of pairwise dot products
        # sim[i, j] = φ(s_i, a_i) · Ψ(g_j)
        similarity_matrix = torch.matmul(sa_embeddings, g_embeddings.t())
        
        return similarity_matrix


class TextTrajectoryDataset(Dataset):
    """
    Dataset for text-based trajectories with states, actions, and goals.
    """
    def __init__(self, states: List[str], actions: List[str], goals: List[str]):
        assert len(states) == len(actions) == len(goals)
        self.states = states
        self.actions = actions
        self.goals = goals
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            "state": self.states[idx],
            "action": self.actions[idx],
            "goal": self.goals[idx]
        }


def contrastive_infonce_loss(similarity_matrix: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Compute InfoNCE contrastive loss based on similarity matrix.
    
    Args:
        similarity_matrix: NxN matrix of pairwise dot products
        temperature: Temperature parameter to scale logits
        
    Returns:
        InfoNCE loss value
    """
    # Positive samples are on the diagonal
    batch_size = similarity_matrix.size(0)
    labels = torch.arange(batch_size).to(similarity_matrix.device)
    
    # Apply temperature scaling
    logits = similarity_matrix / temperature
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels)
    
    return loss


def train_critic(model: ContrastiveCritic, train_loader: DataLoader, 
                optimizer: torch.optim.Optimizer, num_epochs: int = 10, 
                device: str = "cuda") -> List[float]:
    """
    Train the contrastive critic model.
    
    Args:
        model: ContrastiveCritic model
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        num_epochs: Number of training epochs
        device: Device to use for training
        
    Returns:
        List of training losses per epoch
    """
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch in train_loader:
            states = batch["state"]
            actions = batch["action"]
            goals = batch["goal"]
            
            # Forward pass
            similarity_matrix = model(states, actions, goals)
            
            # Compute loss
            loss = contrastive_infonce_loss(similarity_matrix)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")
    
    return losses


def evaluate_critic(model: ContrastiveCritic, test_loader: DataLoader, device: str = "cuda") -> Dict:
    """
    Evaluate the contrastive critic model.
    
    Args:
        model: ContrastiveCritic model
        test_loader: DataLoader for test data
        device: Device to use for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            states = batch["state"]
            actions = batch["action"]
            goals = batch["goal"]
            
            # Forward pass
            similarity_matrix = model(states, actions, goals)
            
            # Compute loss
            loss = contrastive_infonce_loss(similarity_matrix)
            total_loss += loss.item()
            
            # Compute accuracy (matching diagonal)
            predicted = torch.argmax(similarity_matrix, dim=1)
            labels = torch.arange(len(states)).to(device)
            correct += (predicted == labels).sum().item()
            total += len(states)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy
    }


if __name__ == "__main__":
    # Example usage
    # This would be replaced with actual training data
    example_states = [
        "The user asks for weather information.",
        "The user requests a restaurant recommendation.",
        "The user wants to know the capital of France."
    ]
    
    example_actions = [
        "I provide the current weather forecast for their location.",
        "I recommend several top-rated restaurants in their area.",
        "I inform them that Paris is the capital of France."
    ]
    
    example_goals = [
        "Provide accurate weather information to the user.",
        "Help the user find a good restaurant.",
        "Answer the user's geography question correctly."
    ]
    
    # Create dataset and dataloader
    dataset = TextTrajectoryDataset(example_states, example_actions, example_goals)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveCritic(device=device)
    
    # # Initialize optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    # # Train model
    # print("Training model...")
    # # In a real scenario, you would uncomment this:
    # # losses = train_critic(model, dataloader, optimizer, num_epochs=5)
    
    # print("Model implementation complete!")
    
    # Train model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    losses = train_critic(model, dataloader, optimizer, num_epochs=10)

    # Evaluate model
    metrics = evaluate_critic(model, dataloader)
    print(f"Evaluation metrics: {metrics}")