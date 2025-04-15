import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaModel, LlamaTokenizer
from typing import List, Tuple, Dict, Optional


class ContrastiveCritic(nn.Module):
    """
    Contrastive Critic model Q(s, a, g) where:
    - s is text (state)
    - a is text (action)
    - g is text (goal)
    
    The model determines whether (s, a) and g are on the same trajectory.
    """
    
    def __init__(
        self,
        model_name: str = "huggyllama/llama-7b",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 0.1
    ):
        """
        Initialize the Contrastive Critic model.
        
        Args:
            model_name: Name or path of the LLaMA model
            device: Device to load the model on
            temperature: Temperature parameter for InfoNCE loss
        """
        super().__init__()
        
        self.device = device
        self.temperature = temperature
        
        # Initialize tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize encoders
        self.phi_encoder = LlamaModel.from_pretrained(model_name).to(device)  # φ(s, a)
        self.psi_encoder = LlamaModel.from_pretrained(model_name).to(device)  # Ψ(g)
        
        # Freeze pre-trained weights initially (optional, depends on training strategy)
        # self._freeze_params(self.phi_encoder)
        # self._freeze_params(self.psi_encoder)
        
        # Add projection layers (optional for dimension reduction)
        # self.phi_projection = nn.Linear(self.phi_encoder.config.hidden_size, 768).to(device)
        # self.psi_projection = nn.Linear(self.psi_encoder.config.hidden_size, 768).to(device)
    
    def _freeze_params(self, model: nn.Module) -> None:
        """Freeze parameters of the model."""
        for param in model.parameters():
            param.requires_grad = False
    
    def _unfreeze_params(self, model: nn.Module) -> None:
        """Unfreeze parameters of the model."""
        for param in model.parameters():
            param.requires_grad = True
    
    def encode_phi(self, states: List[str], actions: List[str]) -> torch.Tensor:
        """
        Encode state-action pairs using φ encoder.
        
        Args:
            states: List of state text strings
            actions: List of action text strings
            
        Returns:
            Tensor of embeddings for state-action pairs
        """
        # Concatenate state and action for each pair
        sa_texts = [f"{s} {a}" for s, a in zip(states, actions)]
        
        # Tokenize
        encodings = self.tokenizer(
            sa_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Forward pass through phi encoder
        with torch.no_grad() if not self.phi_encoder.training else torch.enable_grad():
            outputs = self.phi_encoder(**encodings)
            
        # Get last token representation of last layer
        # Shape: (batch_size, hidden_size)
        last_token_indices = encodings.attention_mask.sum(dim=1) - 1
        batch_size = encodings.input_ids.shape[0]
        hidden_size = outputs.last_hidden_state.shape[-1]
        
        # Extract the hidden state for the last token of each sequence
        sa_embeddings = torch.zeros((batch_size, hidden_size), device=self.device)
        for i in range(batch_size):
            sa_embeddings[i] = outputs.last_hidden_state[i, last_token_indices[i]]
        
        # Apply projection if needed
        # sa_embeddings = self.phi_projection(sa_embeddings)
        
        # Normalize embeddings
        sa_embeddings = F.normalize(sa_embeddings, p=2, dim=1)
        
        return sa_embeddings
    
    def encode_psi(self, goals: List[str]) -> torch.Tensor:
        """
        Encode goals using Ψ encoder.
        
        Args:
            goals: List of goal text strings
            
        Returns:
            Tensor of embeddings for goals
        """
        # Tokenize
        encodings = self.tokenizer(
            goals,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Forward pass through psi encoder
        with torch.no_grad() if not self.psi_encoder.training else torch.enable_grad():
            outputs = self.psi_encoder(**encodings)
        
        # Get last token representation of last layer
        # Shape: (batch_size, hidden_size)
        last_token_indices = encodings.attention_mask.sum(dim=1) - 1
        batch_size = encodings.input_ids.shape[0]
        hidden_size = outputs.last_hidden_state.shape[-1]
        
        # Extract the hidden state for the last token of each sequence
        g_embeddings = torch.zeros((batch_size, hidden_size), device=self.device)
        for i in range(batch_size):
            g_embeddings[i] = outputs.last_hidden_state[i, last_token_indices[i]]
        
        # Apply projection if needed
        # g_embeddings = self.psi_projection(g_embeddings)
        
        # Normalize embeddings
        g_embeddings = F.normalize(g_embeddings, p=2, dim=1)
        
        return g_embeddings
    
    def forward(self, states: List[str], actions: List[str], goals: List[str]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the critic model.
        
        Args:
            states: List of state text strings
            actions: List of action text strings
            goals: List of goal text strings
            
        Returns:
            Dictionary containing:
                - 'q_values': Matrix of dot products (batch_size x batch_size)
                - 'sa_embeddings': Embeddings of state-action pairs
                - 'g_embeddings': Embeddings of goals
        """
        # Encode state-action pairs and goals
        sa_embeddings = self.encode_phi(states, actions)  # (batch_size, hidden_size)
        g_embeddings = self.encode_psi(goals)             # (batch_size, hidden_size)
        
        # Compute Q values as dot products: Q(s, a, g) = φ(s, a) · Ψ(g)
        # This creates an NxN matrix of all pairwise dot products
        q_values = torch.matmul(sa_embeddings, g_embeddings.transpose(0, 1))  # (batch_size, batch_size)
        
        return {
            'q_values': q_values,
            'sa_embeddings': sa_embeddings,
            'g_embeddings': g_embeddings
        }
    
    def compute_loss(self, q_values: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss.
        
        Args:
            q_values: Matrix of dot products (batch_size x batch_size)
            
        Returns:
            Contrastive loss value
        """
        # Scale by temperature
        logits = q_values / self.temperature
        
        # Create target labels (diagonal matrix)
        # The positive pairs are along the diagonal (i.e., same index)
        batch_size = q_values.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def training_step(
        self,
        states: List[str],
        actions: List[str],
        goals: List[str],
        optimizer: torch.optim.Optimizer,
        grad_clip: Optional[float] = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Perform a single training step.
        
        Args:
            states: List of state text strings
            actions: List of action text strings
            goals: List of goal text strings
            optimizer: Optimizer to use for the gradient step
            grad_clip: Value for gradient clipping (None for no clipping)
            
        Returns:
            Dictionary with loss and other metrics
        """
        # Set to training mode
        self.train()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = self.forward(states, actions, goals)
        q_values = outputs['q_values']
        
        # Compute loss
        loss = self.compute_loss(q_values)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
        
        # Update parameters
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            # Calculate accuracy (correct predictions on diagonal)
            batch_size = q_values.shape[0]
            predictions = torch.argmax(q_values, dim=1)
            labels = torch.arange(batch_size, device=self.device)
            accuracy = (predictions == labels).float().mean()
            
            # Calculate average positive and negative dot products
            positive_dots = torch.diagonal(q_values).mean()
            mask = torch.ones_like(q_values, dtype=torch.bool)
            mask.fill_diagonal_(False)
            negative_dots = q_values[mask].mean()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'positive_dots': positive_dots.item(),
            'negative_dots': negative_dots.item()
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        states: List[str],
        actions: List[str],
        goals: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate the model without gradient computation.
        
        Args:
            states: List of state text strings
            actions: List of action text strings
            goals: List of goal text strings
            
        Returns:
            Dictionary with loss and metrics
        """
        # Set to evaluation mode
        self.eval()
        
        # Forward pass
        outputs = self.forward(states, actions, goals)
        q_values = outputs['q_values']
        
        # Compute loss
        loss = self.compute_loss(q_values)
        
        # Calculate metrics
        batch_size = q_values.shape[0]
        predictions = torch.argmax(q_values, dim=1)
        labels = torch.arange(batch_size, device=self.device)
        accuracy = (predictions == labels).float().mean()
        
        # Calculate average positive and negative dot products
        positive_dots = torch.diagonal(q_values).mean()
        mask = torch.ones_like(q_values, dtype=torch.bool)
        mask.fill_diagonal_(False)
        negative_dots = q_values[mask].mean()
        
        return {
            'q_values': q_values,
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'positive_dots': positive_dots.item(),
            'negative_dots': negative_dots.item()
        }
    
    def get_q_value(self, state: str, action: str, goal: str) -> float:
        """
        Get a single Q-value for a state-action-goal triplet.
        
        Args:
            state: State text string
            action: Action text string
            goal: Goal text string
            
        Returns:
            Q-value as a float
        """
        self.eval()
        with torch.no_grad():
            sa_embedding = self.encode_phi([state], [action])
            g_embedding = self.encode_psi([goal])
            q_value = torch.matmul(sa_embedding, g_embedding.transpose(0, 1))
        
        return q_value.item()


# Example usage
def example_usage():
    # Initialize model
    critic = ContrastiveCritic(model_name="huggyllama/llama-7b")
    
    # Training data
    states = [
        "The user is asking about the weather.",
        "The user is inquiring about dinner options.",
        "The user wants to know the time."
    ]
    
    actions = [
        "I'll check the weather forecast for you.",
        "I can suggest some dinner recipes.",
        "The current time is 3:00 PM."
    ]
    
    goals = [
        "Provide weather information accurately.",
        "Help with meal planning suggestions.",
        "Give current time information."
    ]
    
    # Optimizer
    optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-5)
    
    # Training step
    metrics = critic.training_step(states, actions, goals, optimizer)
    print(f"Training metrics: {metrics}")
    
    # Evaluation
    eval_metrics = critic.evaluate(states, actions, goals)
    print(f"Evaluation metrics: {eval_metrics}")
    
    # Get single Q-value
    q = critic.get_q_value(states[0], actions[0], goals[0])
    print(f"Q-value for first example: {q}")


if __name__ == "__main__":
    example_usage()