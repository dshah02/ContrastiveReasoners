import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class ContrastiveCritic(nn.Module):
    """
    A contrastive critic model Q(s, a, g) that uses two transformer encoders to compute
    the similarity between state-action pairs (s, a) and goals (g).
    """
    def __init__(self, model_name="meta-llama/Llama-3.1-8B"):
        """
        Initialize the critic with two Llama transformer encoders and a shared tokenizer.
        
        Args:
            model_name (str): Name of the pre-trained model to load (default: Llama-3.1-8B).
        """
        super().__init__()
        # Load the tokenizer for text preprocessing
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load two instances of the transformer model as encoders
        self.phi = AutoModel.from_pretrained(model_name)  # Encoder for (s, a)
        self.psi = AutoModel.from_pretrained(model_name)  # Encoder for g
        # Hidden dimension of the transformer (used for embeddings)
        self.hidden_dim = self.phi.config.hidden_size

    def get_embeddings(self, input_ids_phi, input_ids_psi):
        """
        Compute normalized embeddings from the transformer encoders.
        
        Args:
            input_ids_phi (torch.Tensor): Tokenized input IDs for (s, a) pairs.
            input_ids_psi (torch.Tensor): Tokenized input IDs for g.
        
        Returns:
            tuple: (phi_emb, psi_emb) - Normalized embeddings for (s, a) and g.
        """
        # Get hidden states from phi encoder (for s, a)
        phi_outputs = self.phi(input_ids_phi)
        phi_emb = phi_outputs.last_hidden_state[:, -1, :]  # Last token, last layer
        # Get hidden states from psi encoder (for g)
        psi_outputs = self.psi(input_ids_psi)
        psi_emb = psi_outputs.last_hidden_state[:, -1, :]  # Last token, last layer
        # Normalize embeddings to unit length (dot product = cosine similarity)
        phi_emb = F.normalize(phi_emb, dim=1)
        psi_emb = F.normalize(psi_emb, dim=1)
        return phi_emb, psi_emb

    def forward(self, s_list, a_list, g_list):
        """
        Compute Q-values for a batch of (s, a, g) triples.
        
        Args:
            s_list (list[str]): List of state texts.
            a_list (list[str]): List of action texts.
            g_list (list[str]): List of goal texts.
        
        Returns:
            torch.Tensor: Q-values (dot products) for each (s, a, g) triple.
        """
        # Concatenate s and a with a space separator
        sa_list = [s + " " + a for s, a in zip(s_list, a_list)]
        # Tokenize inputs for phi (s, a) and psi (g)
        input_ids_phi = self.tokenizer(sa_list, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.phi.device)
        input_ids_psi = self.tokenizer(g_list, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.psi.device)
        # Compute embeddings
        phi_emb, psi_emb = self.get_embeddings(input_ids_phi, input_ids_psi)
        # Q(s, a, g) = dot product of embeddings
        Q_values = (phi_emb * psi_emb).sum(dim=1)
        return Q_values

    def compute_loss(self, s_list, a_list, g_list):
        """
        Compute the InfoNCE contrastive loss for a batch of (s, a, g) triples.
        
        Args:
            s_list (list[str]): List of state texts.
            a_list (list[str]): List of action texts.
            g_list (list[str]): List of goal texts.
        
        Returns:
            torch.Tensor: Contrastive loss value.
        """
        # Concatenate s and a with a space separator
        sa_list = [s + " " + a for s, a in zip(s_list, a_list)]
        # Tokenize inputs for phi (s, a) and psi (g)
        input_ids_phi = self.tokenizer(sa_list, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.phi.device)
        input_ids_psi = self.tokenizer(g_list, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.psi.device)
        # Compute embeddings
        phi_emb, psi_emb = self.get_embeddings(input_ids_phi, input_ids_psi)
        # Compute NxN matrix of dot products
        M = torch.mm(phi_emb, psi_emb.t())
        # Labels: diagonal indices (0, 1, ..., N-1) for positive pairs
        labels = torch.arange(M.size(0)).to(M.device)
        # InfoNCE loss via cross-entropy
        loss = F.cross_entropy(M, labels)
        return loss


# Example usage
if __name__ == "__main__":
    # Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveCritic("meta-llama/Llama-3.1-8B")
    model.to(device)

    # Sample batch
    s_batch = ["move forward", "turn left"]
    a_batch = ["step ahead", "rotate"]
    g_batch = ["reach target", "face wall"]

    # Compute Q-values
    Q_values = model(s_batch, a_batch, g_batch)
    print("Q-values:", Q_values)

    # Training step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss = model.compute_loss(s_batch, a_batch, g_batch)
    print("Loss:", loss.item())
    loss.backward()
    optimizer.step()
    print("Gradient step completed")