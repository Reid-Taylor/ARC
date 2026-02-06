import torch
from torch.nn import functional as F
from jaxtyping import Float

POSITIONAL_ENCODINGS: Float[torch.Tensor, "1 30 30"] = (torch.arange((30*30)) / (30*30)).reshape(1,30,30)
BATCH_SIZE:int = 1
AUGMENTATIONS: list[str] = ["color_map", "roll", "reflect", "rotate", "scale_grid", "isolate_color"]

def entropy_density_loss(embeddings: torch.Tensor, lambda_entropy: float = 0.01) -> torch.Tensor:
    """Encourage high entropy in embedding magnitudes"""
    probs = F.softmax(torch.abs(embeddings), dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    return lambda_entropy * torch.mean(-entropy)

def variance_density_loss(embeddings: torch.Tensor, lambda_var: float = 0.01) -> torch.Tensor:
    """Encourage high variance to prevent mode collapse"""
    variance = torch.var(embeddings, dim=-1)
    return lambda_var * torch.mean(-variance)

def anti_sparsity_loss(embeddings: torch.Tensor, threshold: float = 0.1, lambda_sparse: float = 0.01) -> torch.Tensor:
    """Penalize activations below threshold"""
    sparse_penalty = torch.mean(torch.relu(threshold - torch.abs(embeddings)))
    return lambda_sparse * sparse_penalty
