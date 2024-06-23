import torch
import torch.nn.functional as F

# Example logits
logits = torch.tensor([[1.2, 0.5, -0.3],
                       [-0.1, 2.1, 0.8],
                       [0.3, -0.7, 1.5],
                       [2.0, -0.4, 0.1]])

# Compute log softmax
log_probs = F.log_softmax(logits, dim=1)

print(log_probs)

# Target labels
targets = torch.tensor([0, 1, 2, 0])

# Compute NLL loss
loss = F.nll_loss(log_probs, targets)
