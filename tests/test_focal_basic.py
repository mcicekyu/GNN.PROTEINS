import torch
from src.utils.features import focal_loss

def test_focal_loss_shapes():
    logits = torch.randn(4, 3)
    targets = torch.tensor([0,1,2,0], dtype=torch.long)
    loss = focal_loss(logits, targets, gamma=2.0, weight=None)
    assert torch.is_tensor(loss) and loss.dim() == 0