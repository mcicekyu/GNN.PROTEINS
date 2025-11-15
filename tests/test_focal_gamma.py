import torch
import torch.nn.functional as F
from src.utils.features import focal_loss

def test_focal_loss_gamma_zero_equals_cross_entropy():
    logits = torch.randn(5, 3, dtype=torch.float32)
    targets = torch.tensor([0,1,2,1,0], dtype=torch.long)
    fl = focal_loss(logits, targets, gamma=0.0, weight=None)
    ce = F.cross_entropy(logits, targets)
    assert torch.allclose(fl.detach(), ce.detach(), atol=1e-6)

def test_focal_loss_with_weights_and_various_gamma():
    logits = torch.tensor([[5.0, 0.1, 0.1],
                           [0.1, 5.0, 0.1],
                           [0.1, 0.1, 5.0]], dtype=torch.float32)
    targets = torch.tensor([0,1,2], dtype=torch.long)
    # class weights favor class 0 heavily
    weights = torch.tensor([0.1, 1.0, 1.0], dtype=torch.float32)
    l1 = focal_loss(logits, targets, gamma=0.0, weight=weights)
    l2 = focal_loss(logits, targets, gamma=2.0, weight=weights)
    # should return scalar tensors
    assert torch.is_tensor(l1) and l1.dim() == 0
    assert torch.is_tensor(l2) and l2.dim() == 0