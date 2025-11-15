# choose device (use MPS on macOS if available, otherwise CPU)
import torch
use_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
device = torch.device("mps" if use_mps else "cpu")
print("selected device:", device)

# example: define a simple model and sample tensors, then move them to the chosen device
# adjust input/output sizes to match your real model and data
model = torch.nn.Sequential(
	torch.nn.Linear(10, 5),
	torch.nn.ReLU(),
	torch.nn.Linear(5, 1)
)

# create example input and target tensors
x = torch.randn(2, 10)
y = torch.randn(2, 1)

# move model and tensors to device
model = model.to(device)
x = x.to(device)          # input tensor
y = y.to(device)          # label tensor (if needed)

# forward pass
output = model(x)