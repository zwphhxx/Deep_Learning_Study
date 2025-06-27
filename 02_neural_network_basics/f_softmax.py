import torch

torch.manual_seed(42)
scores = torch.randn(10)
print(scores)
probabilities = torch.softmax(scores, dim=0)
print(probabilities)
