import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
x = torch.randn(3,3)
print("random tensor:", x)
