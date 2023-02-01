import torch

a=torch.tensor([[1,4,2,3,17],
                [6,15,3,14,2]])

print(a)
print(a==1)
print(torch.numel(a[a==1]))
