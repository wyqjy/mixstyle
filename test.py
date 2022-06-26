import torch

# perm = torch.arange(64 - 1, -1, -1)  # inverse index
# perm_b, perm_a = perm.chunk(2)
# print(perm)
# print('-'*50)
# print(perm_b)
# print('-'*50)
# print(perm_a)
# perm = torch.cat([perm_b, perm_a], 0)
# print('-'*50)
# print(perm)
print(torch.cuda.is_available())

k=list()
