import copy
import random

import torch
from collections import defaultdict
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
# print(torch.cuda.is_available())
#
# Wb = defaultdict(lambda: defaultdict(list))
# for i in range(5):
#     for j in range(3):
#         Wb[i][j].append(i * j)
#         Wb[i][j].append(i + 2)
#         Wb[i][j].append(i + 3)
# ws = copy.deepcopy(Wb)
#
# for i in range(3):
#     x=random.sample(Wb[0][0], 1)
#     print(x)
#     Wb[0][0].remove(x[0])
#     print(len(Wb[0][0]))
#
# Wb[1][0]=[]
#
# print(Wb)
print(torch.arange(-1,127,1))
print(torch.arange(127,-1,-1))