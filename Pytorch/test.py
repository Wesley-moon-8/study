import torch
import math
scores = torch.tensor([[85,87],
                      [88,89],
                      [96,89]])
subject_total = scores.sum(dim=0)  # 每列求和
subject_avg = scores.mean(dim=0)   # 每列求平均
print(subject_total)
print(subject_avg)
