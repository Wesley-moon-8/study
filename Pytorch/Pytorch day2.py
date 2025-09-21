import torch

# 3个学生的2门成绩（3行2列）
scores = torch.tensor([[72, 75],
                      [85, 89],
                      [77, 79]], dtype=torch.float32)  # 用float32避免求平均时精度问题

# 1. 按行方向（dim=0）：求每门课的总分和平均分（3个学生的同一门课相加）
subject_total = scores.sum(dim=0)  # 每列求和
subject_avg = scores.mean(dim=0)   # 每列求平均
print("每门课总分：", subject_total)  # tensor([234., 243.])（72+85+77=234；75+89+79=243）
print("每门课平均分：", subject_avg)  # tensor([78.0000, 81.0000])

# 2. 按列方向（dim=1）：求每个学生的总分和平均分（同一个学生的两门课相加）
student_total = scores.sum(dim=1)  # 每行求和
student_avg = scores.mean(dim=1)   # 每行求平均
print("\n每个学生总分：", student_total)  # tensor([147., 174., 156.])
print("每个学生平均分：", student_avg)  # tensor([73.5000, 87.0000, 78.0000])
