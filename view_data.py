# import torch
import torch

# 加载 train.pt 数据
train_data = torch.load('aug_data/train.pt')

# 查看 train_data 的类型和内容
print(type(train_data))
print(train_data.keys() if isinstance(train_data, dict) else None)
print(train_data[0] if isinstance(train_data, list) else None)
print(len(train_data))

# import torch

# # 加载 train.pt 数据
# train_data = torch.load('train.pt')

# 初始化小类计数
subclass_counts = {}

# 统计每个小类的数量
for data in train_data:
    _, label = data
    label = int(label)
    if label in subclass_counts:
        subclass_counts[label] += 1
    else:
        subclass_counts[label] = 1

# 打印各个小类的数量
print("各个超类的数量:")
for subclass, count in subclass_counts.items():
    print(f"超类 {subclass}: {count} 张图片")
