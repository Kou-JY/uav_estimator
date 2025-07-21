import torch

# 加载数据
data = torch.load("D:/code/uav_angle_estimator/data/processed/segment_1.pt")

# 打印类型和长度
# print(type(data))
# if isinstance(data, list):
#     print(f"列表长度: {len(data)}")
#     print(f"前几个元素: {data[:5]}")
# else:
#     print(data)



# 查看字典内容
print(data.keys())  # 打印字典的所有键

# 查看各个张量的内容（部分）
print(f"Inputs shape: {data['Inputs'].shape}")
print(f"AOA shape: {data['AOA'].shape}")
print(f"Slip shape: {data['Slip'].shape}")
