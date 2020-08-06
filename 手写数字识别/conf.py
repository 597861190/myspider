"""
项目配置
"""
import torch
train_batch_size = 128  #训练集
test_batch_size = 1000  #测试集
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")