"""
进行模型的训练
"""
from dataset import get_dataloader
from models import MnistModel
from torch import optim
import torch.nn.functional as F
import conf
from tqdm import tqdm
import numpy as np
#1.实例化模型, 优化器 ,损失函数
model = MnistModel().to(conf.device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#2,进行循环，进行训练
def train(epoch):
    train_dataloader = get_dataloader(train=True)
    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    total_loss = []
    for idx, (input, target) in bar:
        input = input.to(conf.device)
        target = target.to(conf.device)
        #梯度置为0
        optimizer.zero_grad()
        #计算得到预测值
        output = model(input)
        #得到损失
        loss = F.nll_loss(output, target)
        #反向传播计算损失    
        loss.backward()
        total_loss.append(loss.item())
        #参数的更新
        optimizer.step()
        #打印数据
        if idx%10==0:
            bar.set_description("epoch:{} idx:{}, loss:{:.6f}".format(epoch, idx, np.mean(total_loss)))
if __name__ == '__main__':
    for i in range(5):
        train(i)