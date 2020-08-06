import torch.nn as nn
import torch.nn.functional as F
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 100) #输入形状
        self.fc2 = nn.Linear(100, 10) #输出形状
    def forward(self, image):
        image_viwed = image.view(-1, 1*28*28)#[batch_size, 1*28*28] -1占位符
        fc1_out = self.fc1(image_viwed) #[batch_size, 100]
        fc1_out_relu = F.relu(fc1_out) #[batch_size, 100]
        out = self.fc2(fc1_out_relu)#[batch_size, 10]
        return F.log_softmax(out, dim=-1)