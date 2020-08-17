"""
构建模型
"""
import torch.nn as nn
import config
import torch.nn.functional as F
class ImbModel(nn.Module):
    def __init__(self):
        super(ImbModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.ws), embedding_dim=200, padding_idx=config.ws.PAD

                                      ) #1.词语的数量 2.用长度为多少的向量表示词语
        self.fc = nn.Linear(config.max_len*200, 2)
    def forward(self, input):
        """

        :param inpout:
        :return:
        """
        input_embeded  =self.embedding(input) #input embeded ：[batch_size, max_len, 200]
        #变形
        input_embeded_viewed = input_embeded.view(input_embeded.size(0), -1) #x.size()表示向量第一个维度的值 也就是batch_size,-1表示max_len和200的乘机
        #全连接
        out = self.fc(input_embeded_viewed)
        return F.log_softmax(out, dim=-1)
