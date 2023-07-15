import torch
from torch import nn

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()  # 继承父类的初始化方法
        self.loss = nn.MSELoss(reduction='mean')  # 使用均值作为减少维度的方式进行实例化

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)  # 计算rgb_coarse与targets之间的均方误差损失
        if 'rgb_fine' in inputs:  # 如果输入中包含rgb_fine
            loss += self.loss(inputs['rgb_fine'], targets)  # 将rgb_fine与targets之间的均方误差损失添加到总损失中

        return loss

loss_dict = {'mse': MSELoss}  # 定义一个字典，键为'mse'，值为MSELoss类的对象
