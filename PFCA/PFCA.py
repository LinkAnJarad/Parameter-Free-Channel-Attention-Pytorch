import torch
from torch import nn

class ParameterFreeChannelAttention(nn.Module):
    '''
    Shi, Yuxuan & Xu, Jun & Yang, Lingxiao & An, Wangpeng & Zhen, Xiantong. (2022). PARAMETER-FREE CHANNEL ATTENTION FOR IMAGE CLASSIFICATION AND SUPER-RESOLUTION. 10.13140/RG.2.2.20039.78241.
    '''
    def __init__(self, kernel_size, _lambda=1e-4):
        super().__init__()
        
        self.pooling = nn.AvgPool2d(kernel_size)
        self._lambda = _lambda
        
    def forward(self, x):
        U = self.pooling(x).squeeze()
        mean, variance = U.mean(-1, keepdim=True), U.var(-1, keepdim=True)
        V = (U - mean)**2 + (2*(variance + self._lambda))
        V = V/(4* (variance + self._lambda))
        N, C = V.shape
        att = V.sigmoid().reshape(N, C, 1, 1)
        return x * att
