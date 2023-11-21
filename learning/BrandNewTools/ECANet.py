from math import log
from torch import nn

def EfficientChannelAttention(x, gamma, b=1):
    # x: input features with shape [N, C, H, W]
    # gamma, b: parameters of mapping function

    N, C, H, W = x.size()

    t = int(abs((log(C, 2) + b) / gamma))
    k = t if t % 2 else t + 1

    avg_pool = nn.AdaptiveAvgPool2d(1)
    conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2),
                     bias=False)
    sigmoid = nn.Sigmoid()

    y = avg_pool(x)
    y = conv(y.squeeze(-1).transpose(-1, -2))
    y = y.transpose(-1, -2).unsqueeze(-1)
    y = sigmoid(y)

    return x * y.expand_as(x)