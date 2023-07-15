import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()  # 将深度图转换为numpy数组，并将其存储在变量x中
    x = np.nan_to_num(x)  # 将数组中的NaN值替换为0
    mi = np.min(x)  # 获取深度图中的最小值
    ma = np.max(x)  # 获取深度图中的最大值
    x = (x-mi)/(ma-mi+1e-8)  # 将深度图归一化到0~1的范围内
    x = (255*x).astype(np.uint8)  # 将深度图转换为0~255之间的整数
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))  # 使用cv2.applyColorMap函数将灰度图转换为彩色图
    x_ = T.ToTensor()(x_)  # 将PIL图像转换为Tensor格式，形状为(3, H, W)
    return x_  # 返回处理后的图像
