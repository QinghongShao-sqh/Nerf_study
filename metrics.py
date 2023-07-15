import torch
from kornia.losses import ssim as dssim

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2  # 计算预测图像和真实图像之间的均方误差
    if valid_mask is not None:  # 如果有效掩码不为空
        value = value[valid_mask]  # 仅保留有效掩码指示的像素值
    if reduction == 'mean':  # 如果使用平均值作为减少维度的方式
        return torch.mean(value)  # 返回均方误差的平均值
    return value  # 返回均方误差

def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))  # 返回图像的峰值信噪比

def ssim(image_pred, image_gt, reduction='mean'):
    """
    image_pred and image_gt: (1, 3, H, W)  # 输入图像的形状为(1, 3, H, W)
    """
    dssim_ = dssim(image_pred, image_gt, 3, reduction)  # 使用kornia库中的ssim函数计算结构相似性指数
    return 1-2*dssim_  # 返回结构相似性指数的相反数（范围为[-1, 1]）
