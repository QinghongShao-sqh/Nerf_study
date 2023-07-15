import torch
from torch.utils.data import Dataset
import json
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *
class BlenderDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 800)):
        self.root_dir = root_dir  # 数据集的根目录
        self.split = split  # 数据集的划分（训练集、验证集、测试集）
        assert img_wh[0] == img_wh[1], 'image width must equal image height!'  # 检查图像的宽度和高度是否相等
        self.img_wh = img_wh  # 图像的宽高
        self.define_transforms()  # 定义数据转换方法

        self.read_meta()
        self.white_back = True

    def read_meta(self):
        with open(os.path.join(self.root_dir,
                               f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5*800/np.tan(0.5*self.meta['camera_angle_x'])  # 原始焦距，当W=800时
        self.focal *= self.img_wh[0]/800  # 修改焦距以匹配大小self.img_wh

        self.near = 2.0  # 近截面
        self.far = 6.0  # 远截面
        self.bounds = np.array([self.near, self.far])  # 截面范围

        self.directions = get_ray_directions(h, w, self.focal)  # 计算光线方向

        if self.split == 'train':  # 创建所有光线和RGB数据的缓冲区
            self.image_paths = []  # 图像路径列表
            self.poses = []  # 姿态矩阵列表
            self.all_rays = []  # 所有光线列表
            self.all_rgbs = []  # 所有RGB列表
            for frame in self.meta['frames']:  # 遍历每个帧的元数据
                pose = np.array(frame['transform_matrix'])[:3, :4]  # 获取姿态矩阵
                self.poses += [pose]  # 将姿态矩阵添加到列表中

                image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")  # 图像路径
                self.image_paths += [image_path]  # 将图像路径添加到列表中
                img = Image.open(image_path)  # 打开图像文件
                img = img.resize(self.img_wh, Image.LANCZOS)  # 调整图像大小
                img = self.transform(img)  # 执行数据转换
                img = img.view(4, -1).permute(1, 0)  # 转换为张量
                img = img[:, :3]*img[:, -1:] + (1-img[:, -1:])  # 将A通道混合到RGB中
                self.all_rgbs += [img]  # 将RGB数据添加到列表中

                rays_o, rays_d = get_rays(self.directions, c2w)  # 计算光线方向
                self.all_rays += [torch.cat([rays_o, rays_d,
                                             self.near*torch.ones_like(rays_o[:, :1]),
                                             self.far*torch.ones_like(rays_o[:, :1])],
                                             1)]  # 将光线数据添加到列表中

            self.all_rays = torch.cat(self.all_rays, 0)  # 将光线数据列表拼接为一个张量
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # 将RGB数据列表拼接为一个张量

    def define_transforms(self):  # 定义数据转换方法
        self.transform = T.ToTensor()  # 转换为张量的方法

    def __len__(self):  # 返回数据集的长度
        if self.split == 'train':
            return len(self.all_rays)  # 返回训练集样本的数量
        if self.split == 'val':
            return 8  # 返回验证集样本的数量（最多为8，支持多GPU）
        return len(self.meta['frames'])  # 返回整个数据集样本的数量

    def __getitem__(self, idx):  # 获取数据集中的一个样本
        if self.split == 'train':  # 如果是训练集，从缓冲区中获取数据
            sample = {'rays': self.all_rays[idx],  # 光线数据
                      'rgbs': self.all_rgbs[idx]}  # RGB数据

        else:  # 如果是验证集或测试集，根据元数据创建数据
            frame = self.meta['frames'][idx]  # 获取元数据中的帧信息
            c2w = torch.FloatTensor(frame['transform_matrix'])[:3, :4]  # 获取姿态矩阵

            img = Image.open(os.path.join(self.root_dir, f"{frame['file_path']}.png"))  # 打开图像文件
            img = img.resize(self.img_wh, Image.LANCZOS)  # 调整图像大小
            img = self.transform(img)  # 执行数据转换
            valid_mask = (img[-1]>0).flatten()  # 确定有效的颜色区域
            img = img.view(4, -1).permute(1, 0)  # 转换为张量
            img = img[:, :3]*img[:, -1:] + (1-img[:, -1:])  # 将A通道混合到RGB中

            rays_o, rays_d = get_rays(self.directions, c2w)  # 计算光线方向

            rays = torch.cat([rays_o, rays_d,
                              self.near*torch.ones_like(rays_o[:, :1]),
                              self.far*torch.ones_like(rays_o[:, :1])],
                             1)  # 光线数据

            sample = {'rays': rays,  # 光线数据
                      'rgbs': img,  # RGB数据
                      'c2w': c2w,  # 姿态矩阵
                      'valid_mask': valid_mask}  # 有效的颜色区域掩码

        return sample  # 返回样本字典

