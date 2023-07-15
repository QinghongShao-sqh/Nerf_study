import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *


def normalize(v):
    # 归一化向量
    return v/np.lina;g.norm(v)

def average_poses(poses):
    #计算平均位姿
    center = poses[...,3].mean(0)     #.mean 计算平均
    #for example
    # array([[1, 2],
    #        [3, 4]])
    # >> > np.mean(a)
    # 2.5
    # 计算 z轴
    z = normalize(poses[...,2].mean(0))
    # 计算y `轴(不需要normalize，因为它不是最终输出)
    y_ = poses[...,1].mean(0)
    #计算x轴
    x = normalize(np.crpssy(y_,z))
    # 在三维几何中，向量a和向量b的叉乘结果是一个向量，更为熟知的叫法是法向量，该向量垂直于a和b向量构成的平面。叉乘的结果是个向量，方向在z轴上,在二维空间里，让我们暂时忽略它的方向，将结果看成一个向量，那么这个结果类似于上述的点积，有公式：
    # 计算两个向量（向量数组）的叉乘  z,x 叉乘， 计算结果就是z，x向量所在平面的法向量（垂直向量）
    y=np.cross(z,x)
    # stack  把 xyz以及center 数组堆叠起来
    #参考https://blog.csdn.net/qq_37006426/article/details/116238818?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168884342416800180672628%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168884342416800180672628&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-116238818-null-null.142^v88^control_2,239^v2^insert_chatgpt&utm_term=np.stack&spm=1018.2226.3001.4187
    pose_avg = np.stack([x,y,z,center],1)

    return  pose_avg

def center_poses(poses):
    # inpputs: （ 图片数量N images ， 3， 4）
   # outputs:
#         （N_images ,3,4） 位姿中心
#          pose_avg:(3,4)  average pose
    pose_avg = average_poses(poses)
    pose_avg_homo =np.eye(4)
    pose_avg_homo[:3] = pose_avg
    #通过简单地将0、0、0、1加到最后一行 ,转换为齐次坐标以加快计算速度
    last_row = np.tile(np.array([0,0,0,1]),(len(poses),1,1)) # (N_images ,1,4)
    poses_homo = \
        np.concatenate([poses , last_row],1)  #  N_images,4,4    homogeneous coordinate 其次坐标
                       # np.linalg.inv 求矩阵的逆矩阵
    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # N_images , 4,4
    poses_centered = poses_centered[:,:3] # N_images ,3, 4

    return  poses_centered,np.linalg.inv(pose_avg_homo)


def create_spiral_poses(radii,focus_depth, n_poses=120):
#Computes poses that follow a spiral path for rendering purpose.

    # inputs:
             # radii:(3) radii of the spiral for each axis
             # focus_depth: float ,the depth that the spiral poses look at
             # int, number of poses to create along the path
    # output:
             #poses_spiral: (n_poses,3,4) the poses in the spiral path
    poses_spiral = []
    for t  in np.linspace(0, 4* np.pi, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        #螺旋的参数化函数
        center =np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii
        #视图z轴是指向@focus_depth平面的向量 to @center
        z = normalize(center - np.array([0,0, -focus_depth]))
        # compute other axes in average_poses

        y_ = np.array([0,1,0])# (3)
        x = normalize(np.cross(y_,z))

        y= np.cross(z,x) # 3
        poses_spiral += [np.stack([x,y,z, center] , 1)] # 3,4

    return  np.stack(poses_spiral , 0) # n_poses,3,4


class LLFFDataset(Dataset):
    def __init__(self, root_dir,split='train', img_wh=(504,378), spheric_poses=False,val_num=1):

        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.spheric_poses = spheric_poses
        self.val_num = max(1,val_num) # 大于等于1
        self.define_transforms()

        self.read_meta()
        self.white_back = False

    def read_meta(self):
        # 读取一个名为poses_bounds.npy的文件，其中包含了姿态和边界信息。poses_bounds是一个形状为(N_images, 17)的NumPy数组
        poses_bounds = np.load(os.path.join(self.root_dir,
                                            'poses_bounds.npy'  # N_imgaes 17
                                            ))
        #使用glob模块找到root_dir下的所有图片路径
        self.image_paths = sorted (glob.glob(os.path.join(self.root_dir,'images/*')))
        # load image
        # if用来确保图像和姿态的数量对应
        if self.split in ['train','val']:
            assert  len(poses_bounds) == len(self.image_paths),\
                '图片和位姿 匹配错误，请重新运行Colmap'
        #从poses_bounds中提取前15列，并将其形状重塑为(N_images, 3, 5)的数组。这些数据用于存储姿态信息
        poses = poses_bounds[: , :15].reshape(-1,3,5) #N_images ,3,5
        #从poses_bounds中提取最后两列，并将其存储在bounds属性中。这些数据用于存储边界信息
        self.bounds = poses_bounds[: , -2:] # N_images, 2

        # first 根据训练分辨率调整焦距
        H,W,self.focal = poses[ 0, :, -1]
        assert  H*self.img_wh[0] == W*self.img_wh[1], \
            f'You must set @img_wh to have the same aspect ratio as ({W}, {H}) !'

        self.focal *= self.img_wh[0]/W

        # 步骤2：修正姿态
        # 原始姿态的旋转顺序是"down right back"，将其转换为"right up back"
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # 对姿态进行中心化处理
        self.poses, self.pose_avg = center_poses(poses)
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_idx = np.argmin(distances_from_center)  # 选择离中心图像最近的图像作为验证图像

        # 步骤3：校正尺度，使最近的深度略大于1.0
        near_original = self.bounds.min()
        scale_factor = near_original * 0.75  # 0.75为默认参数，最近深度为1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        # 计算所有像素的射线方向，所有图像都相同（相同的H、W、focal）
        self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal)  # (H, W, 3)

        if self.split == 'train':  # 创建所有射线和RGB数据的缓冲区
            # 使用前N_images-1个图像进行训练，最后一个是验证图像
            self.all_rays = []
            self.all_rgbs = []
            for i, image_path in enumerate(self.image_paths):
                if i == val_idx:  # 排除验证图像
                    continue
                c2w = torch.FloatTensor(self.poses[i])

                img = Image.open(image_path).convert('RGB')
                assert img.size[1] * self.img_wh[0] == img.size[0] * self.img_wh[1], \
                    f'{image_path}的宽高比与img_wh不一致，请检查数据！'
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (3, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
                self.all_rgbs += [img]

                rays_o, rays_d = get_rays(self.directions, c2w)  # 都是(h*w, 3)
                if not self.spheric_poses:
                    near, far = 0, 1
                    rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                  self.focal, 1.0, rays_o, rays_d)
                else:
                    near = self.bounds.min()
                    far = min(8 * near, self.bounds.max())  # 只关注中心物体

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             near * torch.ones_like(rays_o[:, :1]),
                                             far * torch.ones_like(rays_o[:, :1])],
                                            1)]  # (h*w, 8)

            self.all_rays = torch.cat(self.all_rays, 0)  # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # ((N_images-1)*h*w, 3)

        elif self.split == 'val':
            print('验证图像是', self.image_paths[val_idx])
            self.c2w_val = self.poses[val_idx]
            self.image_path_val = self.image_paths[val_idx]

        else:  # 测试时创建参数化渲染路径
            if self.split.endswith('train'):  # 在训练集上进行测试
                self.poses_test = self.poses
            elif not self.spheric_poses:
                focus_depth = 3.5  # 编码，这个数值在数值上接近原始代码中给出的公式
                # 数学上，如果near=1，far=无穷大，则这个数值将收敛到4
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                self.poses_test = create_spiral_poses(radii, focus_depth)
            else:
                radius = 1.1 * self.bounds.min()
                self.poses_test = create_spheric_poses(radius)
#定义了数据集的transforms，将图像转换为张量
    def define_transforms(self):
        self.transform = T.ToTensor()

    # 根据数据集的划分方式（split）返回不同的长度。如果是训练集（'train'），返回all_rays的长度；如果是验证集（'val'），返回val_num；如果是测试集，返回poses_test的长度。

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return self.val_num
        return len(self.poses_test)
#如果是训练集，从缓冲区（all_rays和all_rgbs）中获取对应索引（idx）的rays和rgbs。
    def __getitem__(self, idx):
        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}
#如果是训练集，从缓冲区（all_rays和all_rgbs）中获取对应索引（idx）的rays和rgbs。
        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.c2w_val)
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])
#根据方向（directions）和c2w获取rays_o和rays_d。如果不使用球形姿态（spheric_poses为False），将rays_o和rays_d转换为归一化设备坐标（NDC）空间。否则，根据bounds计算near和far
            rays_o, rays_d = get_rays(self.directions, c2w)
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
            else:
                near = self.bounds.min()
                far = min(8 * near, self.bounds.max())
#将rays_o、rays_d、near和far拼接为rays，形状为(h*w, 8)
            rays = torch.cat([rays_o, rays_d,
                              near * torch.ones_like(rays_o[:, :1]),
                              far * torch.ones_like(rays_o[:, :1])],
                             1)  # (h*w, 8)
#将rays_o、rays_d、near和far拼接为rays，形状为(h*w, 8)
            sample = {'rays': rays,
                      'c2w': c2w}
#如果是验证集，从预定义的image_path_val路径中读取图像，并进行大小调整和转换为张量。将图像reshape为(h*w, 3)的形状，并将其存储在sample字典中的'rgbs'键下
            if self.split == 'val':
                img = Image.open(self.image_path_val).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img)  # (3, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3)
                sample['rgbs'] = img

        return sample












