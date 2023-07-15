import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser

from models.rendering import render_rays
from models.nerf import *

from utils import load_ckpt
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *

torch.backends.cudnn.benchmark = True


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,  # 添加root_dir命令行参数，类型为字符串
                        default='/data/nerf_synthetic/lego',
                        help='root directory of dataset')  # 数据集的根目录
    parser.add_argument('--dataset_name', type=str, default='blender',  # 添加dataset_name命令行参数，类型为字符串，默认为'blender'
                        choices=['blender', 'llff'],
                        help='which dataset to validate')  # 需要验证的数据集
    parser.add_argument('--scene_name', type=str, default='test',  # 添加scene_name命令行参数，类型为字符串，默认为'test'
                        help='scene name, used as output folder name')  # 场景名称，用作输出文件夹名称
    parser.add_argument('--split', type=str, default='test',  # 添加split命令行参数，类型为字符串，默认为'test'
                        help='test or test_train')  # 测试集或训练集
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],  # 添加img_wh命令行参数，类型为整型列表，默认为[800, 800]
                        help='resolution (img_w, img_h) of the image')  # 图像的分辨率（img_w，img_h）
    parser.add_argument('--spheric_poses', default=False, action="store_true",  # 添加spheric_poses命令行参数，类型为布尔值，默认为False
                        help='whether images are taken in spheric poses (for llff)')  # 图像是否采用球形姿态（用于llff数据集）

    parser.add_argument('--N_samples', type=int, default=64,  # 添加N_samples命令行参数，类型为整型，默认为64
                        help='number of coarse samples')  # 粗糙采样点的数量
    parser.add_argument('--N_importance', type=int, default=128,  # 添加N_importance命令行参数，类型为整型，默认为128
                        help='number of additional fine samples')  # 额外细致采样点的数量
    parser.add_argument('--use_disp', default=False, action="store_true",  # 添加use_disp命令行参数，类型为布尔值，默认为False
                        help='use disparity depth sampling')  # 是否使用视差深度采样
    parser.add_argument('--chunk', type=int, default=32*1024*4,  # 添加chunk命令行参数，类型为整型，默认为32*1024*4
                        help='chunk size to split the input to avoid OOM')  # 为了避免OOM，将输入分块的块大小

    parser.add_argument('--ckpt_path', type=str, required=True,  # 添加ckpt_path命令行参数，类型为字符串，必需
                        help='pretrained checkpoint path to load')  # 需要加载的预训练模型的路径

    parser.add_argument('--save_depth', default=False, action="store_true",  # 添加save_depth命令行参数，类型为布尔值，默认为False
                        help='whether to save depth prediction')  # 是否保存深度预测
    parser.add_argument('--depth_format', type=str, default='pfm',  # 添加depth_format命令行参数，类型为字符串，默认为'pfm'
                        choices=['pfm', 'bytes'],
                        help='which format to save')  # 保存深度的格式

    return parser.parse_args()  # 解析命令行参数并返回

@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, N_samples, N_importance, use_disp,
                      chunk,
                      white_back):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]  # 获取输入rays的batch size
    chunk = 1024*32  # 设置chunk大小
    results = defaultdict(list)  # 创建一个空的defaultdict(list)用于存储结果
    for i in range(0, B, chunk):  # 对输入rays进行分块处理
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],  # 从rays中取出一块进行渲染
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=True)  # 进行光线渲染

        for k, v in rendered_ray_chunks.items():  # 遍历渲染结果的键值对
            results[k] += [v]  # 将渲染结果添加到results字典中

    for k, v in results.items():  # 遍历results字典
        results[k] = torch.cat(v, 0)  # 将结果连接起来
    return results  # 返回结果

if __name__ == "__main__":
    args = get_opts()  # 获取命令行参数
    w, h = args.img_wh  # 获取图像的宽度和高度

    kwargs = {'root_dir': args.root_dir,  # 构造kwargs字典，包含数据集相关参数
              'split': args.split,
              'img_wh': tuple(args.img_wh)}
    if args.dataset_name == 'llff':  # 如果数据集名称为llff，则添加spheric_poses参数
        kwargs['spheric_poses'] = args.spheric_poses
    dataset = dataset_dict[args.dataset_name](**kwargs)  # 根据数据集名称创建数据集对象

    embedding_xyz = Embedding(3, 10)  # 创建位置嵌入编码器
    embedding_dir = Embedding(3, 4)  # 创建方向嵌入编码器
    nerf_coarse = NeRF()  # 创建粗糙采样的NeRF模型
    nerf_fine = NeRF()  # 创建细致采样的NeRF模型
    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')  # 加载预训练的粗糙采样模型
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')  # 加载预训练的细致采样模型
    nerf_coarse.cuda().eval()  # 将粗糙采样模型移动到GPU并设置为评估模式
    nerf_fine.cuda().eval()  # 将细致采样模型移动到GPU并设置为评估模式

    models = [nerf_coarse, nerf_fine]  # 创建模型列表
    embeddings = [embedding_xyz, embedding_dir]  # 创建嵌入编码器列表

    imgs = []  # 创建空列表用于存储图像路径
    psnrs = []  # 创建空列表用于存储PSNR值
    dir_name = f'results/{args.dataset_name}/{args.scene_name}'  # 创建保存结果的文件夹名称
    os.makedirs(dir_name, exist_ok=True)  # 创建结果文件夹

    for i in tqdm(range(len(dataset))):  # 遍历数据集中的每个样本
        sample = dataset[i]  # 获取样本数据
        rays = sample['rays'].cuda()  # 将光线数据移动到GPU
        results