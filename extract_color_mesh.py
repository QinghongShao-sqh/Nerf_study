import torch
import os
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import mcubes
import open3d as o3d
from plyfile import PlyData, PlyElement
from argparse import ArgumentParser

from models.rendering import *
from models.nerf import *

from utils import load_ckpt

from datasets import dataset_dict

torch.backends.cudnn.benchmark = True


def get_opts():
    parser = ArgumentParser()  # 创建一个参数解析器

    # 添加命令行参数
    parser.add_argument('--root_dir', type=str,
                        default='/data/nerf_synthetic/lego',
                        help='数据集的根目录')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff'],
                        help='要验证的数据集名称')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='场景名称，用作输出的PLY文件名')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='图像的分辨率（宽, 高）')

    parser.add_argument('--N_samples', type=int, default=64,
                        help='用于推断累积不透明度的采样数')
    parser.add_argument('--chunk', type=int, default=32 * 1024,
                        help='用于避免OOM的分块大小')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='预训练模型的路径')

    parser.add_argument('--N_grid', type=int, default=256,
                        help='网格的大小（一边），越大=越高分辨率')
    parser.add_argument('--x_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='物体的x范围')
    parser.add_argument('--y_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='物体的y范围')
    parser.add_argument('--z_range', nargs="+", type=float, default=[-1.0, 1.0],
                        help='物体的z范围')
    parser.add_argument('--sigma_threshold', type=float, default=20.0,
                        help='认为位置被占据的阈值')
    parser.add_argument('--occ_threshold', type=float, default=0.2,
                        help='认为顶点被遮挡的阈值。较大=较少的遮挡像素')

    #### 使用顶点法线的方法 ####
    parser.add_argument('--use_vertex_normal', action="store_true",
                        help='使用顶点法线来计算颜色')
    parser.add_argument('--N_importance', type=int, default=64,
                        help='用于推断累积不透明度的精细采样数')
    parser.add_argument('--near_t', type=float, default=1.0,
                        help='开始射线的近处边界因子')

    return parser.parse_args()



@torch.no_grad()
def f(models, embeddings, rays, N_samples, N_importance, chunk, white_back):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i + chunk],
                        N_samples,
                        False,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


if __name__ == "__main__":
    args = get_opts()

    kwargs = {'root_dir': args.root_dir,  # 设置关键字参数root_dir
              'img_wh': tuple(args.img_wh)}  # 设置关键字参数img_wh为元组(args.img_wh)
    if args.dataset_name == 'llff':  # 如果数据集名称为llff
        kwargs['spheric_poses'] = True  # 设置关键字参数spheric_poses为True
        kwargs['split'] = 'test'  # 设置关键字参数split为'test'
    else:  # 如果数据集名称不为llff
        kwargs['split'] = 'train'  # 设置关键字参数split为'train'
    dataset = dataset_dict[args.dataset_name](**kwargs)  # 根据数据集名称创建数据集对象，传入关键字参数

    embedding_xyz = Embedding(3, 10)  # 创建嵌入层对象embedding_xyz，输入维度为3，输出维度为10
    embedding_dir = Embedding(3, 4)  # 创建嵌入层对象embedding_dir，输入维度为3，输出维度为4
    embeddings = [embedding_xyz, embedding_dir]  # 将embedding_xyz和embedding_dir添加到列表embeddings中
    nerf_fine = NeRF()  # 创建NeRF模型对象nerf_fine
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')  # 加载训练好的nerf_fine模型
    nerf_fine.cuda().eval()  # 将nerf_fine模型移动到GPU并设置为评估模式

    # 定义查询的稠密网格
    N = args.N_grid  # 网格维度
    xmin, xmax = args.x_range  # 网格x轴范围
    ymin, ymax = args.y_range  # 网格y轴范围
    zmin, zmax = args.z_range  # 网格z轴范围
    # assert xmax-xmin == ymax-ymin == zmax-zmin, 'the ranges must have the same length!'
    x = np.linspace(xmin, xmax, N)  # 在x轴范围内生成N个等间距的点
    y = np.linspace(ymin, ymax, N)  # 在y轴范围内生成N个等间距的点
    z = np.linspace(zmin, zmax, N)  # 在z轴范围内生成N个等间距的点

    xyz_ = torch.FloatTensor(np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3)).cuda()  # 将网格点坐标转换为张量，并移动到GPU上
    dir_ = torch.zeros_like(xyz_).cuda()  # 创建与xyz_相同大小的零张量，并移动到GPU上
    # sigma独立于方向，因此在这里的任何值都会产生相同的结果

    # 预测每个网格位置的sigma（占据率）
    print('Predicting occupancy ...')
    with torch.no_grad():  # 关闭梯度计算
        B = xyz_.shape[0]  # B为xyz_的行数
        out_chunks = []  # 存储输出块的列表
        for i in tqdm(range(0, B, args.chunk)):  # tqdm用于显示进度条
            xyz_embedded = embedding_xyz(xyz_[i:i + args.chunk])  # 嵌入xyz_坐标
            dir_embedded = embedding_dir(dir_[i:i + args.chunk])  # 嵌入方向
            xyzdir_embedded = torch.cat([xyz_embedded, dir_embedded], 1)  # 将嵌入的xyz坐标和方向连接起来
            out_chunks += [nerf_fine(xyzdir_embedded)]  # 将nerf_fine模型应用于嵌入的xyz坐标和方向，并将输出添加到out_chunks中
        rgbsigma = torch.cat(out_chunks, 0)  # 将out_chunks中的张量连接起来

    sigma = rgbsigma[:, -1].cpu().numpy()  # 获取sigma（占据率）并将其转换为numpy数组，并移动到CPU上
    sigma = np.maximum(sigma, 0).reshape(N, N, N)  # 将sigma的值限制在0以上，并将其重塑为N*N*N的形状

    # 执行Marching Cubes算法以获取顶点和三角形网格
    print('Extracting mesh ...')
    vertices, triangles = mcubes.marching_cubes(sigma, args.sigma_threshold)  # 使用Marching Cubes算法提取网格顶点和三角形

    ##### 从提取网格到这里，与原始代码库相同。 ######

    vertices_ = (vertices / N).astype(np.float32)  # 将顶点坐标还原到原始网格坐标范围内，并转换为float32类型
    ## 反转x和y坐标（为什么？可能是因为Marching Cubes算法的原因）
    x_ = (ymax - ymin) * vertices_[:, 1] + ymin  # 反转x坐标
    y_ = (xmax - xmin) * vertices_[:, 0] + xmin  # 反转y坐标
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax - zmin) * vertices_[:, 2] + zmin  # 还原z坐标到原始网格坐标范围内
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]  # 设置顶点的数据类型为('x', 'f4'), ('y', 'f4'), ('z', 'f4')

    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])  # 创建一个空数组来存储面的信息
    face['vertex_indices'] = triangles  # 将三角形的顶点索引存储到face中

    PlyData([PlyElement.describe(vertices_[:, 0], 'vertex'),  # 创建PlyData对象，包含顶点信息
             PlyElement.describe(face, 'face')]).write(f'{args.scene_name}.ply')  # 将顶点和面的信息写入PLY文件

    # 通过保留最大簇来去除噪声
    print('Removing noise ...')
    mesh = o3d.io.read_triangle_mesh(f"{args.scene_name}.ply")  # 读取PLY文件为三角形网格对象
    idxs, count, _ = mesh.cluster_connected_triangles()  # 将连接的三角形进行聚类
    max_cluster_idx = np.argmax(count)  # 获取最大簇的索引
    triangles_to_remove = [i for i in range(len(face)) if idxs[i] != max_cluster_idx]  # 需要移除的三角形索引
    mesh.remove_triangles_by_index(triangles_to_remove)  # 移除三角形
    mesh.remove_unreferenced_vertices()  # 移除未引用的顶点
    print(
        f'Mesh has {len(mesh.vertices) / 1e6:.2f} M vertices and {len(mesh.triangles) / 1e6:.2f} M faces.')  # 打印网格顶点数和面数

    vertices_ = np.asarray(mesh.vertices).astype(np.float32)  # 获取网格顶点并转换为float32类型
    triangles = np.asarray(mesh.triangles)  # 获取网格三角形

    # 执行颜色预测
    # 第0步。定义常量（图像宽度、高度和内参矩阵）
    W, H = args.img_wh  # 图像宽度和高度
    K = np.array([[dataset.focal, 0, W / 2],  # 内参矩阵
                  [0, dataset.focal, H / 2],
                  [0, 0, 1]]).astype(np.float32)

    # 第1步。将顶点转换为世界坐标系
    N_vertices = len(vertices_)  # 顶点数量
    vertices_homo = np.concatenate

            ## compute the color on these projected pixel coordinates
            ## using bilinear interpolation.
            ## NOTE: opencv's implementation has a size limit of 32768 pixels per side,
            ## so we split the input into chunks.
            colors = []
            remap_chunk = int(3e4)
            for i in range(0, N_vertices, remap_chunk):
                colors += [cv2.remap(image,
                                     vertices_image[i:i + remap_chunk, 0],
                                     vertices_image[i:i + remap_chunk, 1],
                                     interpolation=cv2.INTER_LINEAR)[:, 0]]
            colors = np.vstack(colors)  # (N_vertices, 3)

            # 预测每个顶点的遮挡情况
            # 我们利用NeRF的概念，通过构建从相机出发并命中每个顶点的光线
            # 通过计算沿着这条路径累积的不透明度，我们可以知道顶点是否被遮挡。
            # 对于在每个输入视图中都被遮挡的顶点，我们假设其颜色与面向我们的相邻顶点的颜色相同。
            # （想象一下一个表面，有一面朝向我们：我们假设另一面的颜色相同）

            # 光线的起点是相机的起点
            rays_o = torch.FloatTensor(dataset.poses[idx][:, -1]).expand(N_vertices, 3)
            # 光线的方向是从相机起点指向顶点的向量
            rays_d = torch.FloatTensor(vertices_) - rays_o  # (N_vertices, 3)
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            near = dataset.bounds.min() * torch.ones_like(rays_o[:, :1])
            # 远平面是顶点的深度，因为我们想要的是从相机起点到顶点的路径上累积的不透明度
            far = torch.FloatTensor(depth) * torch.ones_like(rays_o[:, :1])
            results = f([nerf_fine], embeddings,
                        torch.cat([rays_o, rays_d, near, far], 1).cuda(),
                        args.N_samples,
                        0,
                        args.chunk,
                        dataset.white_back)
            opacity = results['opacity_coarse'].cpu().numpy()[:, np.newaxis]  # (N_vertices, 1)
            opacity = np.nan_to_num(opacity, 1)

            non_occluded = np.ones_like(non_occluded_sum) * 0.1 / depth  # 根据逆深度进行加权
            # 靠近相机的部分更加可信
            non_occluded += opacity < args.occ_threshold

            v_color_sum += colors * non_occluded
            non_occluded_sum += non_occluded

            # 将输出合并并写入文件
            if args.use_vertex_normal:
                v_colors = results['rgb_fine'].cpu().numpy() * 255.0
            else:  # 合并的颜色是所有视图中的平均颜色
                v_colors = v_color_sum / non_occluded_sum
            v_colors = v_colors.astype(np.uint8)
            v_colors.dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
            vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
            vertex_all = np.empty(N_vertices, vertices_.dtype.descr + v_colors.dtype.descr)
            for prop in vertices_.dtype.names:
                vertex_all[prop] = vertices_[prop][:, 0]
            for prop in v_colors.dtype.names:
                vertex_all[prop] = v_colors[prop][:, 0]

            face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
            face['vertex_indices'] = triangles

            PlyData([PlyElement.describe(vertex_all, 'vertex'),
                     PlyElement.describe(face, 'face')]).write(f'{args.scene_name}.ply')

            print('Done!')
