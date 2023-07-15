import torch
from torch import searchsorted

__all__ = ['render_rays']

"""
Function dependencies: (-> means function calls)

@render_rays -> @inference

@render_rays -> @sample_pdf if there is fine model
"""


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):#从离散概率分布中采样

    #计算输入权重（weights）的形状，并将其与eps相加，以避免除以零。然后，将权重归一化为概率密度函数（pdf），
    # 并计算累积分布函数（cdf）。最后，通过在cdf的左侧填充一个零的列，将cdf扩展为0~1（包含1）
    N_rays, N_samples_ = weights.shape
    weights = weights + eps  # prevent division by zero )
    pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples),累积分布函数 cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive  填充为0~1（包含1

    #如果是确定性采样（det为True），则使用等间隔的u值进行采样；否则，使用均匀随机采样生成u值。
    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    #根据u值在cdf上进行二分搜索，找到相应的下标（inds）。然后，根据inds计算采样下限（below）和上限（above）。
    # 接下来，根据inds_sampled从cdf和bins中收集对应的值，得到对应的cdf_g和bins_g。
    inds = searchsorted(cdf, u, side='right')
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    #计算采样间隔（denom）并处理其中小于eps的值，以避免除以零。最后，根据公式进行采样，得到最终的样本值
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < eps] = 1  # # denom等于0意味着一个bin的权重为0，在这种情况下它将不会被采样，因此对它的值进行任何设置都可以（这里设为1）
    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples

def render_rays(models,
                embeddings,
                rays,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024 * 32,
                white_back=False,
                test_time=False
                ):
    def inference(model, embedding_xyz, xyz_, dir_, dir_embedded, z_vals, weights_only=False):

        # 获取采样点的数量
        N_samples_ = xyz_.shape[1]

        # 整理输入的形状
        xyz_ = xyz_.view(-1, 3)  # (N_rays*N_samples_, 3)
        if not weights_only:
            dir_embedded = torch.repeat_interleave(dir_embedded, repeats=N_samples_, dim=0)
            # (N_rays*N_samples_, embed_dir_channels)

        # 嵌入位置信息
        B = xyz_.shape[0]
        out_chunks = []
        for i in range(0, B, chunk):
            # 嵌入位置
            xyz_embedded = embedding_xyz(xyz_[i:i + chunk])
            if not weights_only:
                xyzdir_embedded = torch.cat([xyz_embedded,
                                             dir_embedded[i:i + chunk]], 1)
            else:
                xyzdir_embedded = xyz_embedded
            out_chunks += [model(xyzdir_embedded, sigma_only=weights_only)]

        out = torch.cat(out_chunks, 0)

        # 如果只需要权重，则返回权重
        if weights_only:
            sigmas = out.view(N_rays, N_samples_)
        else:
            rgbsigma = out.view(N_rays, N_samples_, 4)
            rgbs = rgbsigma[..., :3]  # (N_rays, N_samples_, 3)
            sigmas = rgbsigma[..., 3]  # (N_rays, N_samples_)

        # 使用体积渲染来转换这些值
        # 计算每个深度采样点之间的距离
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # (N_rays, 1) 最后一个delta是无穷大
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # 将每个距离乘以其对应方向光线的范数，以转换为现实世界距离（考虑到非单位方向）
        deltas = deltas * torch.norm(dir_.unsqueeze(1), dim=-1)

        # 添加高斯噪声
        noise = torch.randn(sigmas.shape, device=sigmas.device) * noise_std

        # 计算alpha值
        alphas = 1 - torch.exp(-deltas * torch.relu(sigmas + noise))  # (N_rays, N_samples_)
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)  # [1, a1, a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted, -1)[:, :-1]  # (N_rays, N_samples_)
        weights_sum = weights.sum(1)  # (N_rays)，沿射线累积的不透明度
        # 等于数学上的"1 - (1-a1)(1-a2)...(1-an)"

        if weights_only:
            return weights

        # 计算最终加权输出
        rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (N_rays, 3)
        depth_final = torch.sum(weights * z_vals, -1)  # (N_rays)

        if white_back:
            rgb_final = rgb_final + 1 - weights_sum.unsqueeze(-1)

        return rgb_final, depth_final, weights

        # 从列表中提取模型

    model_coarse = models[0]
    embedding_xyz = embeddings[0]
    embedding_dir = embeddings[1]

    # 分解输入
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)

    # 嵌入方向信息
    dir_embedded = embedding_dir(rays_d)  # (N_rays, embed_dir_channels)

    # 采样深度点
    z_steps = torch.linspace(0, 1, N_samples, device=rays.device)  # (N_samples)
    if not use_disp:  # 在深度空间中使用线性采样
        z_vals = near * (1 - z_steps) + far * z_steps
    else:  # 使用视差空间中的线性采样
        z_vals = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)

    z_vals = z_vals.expand(N_rays, N_samples)

    if perturb > 0:  # 扰动采样深度（z_vals）
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) 区间中点
        # 获取采样点之间的区间
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand(z_vals.shape, device=rays.device)
        z_vals = lower + (upper - lower) * perturb_rand

    xyz_coarse_sampled = rays_o.unsqueeze(1) + \
                         rays_d.unsqueeze(1) * z_vals.unsqueeze(2)  # (N_rays, N_samples, 3)

    # 在测试时间计算粗糙模型的权重
    if test_time:
        weights_coarse = \
            inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=True)
        result = {'opacity_coarse': weights_coarse.sum(1)}
    else:
        # 在训练时间计算粗糙模型的颜色、深度和权重
        rgb_coarse, depth_coarse, weights_coarse = \
            inference(model_coarse, embedding_xyz, xyz_coarse_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)
        result = {'rgb_coarse': rgb_coarse,
                  'depth_coarse': depth_coarse,
                  'opacity_coarse': weights_coarse.sum(1)
                  }

    if N_importance > 0:  # 为细模型采样点
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) 区间中点
        z_vals_ = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1],
                             N_importance, det=(perturb == 0)).detach()
        # detach，以防止梯度从这里传播到weights_coarse

        z_vals, _ = torch.sort(torch.cat([z_vals, z_vals_], -1), -1)

        xyz_fine_sampled = rays_o.unsqueeze(1) + \
                           rays_d.unsqueeze(1) * z_vals.unsqueeze(2)
        # (N_rays, N_samples+N_importance, 3)

        model_fine = models[1]
        rgb_fine, depth_fine, weights_fine = \
            inference(model_fine, embedding_xyz, xyz_fine_sampled, rays_d,
                      dir_embedded, z_vals, weights_only=False)

        result['rgb_fine'] = rgb_fine
        result['depth_fine'] = depth_fine
        result['opacity_fine'] = weights_fine.sum(1)

    return result