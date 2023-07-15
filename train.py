import os, sys
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from models.rendering import render_rays

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.logging import TestTubeLogger


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams

        self.loss = loss_dict[hparams.loss_type]()  # 使用指定的损失函数

        self.embedding_xyz = Embedding(3, 10)  # 用于编码位置信息的嵌入层
        self.embedding_dir = Embedding(3, 4)  # 用于编码方向信息的嵌入层
        self.embeddings = [self.embedding_xyz, self.embedding_dir]  # 嵌入层列表

        self.nerf_coarse = NeRF()  # 粗糙渲染器
        self.models = [self.nerf_coarse]  # 模型列表，默认只包含粗糙渲染器
        if hparams.N_importance > 0:  # 如果要使用重要性采样，则添加细节渲染器
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

    def decode_batch(self, batch):
        rays = batch['rays']  # 输入射线
        rgbs = batch['rgbs']  # 目标颜色值
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i + self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk,  # chunk size is effective in val mode
                            self.train_dataset.white_back)  # 使用渲染函数渲染射线

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(split='train', **kwargs)  # 训练集
        self.val_dataset = dataset(split='val', **kwargs)  # 验证集

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)  # 获取优化器
        scheduler = get_scheduler(self.hparams, self.optimizer)  # 获取学习率调度器

        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)  # 训练集数据加载器

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1,  # 一次验证一张图片
                          pin_memory=True)  # 验证集数据加载器

    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}  # 获取当前学习率
        rays, rgbs = self.decode_batch(batch)
        results = self(rays)  # 前向传播
        log['train/loss'] = loss = self.loss(results, rgbs)  # 计算损失
        typ = 'fine' if 'rgb_fine' in results else 'coarse'  # 判断是粗糙渲染器还是细节渲染器

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)  # 计算PSNR
            log['train/psnr'] = psnr_

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
                }

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze()  # 去除射线的多余维度
        rgbs = rgbs.squeeze()  # 去除颜色值的多余维度
        results = self(rays)  # 前向传播
        log = {'val_loss': self.loss(results, rgbs)}  # 计算损失
        typ = 'fine' if 'rgb_fine' in results else 'coarse'  # 判断是粗糙渲染器还是细节渲染器

        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1)  # 调整通道维度顺序
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # 调整通道维度顺序
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W))  # 可视化深度图
            stack = torch.stack([img_gt, img, depth])  # 堆叠为一个张量
            self.logger.experiment.add_images('val/GT_pred_depth',
                                              stack, self.global_step)  # 将图片添加到TensorBoard中

        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)  # 计算PSNR
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()  # 计算平均损失
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()  # 计算平均PSNR

        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
                }


if __name__ == '__main__':
    hparams = get_opts()  # 获取命令行参数
    system = NeRFSystem(hparams)  # 创建NeRF系统实例
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(f'ckpts/{hparams.exp_name}',
                                                                '{epoch:d}'),
                                          monitor='val/loss',
                                          mode='min',
                                          save_top_k=1, )

    logger = TestTubeLogger(
        save_dir="logs",
        name=hparams.exp_name,
        debug=False,
        create_git_tag=False
    )

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      early_stop_callback=None,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      distributed_backend='ddp' if hparams.num_gpus > 1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler=hparams.num_gpus == 1)

    trainer.fit(system)  # 训练模型
