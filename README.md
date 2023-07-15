#  Nerf Project with detailed annotations



# Introduction

This is a replication project of my Nerf paper. Specific features include sharing my notes on the code as I went from nerf from 0 to 1.

I combined a lot of code in the project with the results of the paper and discussion with seniors and students in Station bilibili.

I hope this repository can help you better understand and reproduce the code of Nerf.

关于这个仓库，他是我在学习Nerf过程中的一些总结以及笔记。他具体包括大量的针对代码的注解，也包括我的一些思考。在这之前我曾经几个月发布了一个仓库叫nerf2020，其中是对nerf项目的缩减版项目复现，其实存在很多缺陷，该项目的架构以及结果会更接近于论文表现。

我将项目中的许多代码与论文的结果结合起来，并与bilibili以及关注我公众号的同学讨论。

我希望这个仓库可以帮助您更好地理解和重现Nerf的代码。





Paper Website：[arxiv.org/pdf/2003.08934v2.pdf](https://arxiv.org/pdf/2003.08934v2.pdf)

Project Reference Website：[kwea123/nerf_pl: NeRF (Neural Radiance Fields) and NeRF in the Wild using pytorch-lightning (github.com)](https://github.com/kwea123/nerf_pl)



论文地址：[arxiv.org/pdf/2003.08934v2.pdf](https://arxiv.org/pdf/2003.08934v2.pdf)

项目参考地址：[kwea123/nerf_pl: NeRF (Neural Radiance Fields) and NeRF in the Wild using pytorch-lightning (github.com)](https://github.com/kwea123/nerf_pl)

这里强烈感谢上面提及的提供项目参考的地址，其仓库代码给了我很大的帮助！



下面是我运行的一些结果，可以参考
Output Image/scene

![008]([M:\Edit\Nerf_study\results\blender\lego\008.png](https://github.com/KEXA1/Nerf_study/blob/master/results/blender/lego/003.png))

![014]([M:\Edit\Nerf_study\results\blender\lego\014.png](https://github.com/KEXA1/Nerf_study/blob/master/results/blender/lego/001.png))



原图为(Input Image)

![1]([M:\Edit\Nerf_study\data\plant\images\1.jpg](https://github.com/KEXA1/Nerf_study/blob/master/data/plant/images/1.png))



# HOW TO  START  !

## Blender
<details>
  <summary>Steps</summary>

### Data download

Download `nerf_synthetic.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

### Training model

Run (example)
```
python train.py \
   --dataset_name blender \
   --root_dir $BLENDER_DIR \
   --N_importance 64 --img_wh 400 400 --noise_std 0 \
   --num_epochs 16 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
   --exp_name exp
```

These parameters are chosen to best mimic the training settings in the original repo. See [opt.py](opt.py) for all configurations.

NOTE: the above configuration doesn't work for some scenes like `drums`, `ship`. In that case, consider increasing the `batch_size` or change the `optimizer` to `radam`. I managed to train on all scenes with these modifications.

You can monitor the training process by `tensorboard --logdir logs/` and go to `localhost:6006` in your browser.
</details>

## LLFF
<details>
  <summary>Steps</summary>

### Data download

Download `nerf_llff_data.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

### Training model

Run (example)
```
python train.py \
   --dataset_name llff \
   --root_dir $LLFF_DIR \
   --N_importance 64 --img_wh 504 378 \
   --num_epochs 30 --batch_size 1024 \
   --optimizer adam --lr 5e-4 \
   --lr_scheduler steplr --decay_step 10 20 --decay_gamma 0.5 \
   --exp_name exp
```

These parameters are chosen to best mimic the training settings in the original repo. See [opt.py](opt.py) for all configurations.

You can monitor the training process by `tensorboard --logdir logs/` and go to `localhost:6006` in your browser.
</details>

## Your own data
<details>
  <summary>Steps</summary>

1. Install [COLMAP](https://github.com/colmap/colmap) following [installation guide](https://colmap.github.io/install.html)
2. Prepare your images in a folder (around 20 to 30 for forward facing, and 40 to 50 for 360 inward-facing)
3. Clone [LLFF](https://github.com/Fyusion/LLFF) and run `python img2poses.py $your-images-folder`
4. Train the model using the same command as in [LLFF](#llff). If the scene is captured in a 360 inward-facing manner, add `--spheric` argument.

For more details of training a good model, please see the video [here](#colab).
</details>

## Pretrained models and logs
Download the pretrained models and training logs in [release](https://github.com/kwea123/nerf_pl/releases).

## Comparison with other repos

|           | training GPU memory in GB | Speed (1 step) |
| :---:     |  :---:     | :---:   | 
| [Original](https://github.com/bmild/nerf)  |  8.5 | 0.177s |
| [Ref pytorch](https://github.com/yenchenlin/nerf-pytorch)  |  6.0 | 0.147s |
| This repo | 3.2 | 0.12s |


### Finally, If you have any questions about my project, please leave a comment!
### If my project can help you, I hope you can give me a star!
##The platform where I often move
### Bilibili(To update my paper sharing video) [出门吃三碗饭的个人空间_哔哩哔哩_bilibili](https://space.bilibili.com/38035003?spm_id_from=333.1007.0.0)
### CSDN (To update my blog)[(644条消息) 出门吃三碗饭的博客_CSDN博客-python,大学学习,复习笔记领域博主](https://blog.csdn.net/qq_40514113?spm=1000.2115.3001.5343)
### ZhiHu(To update my thesis notes and to receive counseling)[(2 封私信 / 50 条消息) 出门吃三碗饭 - 知乎 (zhihu.com)](https://www.zhihu.com/people/olkex)
### 公众号(need wechat，and You can buy some wacky items)  AI知识物语https://mp.weixin.qq.com/s/SL5QGtB1svkG_ac11OrR0Q
