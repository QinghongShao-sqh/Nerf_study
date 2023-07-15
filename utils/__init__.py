# optimizer
from torch.optim import SGD, Adam
from .optimizers import *
# scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from .warmup_scheduler import GradualWarmupScheduler

from .visualization import *

def get_optimizer(hparams, models):
    eps = 1e-8
    parameters = []
    for model in models:
        parameters += list(model.parameters())  # 将模型的参数添加到参数列表中
    if hparams.optimizer == 'sgd':  # 如果选择SGD优化器
        optimizer = SGD(parameters, lr=hparams.lr,  # 使用学习率、动量和权重衰减进行实例化
                        momentum=hparams.momentum, weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'adam':  # 如果选择Adam优化器
        optimizer = Adam(parameters, lr=hparams.lr, eps=eps,
                         weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'radam':  # 如果选择RAdam优化器
        optimizer = RAdam(parameters, lr=hparams.lr, eps=eps,
                          weight_decay=hparams.weight_decay)
    elif hparams.optimizer == 'ranger':  # 如果选择Ranger优化器
        optimizer = Ranger(parameters, lr=hparams.lr, eps=eps,
                          weight_decay=hparams.weight_decay)
    else:
        raise ValueError('optimizer not recognized!')

    return optimizer

def get_scheduler(hparams, optimizer):
    eps = 1e-8
    if hparams.lr_scheduler == 'steplr':  # 如果选择StepLR学习率调度器
        scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step,
                                gamma=hparams.decay_gamma)  # 使用里程碑和衰减因子进行实例化
    elif hparams.lr_scheduler == 'cosine':  # 如果选择Cosine学习率调度器
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=eps)  # 使用最大迭代次数和最小学习率进行实例化
    elif hparams.lr_scheduler == 'poly':  # 如果选择Poly学习率调度器
        scheduler = LambdaLR(optimizer,
                             lambda epoch: (1-epoch/hparams.num_epochs)**hparams.poly_exp)  # 使用多项式衰减函数进行实例化
    else:
        raise ValueError('scheduler not recognized!')

    if hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']:  # 如果需要热身并且优化器不是RAdam或Ranger
        scheduler = GradualWarmupScheduler(optimizer, multiplier=hparams.warmup_multiplier,
                                           total_epoch=hparams.warmup_epochs, after_scheduler=scheduler)  # 使用渐进热身调度器进行实例化

    return scheduler

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']  # 返回当前优化器的学习率

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))  # 加载模型的检查点文件
    checkpoint_ = {}
    if 'state_dict' in checkpoint:  # 如果是pytorch-lightning的检查点文件
        checkpoint = checkpoint['state_dict']
    for k, v in checkpoint.items():
        if not k.startswith(model_name):  # 如果键不是以模型名称开头，则跳过
            continue
        k = k[len(model_name)+1:]  # 去掉模型名称的前缀
        for prefix in prefixes_to_ignore:
            if k.startswith(prefix):  # 如果键以指定的前缀开头，则跳过
                print('ignore', k)
                break
        else:
            checkpoint_[k] = v  # 将键和对应的值添加到checkpoint_字典中
    return checkpoint_

def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    model_dict = model.state_dict()  # 获取模型的状态字典
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)  # 提取检查点文件中的模型状态字典
    model_dict.update(checkpoint_)  # 更新模型的状态字典
    model.load_state_dict(model_dict)  # 加载模型的状态字典
