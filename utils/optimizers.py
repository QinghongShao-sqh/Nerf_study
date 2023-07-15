# optimizer
from torch.optim import SGD, Adam
from .optimizers import *
# scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from .warmup_scheduler import GradualWarmupScheduler

from .visualization import *

def get_optimizer(hparams, models):
    eps = 1e-8  # 设置一个很小的数，用于防止除以0的错误
    parameters = []  # 存储所有模型的参数
    for model in models:
        parameters += list(model.parameters())  # 将每个模型的参数添加到parameters中
    if hparams.optimizer == 'sgd':  # 如果优化器是sgd
        optimizer = SGD(parameters, lr=hparams.lr,
                        momentum=hparams.momentum, weight_decay=hparams.weight_decay)  # 使用SGD优化器
    elif hparams.optimizer == 'adam':  # 如果优化器是adam
        optimizer = Adam(parameters, lr=hparams.lr, eps=eps,
                         weight_decay=hparams.weight_decay)  # 使用Adam优化器
    elif hparams.optimizer == 'radam':  # 如果优化器是radam
        optimizer = RAdam(parameters, lr=hparams.lr, eps=eps,
                          weight_decay=hparams.weight_decay)  # 使用RAdam优化器
    elif hparams.optimizer == 'ranger':  # 如果优化器是ranger
        optimizer = Ranger(parameters, lr=hparams.lr, eps=eps,
                          weight_decay=hparams.weight_decay)  # 使用Ranger优化器
    else:
        raise ValueError('optimizer not recognized!')  # 如果优化器不在预设范围内，抛出错误

    return optimizer  # 返回优化器

def get_scheduler(hparams, optimizer):
    eps = 1e-8  # 设置一个很小的数，用于防止除以0的错误
    if hparams.lr_scheduler == 'steplr':  # 如果学习率调度器是steplr
        scheduler = MultiStepLR(optimizer, milestones=hparams.decay_step,
                                gamma=hparams.decay_gamma)  # 使用MultiStepLR调度器
    elif hparams.lr_scheduler == 'cosine':  # 如果学习率调度器是cosine
        scheduler = CosineAnnealingLR(optimizer, T_max=hparams.num_epochs, eta_min=eps)  # 使用CosineAnnealingLR调度器
    elif hparams.lr_scheduler == 'poly':  # 如果学习率调度器是poly
        scheduler = LambdaLR(optimizer,
                             lambda epoch: (1-epoch/hparams.num_epochs)**hparams.poly_exp)  # 使用LambdaLR调度器
    else:
        raise ValueError('scheduler not recognized!')  # 如果学习率调度器不在预设范围内，抛出错误

    if hparams.warmup_epochs > 0 and hparams.optimizer not in ['radam', 'ranger']:  # 如果有预热周期且优化器不是radam或ranger
        scheduler = GradualWarmupScheduler(optimizer, multiplier=hparams.warmup_multiplier,
                                           total_epoch=hparams.warmup_epochs, after_scheduler=scheduler)  # 使用GradualWarmupScheduler调度器

    return scheduler  # 返回学习率调度器

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:  # 遍历优化器的参数组
        return param_group['lr']  # 返回学习率

def extract_model_state_dict(ckpt_path, model_name='model', prefixes_to_ignore=[]):
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))  # 加载模型的检查点
    checkpoint_ = {}  # 存储模型的状态字典
    if 'state_dict' in checkpoint:  # 如果检查点中有'state_dict'
        checkpoint = checkpoint['state_dict']  # 获取'state_dict'
    for k, v in checkpoint.items():  # 遍历检查点中的键值对
        if not k.startswith(model_name):  # 如果键不是以指定的模型名开头，跳过
            continue
        k = k[len(model_name)+1:]  # 去除模型名和一个点的部分
        for prefix in prefixes_to_ignore:  # 遍历要忽略的前缀
            if k.startswith(prefix):  # 如果键以要忽略的前缀开头
                print('ignore', k)  # 打印忽略的键
                break
        else:
            checkpoint_[k] = v  # 将键值对添加到模型的状态字典中
    return checkpoint_  # 返回模型的状态字典

def load_ckpt(model, ckpt_path, model_name='model', prefixes_to_ignore=[]):
    model_dict = model.state_dict()  # 获取模型的状态字典
    checkpoint_ = extract_model_state_dict(ckpt_path, model_name, prefixes_to_ignore)  # 提取模型的状态字典
    model_dict.update(checkpoint_)  # 更新模型的状态字典
    model.load_state_dict(model_dict)  # 加载模型的状态字典
