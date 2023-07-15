from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.   包装的优化器
        multiplier: target learning rate = base lr * multiplier  # 目标学习率 = 基础学习率 * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually  # 在total_epoch时，渐进地达到目标学习率
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)  在目标epoch之后，使用这个调度器（例如ReduceLROnPlateau）
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier  # 目标学习率与基础学习率的比例
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')  # multiplier应大于或等于1
        self.total_epoch = total_epoch  # 达到目标学习率的总epoch数
        self.after_scheduler = after_scheduler  # 目标epoch之后使用的调度器
        self.finished = False  # 是否完成渐进学习率的阶段
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]  # 更新base_lrs到目标学习率
                    self.finished = True
                return self.after_scheduler.get_lr()  # 返回目标epoch之后的学习率
            return [base_lr * self.multiplier for base_lr in self.base_lrs]  # 返回目标学习率

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]  # 返回渐进学习率

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau在epoch结束时调用，而其他调度器在epoch开始时调用
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]  # 获取渐进学习率
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr  # 设置参数组的学习率为渐进学习率
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:  # 如果after_scheduler不是ReduceLROnPlateau类型的调度器
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)  # 调用父类的step方法
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)  # 调用step_ReduceLROnPlateau方法
