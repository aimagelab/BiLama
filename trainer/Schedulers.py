import torch


def make_lr_scheduler(kind: str, optimizer: torch.optim.Optimizer):
    if kind == 'constant':
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=1)
    elif kind == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    else:
        raise ValueError(f"Unknown kind of lr scheduler: {kind}")
    return lr_scheduler
