import torch


def make_lr_scheduler(kind: str, optimizer: torch.optim.Optimizer, kwargs: dict):
    if kind == 'constant':
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=1)
    elif kind == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif kind == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,80,90,95,100], **kwargs)
    elif kind == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif kind == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown kind of lr scheduler: {kind}")
    return lr_scheduler
