import torch
from ignite.handlers import create_lr_scheduler_with_warmup


def make_lr_scheduler(kind, optimizer, kwargs, warmup, config):
    lr = config['learning_rate']
    lr_min = config['learning_rate_min']
    epochs = config['num_epochs']
    if kind == 'constant':
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1., total_iters=1)
    elif kind == 'exponential':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif kind == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif kind == 'step':
        kwargs = dict(step_size=100, gamma=0.5) | kwargs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif kind == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
    elif kind == 'linear':
        func = lambda epoch: lr - epoch * (lr - lr_min) / epochs
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
    elif kind == 'plateau':
        kwargs = dict(mode='max', factor=0.5, patience=config['patience'] // 2) | kwargs
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown kind of lr scheduler: {kind}")

    if warmup > 0:
        lr_scheduler = create_lr_scheduler_with_warmup(lr_scheduler, lr_min, warmup)

    # plot the scheduler
    # import matplotlib.pyplot as plt
    # import numpy as np
    # lrs = []
    # for i in range(epochs):
    #     lr_scheduler.step()
    #     lrs.append(lr_scheduler.get_lr()[0])
    # plt.plot(np.arange(epochs), lrs)
    # plt.show()

    return lr_scheduler
