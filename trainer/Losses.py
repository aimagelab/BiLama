import torch


def make_criterion(kind: str):
    if kind == 'mean_square_error':
        criterion = torch.nn.MSELoss()
    elif kind == 'cross_entropy':
        criterion = torch.nn.CrossEntropyLoss()
    elif kind == 'negative_log_likelihood':
        criterion = torch.nn.NLLLoss()
    elif kind == 'binary_cross_entropy':
        criterion = torch.nn.BCEWithLogitsLoss()
    elif kind == 'custom_mse':
        criterion = LMSELoss()
    elif kind == 'charbonnier':
        criterion = CharbonnierLoss()
    else:
        raise ValueError(f"Unknown kind of criterion: {kind}")
    return criterion


class LMSELoss(torch.nn.MSELoss):
    def forward(self, inputs, targets):
        mse = super().forward(inputs, targets)
        mse = torch.add(mse, 1e-10)
        return torch.log10(mse)


def get_outnorm(x:torch.Tensor, out_norm:str='') -> torch.Tensor:
    """ Common function to get a loss normalization value. Can
        normalize by either the batch size ('b'), the number of
        channels ('c'), the image size ('i') or combinations
        ('bi', 'bci', etc)
    """
    # b, c, h, w = x.size()
    img_shape = x.shape

    if not out_norm:
        return 1

    norm = 1
    if 'b' in out_norm:
        # normalize by batch size
        # norm /= b
        norm /= img_shape[0]
    if 'c' in out_norm:
        # normalize by the number of channels
        # norm /= c
        norm /= img_shape[-3]
    if 'i' in out_norm:
        # normalize by image/map size
        # norm /= h*w
        norm /= img_shape[-1]*img_shape[-2]

    return norm

class CharbonnierLoss(torch.nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6, out_norm:str='bci'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.out_norm = out_norm

    def forward(self, x, y):
        norm = get_outnorm(x, self.out_norm)
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss*norm
