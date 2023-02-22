import torch

class MultiLoss(torch.nn.Module):
    def __init__(self, losses=[], weights=[]):
        super().__init__()
        self.losses = losses
        self.weights = weights

    def add_loss(self, loss, weight=1.0):
        self.losses.append(loss)
        self.weights.append(weight)

    def __len__(self):
        return len(self.losses)

    def forward(self, inputs, targets):
        loss = 0
        for criterion, weight in zip(self.losses, self.weights):
            loss += weight * criterion(inputs, targets)
        return loss / len(self.losses)

def make_criterion(losses: str):
    criterion = MultiLoss()
    loss_dict = {
        'mean_square_error': (torch.nn.MSELoss(), 1.0),
        'cross_entropy': (torch.nn.CrossEntropyLoss(), 1.0),
        'negative_log_likelihood': (torch.nn.NLLLoss(), 1.0),
        'binary_cross_entropy': (torch.nn.BCEWithLogitsLoss(), 1.0),
        'custom_mse': (LMSELoss(), 1.0),
        'charbonnier': (CharbonnierLoss(), 4.0),
    }
    for loss in losses:
        if loss in loss_dict:
            criterion.add_loss(*loss_dict[loss])
        else:
            raise ValueError(f"Unknown kind of criterion: {loss}")
    assert len(criterion) > 0, "Criterion is empty"
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
