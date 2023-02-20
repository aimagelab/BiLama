import copy
import torch
from torch import nn


def params_to_model_state_dict(params, model):
    state_dict = model.state_dict()
    for i, (name, _) in enumerate(model.named_parameters()):
        state_dict[name] = params[i]
    return state_dict


def model_state_dict_to_params(state_dict, model):
    params = [state_dict[name] for name, _ in model.named_parameters()]
    return params


# class EMA(nn.Module):
#     def __init__(self, model, beta=0.9999, update_every=10, update_after_step=100, inv_gamma=1.0,
#                  power=2 / 3, min_value=0.0):
#         super().__init__()
#         self.ema_params = copy.deepcopy(list(model.parameters()))
#
#         self.parameter_names = {name for name, param in self.ema_model.named_parameters() if param.dtype == torch.float}
#         self.buffer_names = {name for name, buffer in self.ema_model.named_buffers() if buffer.dtype == torch.float}
#
#         self.beta = beta
#         self.update_every = update_every
#         self.update_after_step = update_after_step
#         self.inv_gamma = inv_gamma
#         self.power = power
#         self.min_value = min_value
#
#         self.register_buffer('initted', torch.Tensor([False]))
#         self.register_buffer('step', torch.tensor([0]))
#
#     def update_moving_average(self, source_params):
#         for target, source in zip(self.ema_params, source_params):
#             target.detach().mul_(self.beta).add_(source.detach(), alpha=1 - self.beta)
#
#     def __call__(self, model):
#         ema_model = model.clone()
#         # model_parameters = copy.deepcopy(model.state_dict())
#         ema_model.load_state_dict(params_to_model_state_dict(self.ema_params, model))
#         model.load_state_dict(model_parameters)
