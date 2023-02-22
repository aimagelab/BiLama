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
#
#
# class EMA:
#     def __init__(self, model, ema_parameters):
#         super(EMA, self).__init__()
#         self.model = model
#         self.ema_parameters = ema_parameters
#
#     def __enter__(self, ):
#         self.model_state_dict = copy.deepcopy(self.model.state_dict())
#         self.model.load_state_dict(params_to_model_state_dict(self.ema_parameters, self.model))
#
#         return self.model
#
#     def __exit__(self, exc_type, exc_value, exc_tb):
#         self.model.load_state_dict(self.model_state_dict)
#         return self.model
