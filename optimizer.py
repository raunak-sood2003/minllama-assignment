from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer
import math

class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                
                # State should be stored in this dictionary
                if p not in self.state:
                    self.state[p] = {}
                    self.state[p]["step"] = 0
                    self.state[p]["m_t"] = torch.zeros_like(p.data)
                    self.state[p]["v_t"] = torch.zeros_like(p.data)
                
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group['betas'][0], group['betas'][1]
                eps = group['eps']
                weight_decay = group['weight_decay']
                

                # Get values to keep track of
                t = state['step']
                m_t = state['m_t']
                v_t = state['v_t']

                # Update first and second moments of the gradients
                t += 1
                m_t = beta1 * m_t + (1 - beta1) * grad
                v_t = beta2 * v_t + (1 - beta2) * (grad ** 2)

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                alpha_t = alpha * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data = p.data - alpha_t * m_t / (torch.sqrt(v_t) + eps) - alpha * weight_decay * p.data

                # Update parameters
                state['step'] = t
                state['m_t'] = m_t
                state['v_t'] = v_t
                self.state[p] = state

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.

        return loss