""" Torch optimizer that works as eligibility trace. These traces can be seen as momentum term
but one per environment. Hence, the optimizer requires the environment size at initialization.
"""

from typing import Any, Callable, Dict, Iterable, Optional
import torch

# Eligibility Trace Optimizer
class ETAdam(torch.optimOptimizer):

    def __init__(self,
                 params: Iterable[torch.nn.Parameter],
                 learning_rate: float = 1e-3,
                 fisher_decay: float = 0.99,
                 trace_decay: float = 1.0,
                 weight_decay: float = 0,
                 n_envs: int = 1,
                 eps: float = 1e-8):

        if learning_rate < 0.0:
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if fisher_decay < 0.0:
            raise ValueError("Invalid momentum value: {}".format(fisher_decay))
        if trace_decay < 0.0:
            raise ValueError("Invalid momentum value: {}".format(trace_decay))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(learning_rate=learning_rate, fisher_decay=fisher_decay,
                        trace_decay=trace_decay, eps=eps, weight_decay=weight_decay,
                        n_envs=n_envs)
        super().__init__(params, defaults)

    # def __setstate__(self, state: Dict[str, Any]) -> None:
    #     super().__setstate__(state)
    #     for group in self.param_groups:
    #         group.setdefault("momentum", 0)
    #         group.setdefault("centered", False)

    @torch.no_grad()
    def step(self,
             terminations: torch.tensor,
             closure: Optional[Callable[[], None]] = None) -> Optional[torch.Tensor]:
        """Performs a single optimization step.
        :param closure: A closure that reevaluates the model
            and returns the loss.
        :param terminations: 1D tensor of terminations to reset the enviornment traces.
        :return: loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("RMSpropTF does not support sparse gradients")
                state = self.state[param]

                # Lazy State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["diag_fisher"] = torch.ones_like(
                        param, memory_format=torch.preserve_format)
                    # Per environment momentum term a.k.a the trace tensor
                    state["trace_buffer"] = torch.zeros(
                        shape=(self.n_envs, *param.shape),
                        dtype=param.dtype,
                        memory_format=torch.preserve_format)

                diag_fisher = state["diag_fisher"]
                env_traces = state["trace_buffer"]
                fisher_decay = group["fisher_decay"]
                trace_decay = group["trace_decay"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(param, alpha=group["weight_decay"])

                diag_fisher.mul_(fisher_decay).addcmul_(grad, grad, value=(1 - fisher_decay))
                env_traces.mul_(trace_decay).add_(grad.unsqueeze(0))

                # TF like eps and bias correction term
                sqrt_diag_disher = diag_fisher.add(group["eps"]).sqrt_().div_(
                    1 - fisher_decay**state["step"])

                param.addcdiv_(env_traces.mean(0), sqrt_diag_disher, value=-group["lr"])

                # Reset the trace tensor with the termination vector
                env_traces.mul_(1 - terminations.reshape(-1, *[1]*len(param.shape)))

        return loss

# TODO: Need batch gradient layers