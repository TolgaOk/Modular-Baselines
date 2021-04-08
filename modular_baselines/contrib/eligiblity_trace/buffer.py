from typing import Any, Callable, Dict, Iterable, Optional
import torch


class ETAdam(torch.optimOptimizer):

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        momentum_decay: float = 0.99,
        fisher_decay: float = 0.99,
        trace_decay: float = 1.0,
        weight_decay: float = 0,
        eps: float = 1e-8,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum_decay:
            raise ValueError("Invalid momentum value: {}".format(momentum_decay))
        if not 0.0 <= fisher_decay:
            raise ValueError("Invalid momentum value: {}".format(fisher_decay))
        if not 0.0 <= trace_decay:
            raise ValueError("Invalid momentum value: {}".format(trace_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum_decay=momentum_decay, fisher_decay=fisher_decay,
                        trace_decay=trace_decay, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("momentum", 0)
            group.setdefault("centered", False)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], None]] = None) -> Optional[torch.Tensor]:
        """Performs a single optimization step.
        :param closure: A closure that reevaluates the model
            and returns the loss.
        :return: loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("RMSpropTF does not support sparse gradients")
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # PyTorch initialized to zeros here
                    state["square_avg"] = torch.ones_like(p, memory_format=torch.preserve_format)
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)
                    if group["centered"]:
                        state["grad_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avg = state["square_avg"]
                alpha = group["alpha"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                if group["centered"]:
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    # PyTorch added epsilon after square root
                    # avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-
                                             1).add_(group["eps"]).sqrt_()
                else:
                    # PyTorch added epsilon after square root
                    # avg = square_avg.sqrt().add_(group['eps'])
                    avg = square_avg.add(group["eps"]).sqrt_()

                if group["momentum"] > 0:
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).addcdiv_(grad, avg)
                    p.add_(buf, alpha=-group["lr"])
                else:
                    p.addcdiv_(grad, avg, value=-group["lr"])

        return loss
