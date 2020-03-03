import typing
import torch
import numpy as np

class SimLoss(torch.nn.Module):
    def __init__(self,
                 w: typing.Optional[torch.Tensor],
                 number_of_classes: int = 10,
                 lower_bound: float = 0.5,
                 epsilon: float = 1e-8) -> None:
        super().__init__()

        assert lower_bound >= 0.0
        assert lower_bound < 1.0
        assert number_of_classes > 0

        self.w = w.float()
        self.epsilon = epsilon

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        w = self.w[y, :]
        return torch.mean(-torch.log(torch.sum(w * x, dim=1) + self.epsilon))

    def __repr__(self) -> str:
        return "SimCE"
