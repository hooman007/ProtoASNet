# The code below has been taken from the following GitHub repository:
# https://github.com/lindehesse/INSightR-Net

import torch
from torchmetrics import Metric


class SparsityMetric(Metric):
    def __init__(self, dist_sync_on_step=False, level=0.9, device="cuda"):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.level = level
        self.add_state("percentage_expl", default=torch.tensor(0).to(device), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0).to(device), dist_reduce_fx="sum")

    def update(self, prototype_activations: torch.Tensor):
        # Normalize by dividing by sum
        proto_norm = prototype_activations / torch.sum(prototype_activations, dim=1).unsqueeze(-1)

        # sort and compute cumulative sum
        sorted, indices = torch.sort(proto_norm, descending=True, dim=1)
        cumsum = torch.cumsum(sorted, dim=1)
        num_weights = torch.ge(cumsum, self.level).type(torch.uint8).argmax(dim=1)

        # Gather results
        self.percentage_expl += torch.sum(num_weights)
        self.total += num_weights.numel()

    def update_and_compute(self, prototype_activations: torch.Tensor):
        # Normalize by dividing by sum
        proto_norm = prototype_activations / torch.sum(prototype_activations, dim=1).unsqueeze(-1)

        # sort and compute cumulative sum
        sorted, indices = torch.sort(proto_norm, descending=True, dim=1)
        cumsum = torch.cumsum(sorted, dim=1)
        num_weights = torch.ge(cumsum, self.level).type(torch.uint8).argmax(dim=1)

        # Gather results
        self.percentage_expl += torch.sum(num_weights)
        self.total += num_weights.numel()

        return torch.sum(num_weights).float() / num_weights.numel()

    def compute(self):
        return self.percentage_expl.float() / self.total
