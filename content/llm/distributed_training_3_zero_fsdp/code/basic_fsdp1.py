import functools

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )

    def forward(self, x):
        return self.net(x)


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    setup(rank, world_size)

    model = MyModel().cuda()

    # Optional: auto-wrap large layers.
    # If a submodule has parameter count >= threshold, wrap it as its own FSDP unit.
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1_000_000
    )

    model = FSDP(model, auto_wrap_policy=auto_wrap_policy)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(10):
        x = torch.randn(32, 1024).cuda()
        y = model(x)
        loss = y.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"step {step}, loss {loss.item()}")

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"CUDA device count: {world_size}")

    if world_size == 0:
        raise RuntimeError("No CUDA devices found. This example requires CUDA and NCCL.")

    mp.spawn(train, args=(world_size,), nprocs=world_size)
