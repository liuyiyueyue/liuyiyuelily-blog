import torch
import time
from torch.profiler import profile, ProfilerActivity

device = torch.device("cuda")

# Warmup
x = torch.randn(10, device=device)
torch.cuda.synchronize()

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:

    torch.cuda.synchronize()
    t0 = time.time()

    x = torch.randn(10_000_000, device=device)
    for _ in range(100):
        y = x + 1
    print("CPU finishes")

    torch.cuda.synchronize()
    t1 = time.time()
    print("GPU finishes. Time:", t1 - t0)

prof.export_chrome_trace("3_synchronize_trace.json")

