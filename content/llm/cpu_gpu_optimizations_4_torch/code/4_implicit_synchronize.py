import torch
from torch.profiler import profile, ProfilerActivity
from torch.autograd.profiler import record_function

device = torch.device("cuda")

# Make workload large enough to show clear GPU time
N = 10_000_000
x = torch.randn(N, device=device)

torch.cuda.synchronize()  # warmup

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    record_shapes=False,
    profile_memory=False,
    with_stack=False
) as prof:

    with record_function("ASYNC_COMPUTE"):
        z = x
        for _ in range(20):   # build up async queue
            z = z * 1.000001

    # ---- implicit synchronization ----
    with record_function("IMPLICIT_SYNC"):
        idx = torch.nonzero(z)

prof.export_chrome_trace("4_implicit_sync_trace.json")
