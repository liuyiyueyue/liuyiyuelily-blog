import torch
from torch.profiler import profile, record_function, ProfilerActivity
import time

device = torch.device("cpu")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False
) as prof:

    with record_function("CPU_WORK"):
        # 大 tensor → 长 kernel
        x = torch.randn(50_000_000, device=device)
        y = torch.randn(50_000_000, device=device)
        z = torch.randn(50_000_000, device=device)

prof.export_chrome_trace("1_cpu_trace.json")

