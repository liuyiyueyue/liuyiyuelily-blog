import torch
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device("cuda")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False
) as prof:

    with record_function("GPU_WORK"):
        # Large tensors -> longer-running kernels
        x = torch.randn(50_000_000, device=device)
        y = torch.randn(50_000_000, device=device)
        z = torch.randn(50_000_000, device=device)

prof.export_chrome_trace("2_gpu_trace.json")
