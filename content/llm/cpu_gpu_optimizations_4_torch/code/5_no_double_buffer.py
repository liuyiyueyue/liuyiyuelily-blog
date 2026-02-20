import torch
from torch.profiler import profile, ProfilerActivity, record_function

device = torch.device("cuda")

batch_size = 10_000_000
num_steps = 5

# Pageable host memory (NOT pinned)
host_buffer = torch.randn(batch_size)
device_buffer = torch.empty(batch_size, device=device)

torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:

    for step in range(num_steps):

        with record_function("COPY"):
            device_buffer.copy_(host_buffer)  # blocking copy

        with record_function("COMPUTE"):
            y = device_buffer * 2.0

        torch.cuda.synchronize()  # force strict serialization

prof.export_chrome_trace("5_no_double_buffer_trace.json")
print("Non-overlap trace exported.")
