import torch
from torch.profiler import profile, ProfilerActivity, record_function

device = torch.device("cuda")

batch_size = 10_000_000
num_steps = 5

compute_stream = torch.cuda.Stream()
transfer_stream = torch.cuda.Stream()

# Pinned memory (required for overlap)
host_buffers = [
    torch.randn(batch_size, pin_memory=True),
    torch.randn(batch_size, pin_memory=True),
]

device_buffers = [
    torch.empty(batch_size, device=device),
    torch.empty(batch_size, device=device),
]

torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=False,
) as prof:

    for step in range(num_steps):

        buf_id = step % 2
        next_buf = (step + 1) % 2

        # Compute on current buffer
        if step > 0:
            with torch.cuda.stream(compute_stream):
                with record_function("COMPUTE"):
                    y = device_buffers[buf_id] * 2.0

        # Transfer next batch
        if step < num_steps - 1:
            with torch.cuda.stream(transfer_stream):
                with record_function("COPY"):
                    device_buffers[next_buf].copy_(
                        host_buffers[next_buf],
                        non_blocking=True
                    )

        compute_stream.wait_stream(transfer_stream)

torch.cuda.synchronize()

prof.export_chrome_trace("5_double_buffer_trace.json")
