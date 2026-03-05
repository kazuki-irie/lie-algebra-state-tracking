import torch
import time

from mamba_ssm.modules.mamba2 import Mamba2

batch, length, dim = 2, 123, 128
x = torch.randn(batch, length, dim).to('cuda').half()

model = Mamba2(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim,  # Model dimension d_model
    headdim=4,  # Number of heads
    d_state=64,  # SSM state expansion factor, typically 64 or 128
    d_conv=4,  # Local convolution width
    expand=2,  # Block expansion factor
).to("cuda").half()

s1 = time.time()
y = model(x)
s2 = time.time()
print("Time taken: ", s2 - s1)

s1 = time.time()
y = model(x)
s2 = time.time()
print("Time taken: ", s2 - s1)

assert y.shape == x.shape
