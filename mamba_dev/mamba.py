import torch
import time

from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

# Set up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up the configuration
config = MambaConfig(
    d_model=128,
    n_layer=2,
    vocab_size=10000,
    ssm_cfg={'d_state': 64, 'd_conv': 4, 'expand': 2, 'positive_and_negative_associative_scan': False}
)

# Create the model
model = MambaLMHeadModel(config).to(device).half()

# Generate example data
batch_size = 2
sequence_length = 100
input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_length), device=device)

# Forward pass test
def forward_pass_test():
    model.eval()
    with torch.no_grad():
        s1 = time.time()
        outputs = model(input_ids)
        s2 = time.time()
    print(f"Time taken for forward pass: {s2 - s1:.4f} seconds")
    return outputs

# Backward pass test
def backward_pass_test():
    model.train()
    s1 = time.time()
    outputs = model(input_ids)
    loss = outputs.logits.mean()
    loss.backward()
    s2 = time.time()
    print(f"Time taken for backward pass: {s2 - s1:.4f} seconds")

# Run the tests
print("Running forward pass test...")
outputs = forward_pass_test()
print(f"Output shape: {outputs.logits.shape}")

print("\nRunning forward pass test again...")
outputs = forward_pass_test()

print("\nRunning backward pass test...")
backward_pass_test()

# Cleanup
torch.cuda.empty_cache()