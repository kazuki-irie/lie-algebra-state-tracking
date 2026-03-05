import numpy as np
import torch
import matplotlib.pyplot as plt

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


def compare_specific_entry(tensor1, tensor2, indices, name):
    grad1 = tensor1.grad[indices]
    grad2 = tensor2.grad[indices]

    # Collect gradients into lists
    grad1_list = []
    grad2_list = []

    num_elements = min(32, tensor1.grad.numel())  # Ensure we don't exceed the number of elements
    for idx in range(num_elements):
        grad1_list.append(tensor1.grad.flatten()[idx].item())
        grad2_list.append(tensor2.grad.flatten()[idx].item())

    grad1_array = np.array(grad1_list)
    grad2_array = np.array(grad2_list)

    # Perform least squares fit: grad2 = scale * grad1 + shift
    A_matrix = np.vstack([grad1_array, np.ones(len(grad1_array))]).T
    scale, shift = np.linalg.lstsq(A_matrix, grad2_array, rcond=None)[0]

    # Calculate the predicted values and the error (residuals)
    predicted_grad2 = scale * grad1_array + shift
    residuals = grad2_array - predicted_grad2
    mean_squared_error = np.mean(residuals ** 2)
    # Calculate residuals and R-squared
    residuals = grad2_array - predicted_grad2
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((grad2_array - np.mean(grad2_array)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return scale, shift, r_squared


def test_selective_scan(is_variable_B=True, is_variable_C=True, varBC_groups=1, has_D=True, has_z=True,
                        has_delta_bias=True, delta_softplus=True, return_last_state=True, seqlen=4, itype=torch.float32,
                        wtype=torch.float32):
    device = 'cuda'
    rtol, atol = (6e-4, 2e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 3e-2, 5e-2
    rtolw, atolw = (1e-3, 1e-3)
    if has_z:
        rtolw = max(rtolw, rtol)
        atolw = max(atolw, atol)
    torch.random.manual_seed(0)
    batch_size = 2
    dim = 4
    dstate = 64
    is_complex = wtype == torch.complex64

    A = (-0.5 * (torch.rand(dim, dstate, device=device, dtype=wtype))).requires_grad_()

    if not is_variable_B:
        B_shape = (dim, dstate)
    elif varBC_groups == 1:
        B_shape = (batch_size, dstate, seqlen if not is_complex else seqlen * 2)
    else:
        B_shape = (batch_size, varBC_groups, dstate, seqlen if not is_complex else seqlen * 2)
    B = torch.randn(*B_shape, device=device, dtype=wtype if not is_variable_B else itype, requires_grad=True)

    C = torch.randn(B_shape, device=device, dtype=wtype if not is_variable_C else itype, requires_grad=True)

    if has_D:
        D = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)
    else:
        D = None

    if has_z:
        z = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
    else:
        z = None

    if has_delta_bias:
        delta_bias = (0.5 * torch.rand(dim, device=device, dtype=torch.float32)).requires_grad_()
    else:
        delta_bias = None

    u = torch.randn(batch_size, dim, seqlen, device=device, dtype=itype, requires_grad=True)
    delta = (0.5 * torch.rand(batch_size, dim, seqlen, device=device, dtype=itype)).requires_grad_()

    A_ref = A.detach().clone().requires_grad_()
    B_ref = B.detach().clone().requires_grad_()
    C_ref = C.detach().clone().requires_grad_()

    out, *rest = selective_scan_fn(u, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus,
                                   return_last_state=return_last_state)
    out_ref, *rest_ref = selective_scan_ref(u.detach(), delta.detach(), A_ref, B_ref, C_ref, D, z=z,
                                            delta_bias=delta_bias, delta_softplus=delta_softplus,
                                            return_last_state=return_last_state)

    g = torch.randn_like(out)
    out.backward(g)
    out_ref.backward(g)

    specific_indices = (0, 0)
    scale, shift, mse = compare_specific_entry(A, A_ref, specific_indices, 'A')

    return scale, shift, mse


if __name__ == "__main__":
    sequence_lengths = [50, 100, 256, 512, 1024, 2048, 3000, 4096, 8000]
    scales = []
    shifts = []
    errors = []

    for seqlen in sequence_lengths:
        print(f'\nSEQUENCE LENGTH: {seqlen}')
        scale, shift, mse = test_selective_scan(seqlen=seqlen)
        scales.append(scale)
        shifts.append(shift)
        errors.append(mse)

    # Plot the scale as a function of sequence length
    plt.figure()
    plt.plot(sequence_lengths, scales, label='Scale (slope)', marker='o')
    plt.xlabel('Sequence Length')
    plt.ylabel('Scale')
    plt.title('Scale vs Sequence Length')
    plt.grid(True)
    plt.savefig('scale_vs_sequence_length.png')
    plt.show()

    # Plot the shift as a function of sequence length
    plt.figure()
    plt.plot(sequence_lengths, shifts, label='Shift (intercept)', marker='o')
    plt.xlabel('Sequence Length')
    plt.ylabel('Shift')
    plt.title('Shift vs Sequence Length')
    plt.grid(True)
    plt.savefig('shift_vs_sequence_length.png')
    plt.show()

    # Plot the error (mean squared error) as a function of sequence length
    plt.figure()
    plt.plot(sequence_lengths, errors, label='Mean Squared Error', marker='o', color='red')
    plt.xlabel('Sequence Length')
    plt.ylabel('Error (MSE)')
    plt.title('Error (MSE) vs Sequence Length')
    plt.grid(True)
    plt.savefig('error_vs_sequence_length.png')
    plt.show()
