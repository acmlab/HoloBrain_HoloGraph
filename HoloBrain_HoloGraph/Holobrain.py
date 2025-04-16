import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy.linalg import expm
import math
import sys
# ---- Load your custom CFC utilities ----
# Make sure these are defined in the same file or imported from a module
def thresholding(fc, data_path, ratio=0.8):
    node_num = fc.shape[-1]
    fc[fc < 0] = 0
    fc_tril = np.tril(fc, -1)
    K = np.count_nonzero(fc_tril)
    KT = math.ceil(ratio * ((node_num**2 - node_num) / 2))
    if KT >= K:
        thr = 0
    else:
        thr = np.partition(fc_tril.reshape(-1), -KT)[-KT]
    fc[fc < thr] = 0
    if not np.all(np.sum(fc > 0, -1) > 1):
        warnings.warn(f'Thresholding may be too large: {data_path}')
    return fc

def harmonic_wavelets(graph, wavelets_num=10, beta=1, gamma=0.005, max_iter=1000, min_err=0.0001, node_select=10):
    node_num = graph.shape[-1]
    temp_D = np.eye(node_num) * np.sum(graph, axis=-1, keepdims=True)
    latentlaplacian = temp_D - graph
    _, temp_phi = np.linalg.eigh(latentlaplacian)
    u_vec = np.zeros_like(graph)
    phi_k = np.expand_dims(temp_phi[..., :wavelets_num], -3)
    it, err = 0, np.inf
    diag_idx = np.arange(node_num)
    while err > min_err and it < max_iter:
        np.put_along_axis(u_vec, np.argpartition(graph, -node_select)[..., -node_select:], 1, -1)
        u_vec[..., diag_idx, diag_idx] = 1
        temp_v = 1 - u_vec
        temp_v = np.eye(temp_v.shape[-1]) * np.expand_dims(temp_v, -2)
        Theta = beta * temp_v
        temp_increment = 2 * (np.eye(node_num) - phi_k @ phi_k.swapaxes(-2, -1))
        phi_increment = -gamma * temp_increment @ Theta @ phi_k
        Q, R = np.linalg.qr((np.eye(node_num) - phi_k @ phi_k.swapaxes(-2, -1)) @ phi_increment)
        A = phi_k.swapaxes(-2, -1) @ phi_increment
        temp_matrix1 = np.concatenate([A, -R.swapaxes(-2, -1)], -1)
        temp_matrix2 = np.concatenate([R, np.zeros_like(R)], -1)
        temp_matrix3 = np.concatenate([temp_matrix1, temp_matrix2], -2)
        BC = expm(temp_matrix3)[..., :wavelets_num]
        phi_k = phi_k @ BC[..., :wavelets_num, :] + Q @ BC[..., wavelets_num:, :]
        err = np.max(np.linalg.norm(phi_increment, 'fro', (-2, -1)))
        it += 1
    return phi_k

# def cfc(wavelets, bold, wavelets_num=10):
#     powers = np.sum(
#         np.expand_dims(wavelets[..., :wavelets_num].swapaxes(-2, -1), -3)
#         * np.expand_dims(np.expand_dims(bold.swapaxes(-2, -1), -2), -4),
#         -1,
#     )  # shape: (num_nodes, timepoints, wavelets_num)

#     # flatten time × freq to one feature vector per node
#     num_nodes = powers.shape[0]
#     flat_powers = powers.reshape(num_nodes, -1)  # shape: (nodes, timepoints * wavelets_num)

#     cfcs = np.corrcoef(flat_powers)  # shape: (nodes, nodes)
#     return cfcs, powers
def cfc(wavelets, bold, wavelets_num=10):
    # Ensure BOLD signal shape is (nodes, timepoints)
    if bold.shape[0] != wavelets.shape[1]:
        bold = bold.T

    # Step 1: Compute powers using wavelets
    powers = np.sum(
        np.expand_dims(wavelets[..., :wavelets_num].swapaxes(-2, -1), -3)
        * np.expand_dims(np.expand_dims(bold.swapaxes(-2, -1), -2), -4),
        -1,
    )  # shape: (nodes, timepoints, wavelets_num)

    # Step 2: Move wavelet freq to first axis for correlation (f, t)
    powers = powers.transpose(0, 2, 1)  # now shape: (nodes, wavelets_num, timepoints)

    num_nodes = powers.shape[0]
    cfc_matrix = np.zeros((num_nodes, wavelets_num, wavelets_num))

    # Step 3: Compute freq-freq correlation per node
    for i in range(num_nodes):
        cfc_matrix[i] = np.corrcoef(powers[i])

    return cfc_matrix, powers

# def cfc(wavelets, bold, wavelets_num=10):
#     powers = np.sum(
#         np.expand_dims(wavelets[..., :wavelets_num].swapaxes(-2, -1), -3)
#         * np.expand_dims(np.expand_dims(bold.swapaxes(-2, -1), -2), -4),
#         -1,
#     )
#     cfcs = np.corrcoef(powers.swapaxes(-2, -1))
#     return cfcs, powers

# ---- Main ----
data_x = np.load(sys.argv[1])
# print(data_x.shape)  # (samples, conditions, nodes, timepoints)
# output_dir = "NIFD4"
output_dir = sys.argv[2]
os.makedirs(output_dir, exist_ok=True)
# output_dir2 = "NIFD4/NIFD4_CFC"
output_dir2 = sys.argv[2]
os.makedirs(os.path.join(output_dir2, "whole"), exist_ok=True)

num_samples, num_conditions, num_nodes, num_timepoints = data_x.shape
avg_cfc_matrices = np.zeros((num_conditions, 10, 10))

for cond in range(num_conditions):
    cfc_matrices = np.zeros((num_samples,10, 10))
    all_cfc = []
    for i in range(num_samples):
        bold_signal = data_x[i, cond]  # shape: (nodes, timepoints)
       #  print(bold_signal.shape)
        fc = np.corrcoef(bold_signal)
       #  print(fc.shape)
        graph = thresholding(fc, data_path=f"sample_{i}_cond_{cond}")
        wavelets = harmonic_wavelets(graph)
       #  print(wavelets.shape)
        cfc_matrix, _ = cfc(wavelets, bold_signal)
        mean_cfc = np.mean(cfc_matrix, axis=0)  # shape: (10, 10)
       #  print(cfc_matrix.shape)
        cfc_matrices[i] = mean_cfc
        txt_filename = os.path.join(output_dir2, f"cfc_matrix_condition_{cond+1}_sample_{i+1}.txt")
       #  np.savetxt(txt_filename, mean_cfc, fmt="%.6f")
       #  print(f"Saved CFC to {txt_filename}")
        all_cfc.append(mean_cfc)
        npy_filename = os.path.join(output_dir2, "whole", f"cfc_matrix_condition_{cond+1}_sample_{i+1}_full.npy")
        np.save(npy_filename, cfc_matrix)
        print(f"💾 Saved full CFC matrix: {npy_filename}")
    
#     avg_cfc_matrices[cond] = np.mean(cfc_matrices, axis=0)

    # 🔹 Visualization
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(avg_cfc_matrices[cond], cmap="coolwarm", center=0, square=True, cbar=True)
#     plt.title(f"Condition {cond+1} CFC Matrix")
#     png_filename = os.path.join(output_dir, f"cfc_matrix_condition_{cond+1}.png")
#     plt.savefig(png_filename, dpi=300)
#     plt.close()
#     print(f"Saved PNG: {png_filename}")

#     # 🔹 Thresholding: Keep top 60 connections
#     matrix = avg_cfc_matrices[cond].copy()
#     np.fill_diagonal(matrix, 0)
#     thresholded_matrix = np.zeros_like(matrix)
#     flat_indices = np.argsort(matrix, axis=None)[-60:]
#     row_indices, col_indices = np.unravel_index(flat_indices, matrix.shape)
#     for r, c in zip(row_indices, col_indices):
#         thresholded_matrix[r, c] = matrix[r, c]
#         thresholded_matrix[c, r] = matrix[c, r]

#     txt_filename = os.path.join(output_dir2, f"thresholded_cfc_matrix_condition_{cond+1}.edge")
#     np.savetxt(txt_filename, thresholded_matrix, fmt="%.6f")
#     print(f"Saved TXT: {txt_filename}")
