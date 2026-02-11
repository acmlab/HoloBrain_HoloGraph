import os
import numpy as np
import json
import pandas as pd
from datetime import date, datetime
import torch
from torch.nn import Module


class Wavelet(Module):
    def __init__(self, wavelet=[0, 1, 2], level=2):
        super(Wavelet, self).__init__()
        self.wavelet = wavelet
        self.level = level

    def construct_wavelet(self, adj):
        adj = self.no_zero_adj(adj)
        wavelets = []
        degree = torch.sum(adj, dim=-1)
        if (degree == 0).any():
            raise ValueError("The adj has isolated nodes (degree=0).")
        D = torch.diag_embed(degree)
        D = D.to(adj.device)
        adj = D - adj
        D_inverse = torch.inverse(D)
        D_inverse[D_inverse == float("inf")] = 0.0
        I_n = torch.eye(adj.size(-1)).unsqueeze(0).repeat(adj.size(0), 1, 1).float()
        I_n = I_n.to(adj.device)
        adj = 0.5 * (I_n + torch.bmm(adj, D_inverse))
        adj_sct = adj.float()
        adj_power = adj_sct.clone()
        for order in self.wavelet:
            order = int(order)
            if order == 0:
                wavelets.append(I_n - adj_sct)
                continue
            if order > 1:
                adj_power = torch.bmm(adj_power, adj_power)
            # S^n(S^n-I)
            adj_int = torch.bmm(adj_power, I_n - adj_power)
            wavelets.append(adj_int)

        low_pass = torch.bmm(adj_power, adj_power)  # t^(2^j)
        low_pass = torch.bmm(low_pass, low_pass)  # t^(2^(j+1))
        return wavelets, low_pass

    def no_zero_adj(self, adj):
        batch_size, n, _ = adj.shape
        adj_fixed_batch = adj.clone()
        
        for b in range(batch_size):
            adj = adj_fixed_batch[b]
            zero_row_mask = (adj.sum(dim=-1) == 0)
            zero_row_indices = torch.where(zero_row_mask)[0]
            if len(zero_row_indices) == 0:
                continue

            for zero_row in zero_row_indices:
                prev_row = zero_row - 1
                while prev_row >= 0 and zero_row_mask[prev_row]:
                    prev_row -= 1
                
                if prev_row >= 0: 
                    adj[zero_row, :] = adj[prev_row, :]
                else:  
                    next_row = zero_row + 1
                    while next_row < n and zero_row_mask[next_row]:
                        next_row += 1
                    if next_row < n:
                        adj[zero_row, :] = adj[next_row, :]

            adj_symmetric = (adj + adj.T) / 2
            adj_fixed_batch[b] = adj_symmetric

        return adj_fixed_batch

    def windowed(self, x, adj):
        # x: B x N x Q
        # y: B x N x Q x dim
        wavelets, low_pass = self.construct_wavelet(adj)
        outputs = [[x.transpose(1, 2)]]
        for layer in range(self.level):
            layer_output = []
            for input in outputs[-1]:
                for wavelet in wavelets:
                    out = torch.matmul(wavelet, input)
                    out = torch.abs(out)
                    layer_output.append(out)
            outputs.append(layer_output)

        basis = torch.cat([torch.stack(layer, dim=-1) for layer in outputs], dim=-1)

        basis_shape = basis.shape
        basis = basis.view(basis.shape[0], basis.shape[1], -1)
        scattering_coeff = torch.matmul(low_pass, basis)
        scattering_coeff = scattering_coeff.view(basis_shape)
        # B x N x Q x dim
        return scattering_coeff

    def nonwindowed(self, x, adj):
        wavelets, low_pass = self.construct_wavelet(adj)
        outputs = [[x.transpose(1, 2)]]
        for layer in range(self.level):
            layer_output = []
            for input in outputs[-1]:
                for wavelet in wavelets:
                    out = torch.matmul(wavelet, input)
                    out = torch.abs(out)
                    layer_output.append(out)
            outputs.append(layer_output)

        # B x N x Q x dim
        basis = torch.cat([torch.stack(layer, dim=-1) for layer in outputs], dim=-1)
        Q = 2
        gst = []
        for q in range(1, Q + 1):
            q_st = basis**q
            q_st = torch.mean(q_st, dim=1)
            gst.append(q_st)
        gst = torch.stack(gst, dim=-1)  
        return basis

    def forward(self, x, adj, windowed=True):
        if windowed:
            return self.windowed(x, adj)
        return self.nonwindowed(x, adj)

