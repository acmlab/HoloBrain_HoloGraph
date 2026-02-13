import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
import numpy as np

from source.modules.common_layers_node import ScaleAndBias, Attention, adj_connectivity


class OmegaModule(nn.Module):
    def __init__(self, n, ch, init_omg=0.1, global_omg=False, learn_omg=True):
        super().__init__()
        self.n = n
        self.ch = ch
        self.global_omg = global_omg
        if n % 2 != 0:
            raise NotImplementedError("n must be even for OmegaModule (pairwise oscillators).")

        shape = (1, 1) if global_omg else (ch // 2, 1)
        self.omega_param = nn.Parameter(
            init_omg * (1 / np.sqrt(2)) * torch.ones(shape),
            requires_grad=learn_omg
        )

    def forward(self, x):  # x: [B,N,C]
        B, N, C = x.shape
        x_pairs = x.view(B, N, C // 2, 2)
        omg = torch.norm(self.omega_param, dim=-1, keepdim=True)
        if not self.global_omg:
            omg = omg.view(1, 1, C // 2, 1)
        out = torch.empty_like(x_pairs)
        out[..., 0] =  omg * x_pairs[..., 1]
        out[..., 1] = -omg * x_pairs[..., 0]
        return out.view(B, N, C)


class SyncModule(nn.Module):
    def __init__(self, J, ch, heads=8, feature_dim=116):
        super().__init__()
        self.type = J
        if J == "conv":
            self.net = GCNConv(ch, ch)
        elif J == "attn":
            self.net = Attention(ch, heads=heads, weight="fc")
        elif J == "adj":
            self.net = adj_connectivity(feature_dim)
        else:
            raise NotImplementedError(f"Unknown mapping_type/J: {J}")

    def forward(self, x, graph_struct):
        if self.type == "conv":
            if x.dim() == 3 and x.size(0) == 1:
                return self.net(x.squeeze(0), graph_struct).unsqueeze(0)
            raise NotImplementedError("Batch>1 not supported for conv yet.")
        elif self.type == "adj":
            if hasattr(self.net, "update_latest_A_weighted"):
                self.net.update_latest_A_weighted(graph_struct * self.net.weight)
            return self.net(x, graph_struct)
        else:
            return self.net(x)


class Kuramoto_Solver(nn.Module):
    def __init__(
        self,
        n,
        ch,
        J="conv",
        c_norm="gn",
        use_omega=False,
        init_omg=1.0,
        global_omg=False,
        heads=8,
        learn_omg=True,
        feature_dim=116,
        eps=1e-6,
        return_energy=False,
    ):
        super().__init__()
        assert (ch % n) == 0
        self.n = n
        self.ch = ch
        self.J_type = J
        self.eps = eps
        self.omega_module = OmegaModule(n, ch, init_omg, global_omg, learn_omg) if use_omega else None
        self.sync_module = SyncModule(J, ch, heads, feature_dim)
        self.return_energy = return_energy
        if c_norm == "gn":
            self.norm_y = nn.GroupNorm(ch // n, ch, affine=True)
        elif c_norm == "sandb":
            self.norm_y = ScaleAndBias(ch, token_input=False)
        else:
            self.norm_y = nn.Identity()

    def map_to_sphere(self, x):
        B, N, C = x.shape
        x_view = x.view(B, N, C // self.n, self.n)
        norm2 = (x_view * x_view).sum(dim=-1, keepdim=True).clamp_min(self.eps)
        x_view = x_view * torch.rsqrt(norm2)
        return x_view.view(B, N, C)

    def project_osc(self, vector, x):
        B, N, C = x.shape
        x_v = x.view(B, N, C // self.n, self.n)
        v_v = vector.view(B, N, C // self.n, self.n)
        sim = torch.sum(x_v * v_v, dim=-1, keepdim=True)
        out = v_v - sim * x_v
        return out.view(B, N, C)
    def _energy(self, x, y, W_dense=None):
        """
        A lightweight energy proxy
        """
        if W_dense is None:
            return None
        # x,y: [B,N,C]
        # similarity: [B,N,N] via inner product
        sim = torch.einsum("bnc,bmc->bnm", x, x)
        # coupling term
        e1 = -(sim * W_dense).sum(dim=(1, 2))
        # control alignment term
        e2 = -(x * y).sum(dim=(1, 2))
        return e1 + e2
    def forward(self, x, y, sc, Q, gamma, rho=1.0, return_es=None):
        # y norm expects [B,C,N]
        if y.shape[1] != self.ch:
            y = y.transpose(1, 2).contiguous()
        y = self.norm_y(y)
        y = y.transpose(1, 2).contiguous()  # [B,N,C]

        if x.shape[1] == self.ch:
            x = x.transpose(1, 2).contiguous()  # [B,N,C]

        # graph struct
        if self.J_type == "conv":
            graph_struct = torch.nonzero(sc.squeeze(), as_tuple=False).T
        elif self.J_type == "adj":
            graph_struct = to_dense_adj(sc).squeeze(0) if sc.dim() == 2 else sc.squeeze(0)
        else:
            graph_struct = None

        x = self.map_to_sphere(x)
        xs, es = [], []
        # for energy: if adj & dense available
        W_dense = None
        if (return_es is True) or (return_es is None and self.return_energy):
            if self.J_type == "adj":
                # graph_struct may be [N,N] or [B,N,N]
                if graph_struct is not None:
                    if graph_struct.dim() == 2:
                        W_dense = graph_struct.unsqueeze(0).expand(B, -1, -1)
                    elif graph_struct.dim() == 3:
                        W_dense = graph_struct
        for _ in range(Q):
            # coupling + control
            coupling = self.sync_module(x, graph_struct) if self.J_type != "attn" else self.sync_module(x, graph_struct)
            force = coupling + y
            # omega
            omega_term = self.omega_module(x) if self.omega_module else 0.0
            # tangent projection
            force_proj = self.project_osc(force, x)
            # dxdt
            dxdt = omega_term + rho * force_proj
            # Euler update + project back to sphere
            x = self.map_to_sphere(x + gamma * dxdt)
            xs.append(x)
            # optional energy
            if W_dense is not None:
                E = self._energy(x, y, W_dense=W_dense)
                es.append(E)

        return xs, es, []
