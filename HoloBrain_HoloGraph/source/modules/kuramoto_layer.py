import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
import numpy as np
import torch.nn.functional as F
from source.utils import ScaleAndBias, Attention, adj_connectivity

# class SyncModule(nn.Module):
#     """
#     Coupling / Synchronization.
#     - conv: GCNConv (edge_index)
#     - attn: Attention (no graph needed or uses x only)
#     - adj : dense coupling using W @ x   (paper-consistent)
#     """
#     def __init__(self, J, ch, heads=8, learn_wscale=False):
#         super().__init__()
#         self.type = J

#         if J == "conv":
#             self.net = GCNConv(ch, ch)
#         elif J == "attn":
#             self.net = Attention(ch, heads=heads, weight="fc")
#         elif J == "adj":
#             # optional learnable scalar; set learn_wscale=False for strict paper
#             self.w_scale = nn.Parameter(torch.tensor(1.0), requires_grad=learn_wscale)
#         else:
#             raise NotImplementedError(f"Unknown mapping_type/J: {J}")

#     def forward(self, x, graph_struct):
#         """
#         x: [B, N, C]
#         graph_struct:
#             - conv: edge_index [2, E]
#             - adj : W [N, N] or [B, N, N]
#         """
#         if self.type == "conv":
#             if x.dim() == 3 and x.size(0) == 1:
#                 return self.net(x.squeeze(0), graph_struct).unsqueeze(0)
#             raise NotImplementedError("Batch>1 not supported for conv yet.")

#         if self.type == "adj":
#             W = graph_struct
#             if W.dim() == 2:
#                 W = W.unsqueeze(0)              # [1, N, N]
#             # dense coupling: W @ x
#             return self.w_scale * torch.bmm(W, x)

#         # attn
#         return self.net(x)
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
    """
    Coupling / Synchronization.

    - conv: GCNConv (edge_index)
    - attn: Attention (uses x only)
    - adj : dense coupling using K @ x, where K = W ⊙ A 
            A is chosen as a low-rank symmetric learnable matrix: A = U U^T.

    Args (adj mode):
      - A_rank: rank r for U ∈ R^{N×r}; params ~ N*r (safe for large N)
      - A_act:  "sigmoid" (default) -> gate in (0,1), stable
               "softplus" -> positive gate
               "none"     -> no activation (raw A)
      - normalize_K: if True, row-normalize K to stabilize dynamics
      - eps: small constant for normalization
      - learn_wscale: optional scalar multiplier (default False = strict)
    """
    def __init__(
        self,
        J: str,
        ch: int,
        heads: int = 8,
        learn_wscale: bool = False,
        # --- adj specific ---
        use_A: bool = True,
        A_rank: int = 16,
        A_act: str = "sigmoid",
        normalize_K: bool = True,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.type = J

        if J == "conv":
            self.net = GCNConv(ch, ch)

        elif J == "attn":
            self.net = Attention(ch, heads=heads, weight="fc")

        elif J == "adj":
            # self.net = adj_connectivity(self.feature_dim)
            self.use_A = bool(use_A)
            self.A_rank = int(A_rank)
            self.A_act = str(A_act).lower()
            self.normalize_K = bool(normalize_K)
            self.eps = float(eps)

            self.w_scale = nn.Parameter(torch.tensor(1.0), requires_grad=learn_wscale)

            self.U = None

        else:
            raise NotImplementedError(f"Unknown mapping_type/J: {J}")

    # -------------------------
    # helpers for adj branch
    # -------------------------
    def _ensure_U(self, N: int, device, dtype):
        """
        Lazy-init U ∈ R^{N×r} once N is known.
        If N changes, re-init to match (rare; but safe).
        """
        if (self.U is None) or (self.U.shape[0] != N) or (self.U.shape[1] != self.A_rank):
            U = torch.randn(N, self.A_rank, device=device, dtype=dtype) * 0.01
            self.U = nn.Parameter(U, requires_grad=True)
        return self.U

    def _apply_A_activation(self, A: torch.Tensor) -> torch.Tensor:
        """
        A: [B,N,N] (or broadcastable)
        """
        if self.A_act == "sigmoid":
            return torch.sigmoid(A)
        if self.A_act == "softplus":
            return F.softplus(A)
        if self.A_act in ("none", "identity", ""):
            return A
        raise ValueError(f"Unknown A_act={self.A_act}")

    def _build_K(self, W: torch.Tensor) -> torch.Tensor:
        """
        W: [B,N,N] dense adjacency (non-negative preferred)
        Return K: [B,N,N] symmetric coupling
        """
        # 1) symmetrize
        W = 0.5 * (W + W.transpose(-1, -2))

        # (optional) ensure non-negative coupling if desired
        # W = W.clamp_min(0.)

        if not self.use_A:
            K = W
        else:
            B, N, _ = W.shape
            U = self._ensure_U(N, W.device, W.dtype)      # [N,r]
            A = U @ U.t()                                 # [N,N] symmetric PSD
            A = self._apply_A_activation(A)               # e.g., sigmoid -> (0,1)
            A = A.unsqueeze(0).expand(B, -1, -1)          # [B,N,N]
            K = W * A                                     # Hadamard

        # 2) symmetric normalization (keeps K symmetric)
        if self.normalize_K:
            deg = K.sum(dim=-1).clamp_min(self.eps)       # [B,N]
            D_inv_sqrt = torch.diag_embed(deg.rsqrt())    # [B,N,N]
            K = D_inv_sqrt @ K @ D_inv_sqrt               # symmetric norm

        return K


    # -------------------------
    # forward
    # -------------------------
    def forward(self, x: torch.Tensor, graph_struct):
        """
        x: [B,N,C]
        graph_struct:
          - conv: edge_index [2,E]
          - adj : W [N,N] or [B,N,N]
        """
        if self.type == "conv":
            if x.dim() == 3 and x.size(0) == 1:
                return self.net(x.squeeze(0), graph_struct).unsqueeze(0)
            raise NotImplementedError("Batch>1 not supported for conv yet.")

        if self.type == "adj":
            W = graph_struct
            if W.dim() == 2:
                W = W.unsqueeze(0)  # [1,N,N]
            elif W.dim() != 3:
                raise ValueError(f"adj expects W as [N,N] or [B,N,N], got {tuple(W.shape)}")

            K = self._build_K(W)                    # [B,N,N]
            return self.w_scale * torch.bmm(K, x)   # [B,N,C]

        # attn
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
        # self.sync_module = SyncModule(J, ch, heads=heads, learn_wscale=False)
        self.sync_module = SyncModule(
        J, ch,
        heads=heads,
        learn_wscale=False,  
        use_A=True,           
        A_rank=16,            
        A_act="sigmoid",      
        normalize_K=False,     
    )

        # self.sync_module = SyncModule(J, ch, heads, feature_dim)
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
        y = self.map_to_sphere(y) 
        if x.shape[1] == self.ch:
            x = x.transpose(1, 2).contiguous()  # [B,N,C]
        B = x.size(0)
        # graph struct
        if self.J_type == "conv":
            # accept either edge_index or dense W
            if sc.dim() == 2 and sc.size(0) == 2:
                graph_struct = sc.long()  # edge_index
            else:
                # dense -> edge_index
                W = sc.squeeze(0) if sc.dim() == 3 else sc
                graph_struct = torch.nonzero(W, as_tuple=False).T.long()
        elif self.J_type == "adj":
            # accept either edge_index or dense W
            if sc.dim() == 2 and sc.size(0) == 2:
                N = x.size(1) if x.dim() == 3 else None
                graph_struct = to_dense_adj(sc.long(), max_num_nodes=N).squeeze(0).float()  # [N,N]
            else:
                graph_struct = (sc.squeeze(0) if sc.dim() == 3 else sc).float()             # [N,N]
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
            coupling = self.sync_module(x, graph_struct)
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
