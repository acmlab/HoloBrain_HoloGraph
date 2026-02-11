import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
from source.layers.common_layers_node import ScaleAndBias, Attention, adj_connectivity
import numpy as np

class OmegaModule(nn.Module):
    """
    [Optimized] Handles the intrinsic frequency (omega) rotation dynamics.
    Vectorized implementation avoiding Python loops.
    """
    def __init__(self, n, ch, init_omg=0.1, global_omg=False, learn_omg=True):
        super().__init__()
        self.n = n
        self.ch = ch
        self.global_omg = global_omg

        if n % 2 != 0:
            raise NotImplementedError("n must be even for OmegaModule (pairwise oscillators).")

        # Parameter shape: (1, 1) or (Groups, 1)
        shape = (1, 1) if global_omg else (ch // 2, 1)
        
        self.omega_param = nn.Parameter(
            init_omg * (1 / np.sqrt(2)) * torch.ones(shape), 
            requires_grad=learn_omg
        )

    def forward(self, x):
        """
        Input: x [B, N, C]
        Returns: omega * (-y, x)
        """
        B, N, C = x.shape
        
        # 1. View as coordinate pairs: [B, N, Groups, 2]
        x_pairs = x.view(B, N, C // 2, 2)
        
        # 2. Get omega magnitude
        omg = torch.norm(self.omega_param, dim=-1, keepdim=True) 
        if not self.global_omg:
            omg = omg.view(1, 1, C // 2, 1)

        # 3. Apply Rotation: (-y, x) * omega
        out = torch.empty_like(x_pairs)
        out[..., 0] =  omg * x_pairs[..., 1]
        out[..., 1] = -omg * x_pairs[..., 0]
        
        return out.view(B, N, C)


class SyncModule(nn.Module):
    """
    [Optimized] Handles the Coupling / Synchronization logic.
    Wraps GCN, Attention, or Dense Adjacency mechanisms.
    """
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
            # GCNConv optimization for B=1
            if x.dim() == 3 and x.size(0) == 1:
                return self.net(x.squeeze(0), graph_struct).unsqueeze(0)
            else:
                raise NotImplementedError("Batch>1 optimization pending for GCNConv")
        elif self.type == "adj":
            # Dense Adjacency
            if hasattr(self.net, 'update_latest_A_weighted'):
                self.net.update_latest_A_weighted(graph_struct * self.net.weight)
            return self.net(x, graph_struct)
        else:
            # Attention
            return self.net(x)


class Kuramoto_Solver(nn.Module):
    """
    [Optimized] Kuramoto Solver / KLayer.
    Solves: dx/dt = proj( Omega(x) + Sync(x) + Control(y) )
    """
    def __init__(
        self,
        n,
        ch,
        J="conv",
        c_norm="gn",
        use_omega=False,
        init_omg=1.0,
        ksize=3,
        global_omg=False,
        heads=8,
        learn_omg=True,
        feature_dim=116,
    ):
        super().__init__()
        assert (ch % n) == 0, "Channel dimension must be divisible by n."
        self.n = n
        self.ch = ch
        self.J_type = J
        
        # 1. Omega Module
        self.omega_module = OmegaModule(n, ch, init_omg, global_omg, learn_omg) if use_omega else None

        # 2. Sync Module (Coupling)
        self.sync_module = SyncModule(J, ch, heads, feature_dim)

        # 3. Control Normalization (f_phi equivalent)
        if c_norm == "gn":
            self.norm_y = nn.GroupNorm(ch // n, ch, affine=True)
        elif c_norm == "sandb":
            self.norm_y = ScaleAndBias(ch, token_input=False)
        else:
            self.norm_y = nn.Identity()

    def map_to_sphere(self, x):
        """Project x back to the unit sphere manifold (Renormalization)."""
        B, N, C = x.shape
        x_view = x.view(B, N, C // self.n, self.n)
        norm = torch.norm(x_view, p=2, dim=-1, keepdim=True)
        x_view = x_view / (norm + 1e-6)
        return x_view.reshape(B, N, C)

    def project_osc(self, vector, x):
        """Project update vector onto the tangent space of x."""
        B, N, C = x.shape
        x_v = x.view(B, N, C // self.n, self.n)
        vec_v = vector.view(B, N, C // self.n, self.n)
        
        # Dot product <vector, x>
        sim = torch.sum(x_v * vec_v, dim=-1, keepdim=True)
        
        # Subtract radial component: v - <v,x>x
        out_v = vec_v - sim * x_v
        return out_v.reshape(B, N, C)

    def surrounding_osc(self, x, y, graph_struct):
        """Calculates external driving forces: Sync + Memory(y)"""
        # 1. Sync / Coupling
        coupling = self.sync_module(x, graph_struct)
        # 2. Add Attending Memory (Control y)
        return coupling + y

    def update_osc(self, x, y, graph_struct):
        """Calculates dxdt"""
        # 1. Get driving force (Coupling + Control)
        force = self.surrounding_osc(x, y, graph_struct)

        # 2. Intrinsic rotation
        omega_term = self.omega_module(x) if self.omega_module else 0

        # 3. Tangent Projection (Constraint)
        force_proj = self.project_osc(force, x)

        # 4. Final derivative
        dxdt = omega_term + force_proj
        return dxdt

    def forward(self, x, y, sc, Q, gamma):
        """
        x: Oscillator state [B, C, N] (input) -> [B, N, C] (internal)
        y: Control state    [B, C, N] (input) -> [B, N, C] (internal)
        sc: Graph structure
        Q: Steps
        gamma: Step size
        """
        # --- Pre-processing ---
        # Normalize y (Control)
        y = self.norm_y(y)
        
        # Transpose to [B, N, C] for efficiency
        if y.shape[1] == self.ch: y = y.transpose(1, 2)
        if x.shape[1] == self.ch: x = x.transpose(1, 2)

        # Prepare Graph Structure (Once, outside loop)
        graph_struct = None
        if self.J_type == "conv":
            # Dense -> Sparse Edge Index
            graph_struct = torch.nonzero(sc.squeeze(), as_tuple=False).T
        elif self.J_type == "adj":
            # Ensure Dense
            graph_struct = to_dense_adj(sc).squeeze(0) if sc.dim() == 2 else sc.squeeze(0)
        
        # Initial Map to Sphere
        x = self.map_to_sphere(x)
        xs = []

        # --- Dynamics Loop ---
        for _ in range(Q):
            dxdt = self.update_osc(x, y, graph_struct)
            x = x + gamma * dxdt
            x = self.map_to_sphere(x)
            xs.append(x)

        # Return trajectory and dummy placeholders
        return xs, [], []
