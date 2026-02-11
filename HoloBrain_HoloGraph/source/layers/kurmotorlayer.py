import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
from source.layers.common_layers_node import ScaleAndBias, Attention, adj_connectivity
import numpy as np

class OmegaLayerOptimized(nn.Module):
    """
    Handles the intrinsic frequency (omega) rotation dynamics.
    """
    def __init__(self, n, ch, init_omg=0.1, global_omg=False, learn_omg=True):
        super().__init__()
        self.n = n
        self.ch = ch
        self.global_omg = global_omg

        if n % 2 != 0:
            raise NotImplementedError("n must be even for OmegaLayer (pairwise oscillators).")

        # Pre-calculate parameter shape to avoid overhead in forward pass
        # ch // 2 because we process (x, y) pairs.
        shape = (1, 1) if global_omg else (ch // 2, 1)
        
        # Initialize omega parameters
        self.omg_param = nn.Parameter(
            init_omg * (1 / np.sqrt(2)) * torch.ones(shape), 
            requires_grad=learn_omg
        )

    def forward(self, x):
        """
        Applies rotation dynamics: dx/dt = omega * (-y, x)
        Input: x [B, N, C]
        """
        B, N, C = x.shape
        
        # 1. View as coordinate pairs: [B, N, Groups, 2]
        # The last dimension '2' represents the (x, y) coordinates.
        x_pairs = x.view(B, N, C // 2, 2)
        
        # 2. Get omega magnitude: [1, 1, Groups, 1] or [1, 1, 1, 1]
        # Automatic broadcasting handles the expansion.
        omg = torch.norm(self.omg_param, dim=-1, keepdim=True) 
        if not self.global_omg:
            omg = omg.view(1, 1, C // 2, 1)

        # 3. Apply Rotation: (-y, x) * omega
        # x_pairs[..., 0] is x-coord, x_pairs[..., 1] is y-coord
        # result 0:  omg * y
        # result 1: -omg * x
        out = torch.empty_like(x_pairs)
        out[..., 0] =  omg * x_pairs[..., 1]
        out[..., 1] = -omg * x_pairs[..., 0]
        
        # 4. Flatten back to original shape: [B, N, C]
        return out.view(B, N, C)


class KMLayer(nn.Module):
    """
    Kuramoto Layer (KMLayer).
    
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
        assert (ch % n) == 0, "Channel dimension must be divisible by n (oscillator grouping)."
        self.n = n
        self.ch = ch
        self.J = J
        self.use_omega = use_omega

        # --- Intrinsic Frequency (Omega) ---
        self.omg = (
            OmegaLayerOptimized(n, ch, init_omg, global_omg, learn_omg)
            if self.use_omega
            else None # Explicitly set to None for faster checking
        )

        # --- Coupling / Connectivity Function (J) ---
        if J == "conv":
            self.connectivity = GCNConv(ch, ch)
        elif J == "attn":
            self.connectivity = Attention(ch, heads=heads, weight="fc")
        elif J == "adj":
            self.connectivity = adj_connectivity(feature_dim)
        else:
            raise NotImplementedError(f"Unknown connectivity type: {J}")

        # --- Normalization ---
        if c_norm == "gn":
            self.c_norm = nn.GroupNorm(ch // n, ch, affine=True)
        elif c_norm == "sandb":
            self.c_norm = ScaleAndBias(ch, token_input=False)
        else:
            self.c_norm = nn.Identity()

    def normalize(self, x):
        """
        In-place renormalization to project oscillators back to the manifold (unit sphere).
        Assumes x is [B, N, C].
        Splits C into groups of size n and normalizes each group.
        """
        B, N, C = x.shape
        # View: [B, N, Groups, n_dim]
        x_view = x.view(B, N, C // self.n, self.n)
        
        # Calculate L2 norm along the last dimension
        norm = torch.norm(x_view, p=2, dim=-1, keepdim=True)
        
        # Normalize (add epsilon for numerical stability)
        x_view = x_view / (norm + 1e-6)
        
        return x_view.reshape(B, N, C)

    def project_fast(self, y, x):
        """
        Projection into the tangent space.
        Formula: y_proj = y - <x, y> * x
        
        Assumes x is already normalized (|x|=1).
        """
        B, N, C = x.shape
        
        # 1. View as groups: [B, N, Groups, n_dim]
        x_v = x.view(B, N, C // self.n, self.n)
        y_v = y.view(B, N, C // self.n, self.n)

        # 2. Dot product (Similarity) along the n_dim axis
        # sim: [B, N, Groups, 1]
        sim = torch.sum(x_v * y_v, dim=-1, keepdim=True)

        # 3. Subtract the parallel component
        # out_v: [B, N, Groups, n_dim]
        out_v = y_v - sim * x_v
        
        return out_v.reshape(B, N, C)

    def kupdate(self, x, c, graph_struct):
        """
        Calculates the derivative dx/dt based on the Kuramoto model.
        """
        # 1. Coupling Term (Connectivity)
        # Note: graph_struct is pre-computed outside the loop for efficiency
        if self.J == "conv":
            # x: [B, N, C] -> GCN needs [N, C] (assuming B=1) or stacked batch.
            # Here we squeeze B=1 for efficiency.
            if x.dim() == 3 and x.size(0) == 1:
                _y = self.connectivity(x.squeeze(0), graph_struct).unsqueeze(0)
            else:
                # Fallback for B>1 (requires reshaped batch handling)
                # For this specific project, B=1 is standard.
                raise NotImplementedError("Optimization assumes Batch=1 for GCNConv")
                
        elif self.J == "adj":
            # 'adj' connectivity typically requires dense matrix multiplication
            A_weighted = graph_struct * self.connectivity.weight
            
            # Update internal weights if the module supports it (for visualization/losses)
            if hasattr(self.connectivity, 'update_latest_A_weighted'):
                self.connectivity.update_latest_A_weighted(A_weighted)
                
            _y = self.connectivity(x, graph_struct)
        else:
            # Attention or other methods handling x directly
            _y = self.connectivity(x)

        # 2. Add Control/Bias (Attending Memory)
        y = _y + c

        # 3. Omega Term (Intrinsic Rotation)
        if self.omg is not None:
            omg_x = self.omg(x)
        else:
            omg_x = 0 # Scalar 0 allows broadcasting addition (faster than torch.zeros_like)

        # 4. Projection (Manifold Constraint)
        # Project the update 'y' onto the tangent space of the oscillator manifold
        y_proj = self.project_fast(y, x)

        # 5. Final Derivative
        dxdt = omg_x + y_proj
        return dxdt

    def forward(self, x: torch.Tensor, c: torch.Tensor, sc, Q: int, gamma):
        """
        ODE Solver Loop (Forward Euler).
        
        Args:
            x: Oscillator state [B, C, N] (will be transposed internally)
            y: Control pattern [B, C, N]
            sc: Structural connectivity (dense adj or edge_index)
            Q: Number of time steps
            gamma: Step size / coupling strength
        """
        # --- Pre-computation / Pre-processing ---
        
        # 1. Normalize shapes to [B, N, C] for efficient linear algebra
        y = self.c_norm(c)
        if y.shape[1] == self.ch: y = y.transpose(1, 2)
        if x.shape[1] == self.ch: x = x.transpose(1, 2)
        
        # 2. Prepare Graph Structure ONCE (Avoids CPU-GPU sync inside loop)
        graph_struct = None
        if self.J == "conv":
            # For GCN: Dense Adj -> Sparse Edge Index
            # Doing this once saves massive time compared to doing it inside kupdate
            graph_struct = torch.nonzero(sc.squeeze(), as_tuple=False).T
        elif self.J == "adj":
            # For Dense layers: Ensure Dense Adj
            graph_struct = to_dense_adj(sc).squeeze(0) if sc.dim() == 2 else sc.squeeze(0)
        
        # 3. Initial Normalization
        x = self.normalize(x)
        
        xs = []
        
        # --- ODE Solver Loop ---
        for _ in range(Q):
            # Calculate derivative
            dxdt = self.kupdate(x, y, graph_struct)
            
            # Euler Integration Step
            x = x + gamma * dxdt
            
            # Renormalize to stay on the manifold
            x = self.normalize(x)
            
            xs.append(x)

        # Return trajectory list. 
        # (Transposing back to [B, C, N] is usually handled by the parent class if needed)
        return xs, [], []