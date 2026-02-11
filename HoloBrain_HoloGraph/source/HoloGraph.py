import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj

from source.layers.kurmotolayer import Kuramoto_Solver
from source.layers.common_layers_node import ReadOutConv, MultiConv1D
from source.layers.GST import Wavelet


class HoloGraph(nn.Module):
    """
    HoloGraph: Physics-Informed Graph Neural Network with Kuramoto Dynamics.

    This architecture integrates Geometric Scattering Transforms (GST) with coupled oscillator dynamics 
    to capture both multi-scale structural features and long-range dependencies.

    1.  **Attending Memory / Control Pattern ($y$)**:
        -   Encodes the static graph topology and node features to guide the dynamics.
        -   *Homo*: Uses GCN. *Non-Homo*: Uses Multi-Scale Convolution (MultiConv1D).
    2.  **Oscillator Initialization ($x_0$) - Eq. (4)**:
        -   $x(0) = \text{GST}(X, A)$.
        -   Utilizes the Geometric Scattering Transform to initialize oscillators with 
            multi-scale frequency information.
    3.  **Kuramoto Dynamics - Eq. (6)**:
        -   Evolves the state $x(t)$ over discrete time steps $Q$.
        -   Update rule: $x(t+1) = x(t) + \Delta t \cdot [\omega + \text{Coupling}(x, A) + \text{Control}(y)]$.
    """

    def __init__(
        self,
        n=4,
        ch=256,
        L=1,                # Number of Kuramoto dynamic blocks
        Q=8,             # Legacy: Time steps 
        num_class=4,
        feature_dim=None,  
        J="attn",
        ksize=1,
        c_norm="gn",
        gamma=1.0,
        use_omega=False,
        init_omg=1.0,
        global_omg=False,
        heads=8,
        dropout=0.5,
        use_residual=True,
        learn_omg=False,
        homo=False,

        # --- GST Hyperparameters (Eq. 4) ---
        gst_level=2,
        gst_wavelet=(0, 1, 2),

        # --- Architecture Switches ---
        init_x_from="gst",   # "gst" (Strict Paper Logic) or "y" (Relaxed/Stability)
        fuse_y_into_x=False, # "Plan A": Soft Initialization x(0) = x_gst + alpha * y
        pred_from="y",       # Prediction source: "y" (Control) or "x" (Final State)
        gst_total=4,         # Width multiplier for MultiConv1D (used when homo=False)
        probe_nodes=16,      # Dummy node count for probing GST dimensions (Numel-safe)
    ):
        super().__init__()

        if feature_dim is None:
            raise ValueError("feature_dim is required. Please pass feature_dim=data.x.size(-1).")

        self.n = n
        self.ch = ch
        self.L = int(L)
        # Determine time steps Q
        steps = Q 
        self.steps = [int(steps)] * self.L if isinstance(steps, int) else [int(s) for s in steps]
        self.J = [J] * self.L if isinstance(J, str) else list(J)
        self.gamma = gamma
        self.num_class = int(num_class)
        self.homo = bool(homo)
        self.use_residual = bool(use_residual)
        self.dropout = nn.Dropout(p=float(dropout))

        # For GST, we interpret the Feature Dimension (F) as the Temporal Signal (Q)
        self.input_dim = int(feature_dim) 
        
        self.init_x_from = init_x_from
        self.fuse_y_into_x = bool(fuse_y_into_x)
        self.pred_from = pred_from
        self.gst_total = int(gst_total)

        # ---------------------------------------------------------
        # 1. Control Encoder -> Generates 'y' (Attending Memory)
        # ---------------------------------------------------------
        if self.homo:
            # Homophilic Graphs (e.g., Cora): Use Standard GCN
            self.encoder_y_1 = GCNConv(self.input_dim, ch)
            self.encoder_y_2 = GCNConv(ch, ch)
            self.norm_y_1 = nn.LayerNorm(ch)
            self.norm_y_2 = nn.LayerNorm(ch)
        else:
            # Non-Homophilic/Brain Networks: Use Multi-Scale 1D Convolution
            # Expands features by gst_total factors before projection
            self.encoder_y = MultiConv1D(
                in_channels=self.input_dim,
                out_channels=self.input_dim,
                num_convs=self.gst_total,
            )
            self.proj_y = nn.Conv1d(self.input_dim * self.gst_total, ch, kernel_size=1)

        # ---------------------------------------------------------
        # 2. Geometric Scattering Transform (GST) -> Generates x(0)
        # ---------------------------------------------------------
        self.gst = Wavelet(wavelet=list(gst_wavelet), level=int(gst_level))

        with torch.no_grad():
            Np = max(int(probe_nodes), 2)
            # Create dummy input [1, Q, N] where Q=input_dim
            x_probe = torch.zeros(1, self.input_dim, Np) 
            adj_probe = torch.eye(Np).unsqueeze(0)       
            
            out = self.gst(x_probe, adj_probe)           # Expected: [1, N, Q, dim]
            
            if out.dim() != 4:
                raise RuntimeError(f"GST output mismatch. Expected [B, N, T, dim], got {tuple(out.shape)}")
            
            # Flatten channels: Q * dim
            gst_channels = int(out.size(-2) * out.size(-1)) 

        self.gst_out_channels = gst_channels
        self.patchfy_x = nn.Conv1d(self.gst_out_channels, ch, kernel_size=1)

        # Optional: Learnable fusion gate for "Plan A" initialization
        if self.fuse_y_into_x:
            self.alpha = nn.Parameter(torch.tensor(0.0))

        # ---------------------------------------------------------
        # 3. Dynamics Blocks (Eq. 6)
        # ---------------------------------------------------------
        self.layers = nn.ModuleList()
        for l in range(self.L):
            klayer = Kuramoto_Solver(
                n=n,
                ch=ch,
                J=self.J[l],
                c_norm=c_norm,
                use_omega=use_omega,
                init_omg=init_omg,
                global_omg=global_omg,
                heads=heads,
                learn_omg=learn_omg,
                ksize=ksize,
                feature_dim=self.input_dim,
            )
            readout = ReadOutConv(ch, ch, self.n, 1, 1, 0, self.homo)
            self.layers.append(nn.ModuleList([klayer, readout]))

        # ---------------------------------------------------------
        # 4. Prediction Head
        # ---------------------------------------------------------
        self.out_pred_node = nn.Linear(ch, self.num_class)

    # ============================================================
    # Helper Functions: Shape Normalization
    # ============================================================
    def _infer_N_from_x(self, x):
        """Robustly infers the number of nodes N from input tensor."""
        if x.dim() == 2: return x.size(0)
        if x.dim() == 3 and x.size(0) == 1:
            # Case [1, F, N]: if dim 1 matches feature_dim, N is dim 2
            if x.size(1) == self.input_dim: return x.size(2)
            # Case [1, N, F]: if dim 2 matches feature_dim, N is dim 1
            return x.size(1)
        raise ValueError(f"Unexpected x shape: {tuple(x.shape)}")

    def _to_x_gcn(self, x, edge_index):
        """
        Normalizes input for GCN layers.
        Target Shape: [N, F]
        """
        N = int(edge_index.max().item()) + 1
        if x.dim() == 2:
            if x.size(0) != N or x.size(1) != self.input_dim:
                raise ValueError(f"Shape mismatch: x={tuple(x.shape)}, expected N={N}, F={self.input_dim}")
            return x.contiguous(), N

        if x.dim() == 3 and x.size(0) == 1:
            # [1, N, F]
            if x.size(1) == N and x.size(2) == self.input_dim:
                return x.squeeze(0).contiguous(), N
            # [1, F, N] -> Transpose to [N, F]
            if x.size(1) == self.input_dim and x.size(2) == N:
                return x.squeeze(0).t().contiguous(), N

        raise ValueError(f"Cannot infer GCN input orientation: x={tuple(x.shape)}, N={N}")

    def _to_BTN(self, x, N=None):
        """
        Normalizes input for GST and MultiConv1D.
        Target Shape: [B, Q, N] where B=1, Q=Feature_Dim.
        """
        if x.dim() == 2:
            # [N, F] -> [1, F, N] (equivalent to [1, Q, N])
            if N is not None and x.size(0) != N:
                raise ValueError(f"x has N={x.size(0)} but expected N={N}")
            return x.t().unsqueeze(0).contiguous()

        if x.dim() == 3 and x.size(0) == 1:
            # [1, Q, N]
            if x.size(1) == self.input_dim:
                return x.contiguous()
            # [1, N, Q] -> Transpose to [1, Q, N]
            if x.size(2) == self.input_dim:
                return x.transpose(1, 2).contiguous()

        raise ValueError(f"Cannot convert to [1, Q, N]: x={tuple(x.shape)}, T={self.input_dim}")

    def _ensure_dense_adj(self, sc, N=None):
        """Converts edge_index or sparse adjacency to dense [B, N, N]."""
        if sc.dim() == 2 and sc.size(0) == 2: # Edge Index
            if N is None: N = int(sc.max().item()) + 1
            return to_dense_adj(sc.long(), max_num_nodes=N)
        if sc.dim() == 2: # [N, N]
            return sc.unsqueeze(0).contiguous()
        if sc.dim() == 3: # [B, N, N]
            return sc.contiguous()
        raise ValueError(f"Unexpected adjacency shape: {tuple(sc.shape)}")

    def _flatten_gst(self, gst_out):
        """Flattens GST output [B, N, Q, dim] -> [B, N, Q*dim]."""
        if gst_out.dim() != 4:
            raise ValueError(f"GST output must be 4D [B,N,Q,dim], got {tuple(gst_out.shape)}")
        B, N, Q, dim = gst_out.shape
        return gst_out.reshape(B, N, Q * dim)

    def _multiconv_to_BCN(self, y_raw, N):
        """
        Handles MultiConv1D output variations (3D vs 4D) and standardizes to [B, Channels, N].
        """
        if y_raw.dim() == 3:
            # Already [B, C, N] or [B, N, C]
            if y_raw.size(2) == N: return y_raw.contiguous()
            if y_raw.size(1) == N: return y_raw.transpose(1, 2).contiguous()
            raise RuntimeError(f"MultiConv1D 3D output mismatch N={N}")

        if y_raw.dim() == 4:
            # Likely [B, N, Q, K]. Flatten Q and K into Channels.
            if y_raw.size(1) == N: # [B, N, Q, K]
                B, _, Q, K = y_raw.shape
                y = y_raw.permute(0, 2, 3, 1).contiguous() # [B, Q, K, N]
                return y.view(B, Q * K, N).contiguous()
            if y_raw.size(2) == N: # [B, Q, N, K]
                B, Q, _, K = y_raw.shape
                y = y_raw.permute(0, 1, 3, 2).contiguous() # [B, Q, K, N]
                return y.view(B, Q * K, N).contiguous()
            
        raise RuntimeError(f"Unexpected MultiConv1D output: {tuple(y_raw.shape)}")

    # ============================================================
    # Core Feature Extraction Pipeline
    # ============================================================
    def feature(self, x, sc):
        """
        Executes the physics-informed pipeline:
        1. y <- Encoder(x)
        2. x0 <- GST(x)
        3. x(t) <- Dynamics(x0, y)
        """
        # -----------------------------
        # Branch 1: Homophilic Graphs (Planetoid)
        # -----------------------------
        if self.homo:
            edge_index = sc
            x_gcn, N = self._to_x_gcn(x, edge_index)   # [N, F]
            x_btn = self._to_BTN(x, N=N)               # [1, Q, N]

            # --- A. Control Pattern 'y' (GCN) ---
            y = self.encoder_y_1(x_gcn, edge_index)
            y = self.norm_y_1(y)
            y = F.relu(y)
            y = self.dropout(y)

            identity = y
            y = self.encoder_y_2(y, edge_index)
            y = self.norm_y_2(y)
            if self.use_residual:
                y = y + identity
            y = F.relu(y)
            y = self.dropout(y)

            y = y.t().unsqueeze(0)   # [1, ch, N]

            # --- B. Oscillator Init 'x0' (GST - Eq. 4) ---
            W = to_dense_adj(edge_index.long(), max_num_nodes=N)   # [1, N, N]
            gst_out = self.gst(x_btn, W)                           # [1, N, Q, dim]
            gst_flat = self._flatten_gst(gst_out)                  # [1, N, Q*dim]
            x0 = self.patchfy_x(gst_flat.transpose(1, 2))          # [1, ch, N]

            graph_struct = edge_index

        # -----------------------------
        # Branch 2: Non-Homophilic 
        # -----------------------------
        else:
            # Infer topology (Edge Index or Dense Adj)
            if sc.dim() == 2 and sc.size(0) == 2:
                edge_index = sc
                N = self._infer_N_from_x(x)
                W = to_dense_adj(edge_index.long(), max_num_nodes=N)
                graph_struct = edge_index
            else:
                N = self._infer_N_from_x(x)
                W = self._ensure_dense_adj(sc, N=N)
                graph_struct = W

            x_btn = self._to_BTN(x, N=N)  # [1, Q, N]

            # --- A. Control Pattern 'y' (Multi-Scale Conv) ---
            # Enhanced Encoder: Expands features via gst_total channels
            y_raw = self.encoder_y(x_btn)               
            y_bcn = self._multiconv_to_BCN(y_raw, N)    
            y = self.proj_y(y_bcn)                      # [1, ch, N]

            # --- B. Oscillator Init 'x0' (GST - Eq. 4) ---
            gst_out = self.gst(x_btn, W)                # [1, N, Q, dim]
            gst_flat = self._flatten_gst(gst_out)
            x0 = self.patchfy_x(gst_flat.transpose(1, 2))

        # -----------------------------
        # Initialization Strategy
        # -----------------------------
        if self.init_x_from == "y":
            # Relaxation: Initialize strictly from Control Pattern
            x_state = y.clone()
        else:
            # Strict: Initialize from GST (Paper Eq. 4)
            x_state = x0

        # Plan A: Soft Fusion
        if self.fuse_y_into_x:
            x_state = x_state + self.alpha * y

        saved_y = y.clone()
        pre_ode = x_state.detach()

        # -----------------------------
        # Dynamics Evolution (Eq. 6)
        # -----------------------------
        xs, es, x_L = [], [], []
        for i, (kblock, ro) in enumerate(self.layers):
            # KLayer: x_new = x + dt * (omega + coupling(x) + control(y))
            _xs, _es, _ = kblock(x_state, y, graph_struct, Q=self.steps[i], gamma=self.gamma)
            x_state = _xs[-1]
            x_state = ro(x_state, graph_struct) # Readout
            xs.append(_xs)
            es.append(_es)

        post_ode = x_state.detach()
        return y, x_state, xs, es, x_L, saved_y, pre_ode, post_ode

    def forward(self, input, input_fc, input_sc, return_xs=False, return_es=False, return_features=False):
        """
        Forward Pass.
        Args:
            input: Node features.
            input_fc: (Unused placeholder).
            input_sc: Structural connectivity (Edge Index or Adjacency).
        """
        y, x_state, xs, es, x_L, saved_y, pre_ode, post_ode = self.feature(input, input_sc)

        # Select feature source for classification
        if self.pred_from == "x":
            node_features = x_state.transpose(1, 2)    # [1, N, ch]
        else:
            node_features = y.transpose(1, 2)          # [1, N, ch]

        logits = self.out_pred_node(node_features).squeeze(0)  # [N, num_class]

        # Return Logic
        if not (return_xs or return_es or return_features):
            return logits, x_state, saved_y

        ret = [logits]
        if return_xs: ret.append(xs)
        if return_es: ret.append(es)

        if return_features:
            return ret, x_state, saved_y, pre_ode, post_ode
        return ret, x_state, saved_y


class HoloBrain(nn.Module):
    """
    HoloBrain: Physics-Informed Graph Neural Network for Graph Classification (e.g., Brain Networks).

    Architecture:
      1. Attending Memory (c): Uses MultiConv1D to encode spatiotemporal features.
      2. Oscillator Init (x0): Uses GST (Eq. 4) to capture multi-scale structural frequencies.
      3. Dynamics (KMLayer): Evolves oscillators using Kuramoto dynamics (Eq. 6).
      4. Graph Head: Global Pooling -> MLP for graph-level prediction.

    Input Constraints:
      - feature_dim: MUST be provided (time series length).
      - input_sc: Can be dense adjacency [B,N,N] or edge_index [2,E].
    """

    def __init__(
        self,
        n=4,
        ch=256,
        L=1,
        Q=8,                        
        num_class=2,                
        feature_dim=None,           
        J="attn",
        ksize=1,
        c_norm="gn",
        gamma=1.0,
        use_omega=False,
        init_omg=1.0,
        global_omg=False,
        heads=8,
        dropout=0.5,
        use_residual=True,
        learn_omg=False,
        homo=False,                 # Usually False for Brain networks

        # GST Hyperparams
        gst_level=2,
        gst_wavelet=(0, 1, 2),
        probe_nodes=16,

        # Architecture Switches
        gst_total=4,                # Width multiplier for Control Encoder
        pool="max",                 # Graph pooling: "max" or "avg"
        pred_from="x",              # "x" (Final State) or "c" (Control Pattern)
        fuse_y_into_x=False,        # Plan A: x = x0 + alpha * c
        init_x_from="gst",          # "gst" (Strict Paper) or "c" (Stability)
    ):
        super().__init__()

        if feature_dim is None:
            raise ValueError("HoloBrain requires 'feature_dim' (e.g., time series length).")

        self.n = n
        self.ch = ch
        self.L = int(L)
        
        self.steps = [int(Q)] * self.L if isinstance(Q, int) else [int(q) for q in Q]
        
        self.J = [J] * self.L if isinstance(J, str) else list(J)
        self.gamma = float(gamma)
        self.num_class = int(num_class)
        self.homo = bool(homo)
        self.use_residual = bool(use_residual)
        self.dropout = nn.Dropout(p=float(dropout))

        self.input_dim = int(feature_dim)  
        self.gst_total = int(gst_total)
        self.pred_from = pred_from
        self.fuse_y_into_x = bool(fuse_y_into_x)
        self.init_x_from = init_x_from

        # -------------------------
        # 1. Control Encoder -> c
        # -------------------------
        if self.homo:
            # GCN path (Rare for brain graphs, but kept for compatibility)
            self.c_gcn1 = GCNConv(self.input_dim, ch)
            self.c_gcn2 = GCNConv(ch, ch)
            self.c_norm1 = nn.LayerNorm(ch)
            self.c_norm2 = nn.LayerNorm(ch)
        else:
            # Multi-Scale 1D Conv (Standard for Brain)
            # Expects input [B, T, N] -> encodes temporal/feature dim
            self.c_conv = MultiConv1D(
                in_channels=self.input_dim,
                out_channels=self.input_dim,
                num_convs=self.gst_total,
            )
            self.c_proj = nn.Conv1d(self.input_dim * self.gst_total, ch, kernel_size=1)

        # -------------------------
        # 2. GST -> x0 (Eq. 4)
        # -------------------------
        self.gst = Wavelet(wavelet=list(gst_wavelet), level=int(gst_level))

        # Probe Output Dimension
        with torch.no_grad():
            Np = max(int(probe_nodes), 2)
            x_probe = torch.zeros(1, self.input_dim, Np)   # [1, T, N]
            adj_probe = torch.eye(Np).unsqueeze(0)         # [1, N, N]
            out = self.gst(x_probe, adj_probe)             # [1, N, T, dim]
            
            if out.dim() != 4:
                raise RuntimeError(f"GST Output Error: Expected [B,N,T,dim], got {tuple(out.shape)}")
            
            gst_channels = int(out.size(-2) * out.size(-1)) # T * dim

        self.patchfy_x = nn.Conv1d(gst_channels, ch, kernel_size=1)

        if self.fuse_y_into_x:
            self.alpha = nn.Parameter(torch.tensor(0.0))

        # -------------------------
        # 3. Dynamics (Kuramoto)
        # -------------------------
        self.layers = nn.ModuleList()
        for l in range(self.L):
            kblock = KMLayer(
                n=n, ch=ch, J=self.J[l], c_norm=c_norm, use_omega=use_omega,
                init_omg=init_omg, global_omg=global_omg, heads=heads,
                learn_omg=learn_omg, ksize=ksize, feature_dim=self.input_dim,
            )
            # ReadOutConv applies inter-block transition
            ro = ReadOutConv(ch, ch, self.n, 1, 1, 0, self.homo)
            self.layers.append(nn.ModuleList([kblock, ro]))

        # -------------------------
        # 4. Graph Classification Head
        # -------------------------
        # Pooling: Aggregate Node Embeddings -> Graph Embedding
        if pool == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pool == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError("pool must be 'max' or 'avg'")

        # MLP Head
        self.graph_head = nn.Sequential(
            nn.Linear(ch, 4 * ch),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(4 * ch, self.num_class),
        )

    # ============================================================
    # Helpers
    # ============================================================
    def _infer_N(self, x):
        """Infer Number of Nodes (N)."""
        if x.dim() == 3:
            # If dim 1 is input_dim, N is dim 2 (e.g., [B, F, N])
            if x.size(1) == self.input_dim: return x.size(2)
            # If dim 2 is input_dim, N is dim 1 (e.g., [B, N, F])
            if x.size(2) == self.input_dim: return x.size(1)
        if x.dim() == 2: return x.size(0)
        raise ValueError(f"Cannot infer N from x shape {tuple(x.shape)}")

    def _to_BTN(self, x, N=None):
        """Normalize input to [B, T, N] where T=feature_dim."""
        if x.dim() == 2: # [N, F] -> [1, F, N]
            if x.size(1) != self.input_dim: raise ValueError(f"Dim mismatch F={self.input_dim}")
            return x.t().unsqueeze(0).contiguous()
        
        if x.dim() == 3:
            # [B, T, N]
            if x.size(1) == self.input_dim: return x.contiguous()
            # [B, N, T] -> [B, T, N]
            if x.size(2) == self.input_dim: return x.transpose(1, 2).contiguous()
            
        raise ValueError(f"Cannot convert x={tuple(x.shape)} to [B, T, N]")

    def _to_x_gcn(self, x, edge_index):
        """Normalize input for GCN [N, F]."""
        N = int(edge_index.max().item()) + 1
        if x.dim() == 2: return x.contiguous(), N
        if x.dim() == 3: return x.squeeze(0).contiguous(), N # Assumes B=1 for homo
        return x, N

    def _ensure_dense_adj(self, sc, N):
        """Ensure adjacency is dense [B, N, N]."""
        if sc.dim() == 2 and sc.size(0) == 2: # Edge Index
            return to_dense_adj(sc.long(), max_num_nodes=N)
        if sc.dim() == 2: return sc.unsqueeze(0).contiguous()
        if sc.dim() == 3: return sc.contiguous()
        raise ValueError(f"Unexpected adj shape: {tuple(sc.shape)}")

    def _flatten_gst(self, out):
        B, N, T, dim = out.shape
        return out.reshape(B, N, T * dim)

    def _multiconv_to_BCN(self, y_raw, N):
        """Handle MultiConv1D output variations."""
        if y_raw.dim() == 3:
            if y_raw.size(2) == N: return y_raw.contiguous()
            return y_raw.transpose(1, 2).contiguous()
        
        if y_raw.dim() == 4:
            # [B, N, T, K] -> [B, T*K, N]
            if y_raw.size(1) == N: 
                B, _, T, K = y_raw.shape
                y = y_raw.permute(0, 2, 3, 1).contiguous()
                return y.view(B, T * K, N).contiguous()
            if y_raw.size(2) == N:
                B, T, _, K = y_raw.shape
                y = y_raw.permute(0, 1, 3, 2).contiguous()
                return y.view(B, T * K, N).contiguous()
                
        raise RuntimeError(f"Unexpected MultiConv output: {tuple(y_raw.shape)}")

    # ============================================================
    # Forward Pass
    # ============================================================
    def feature(self, inp, sc):
        # Infer N and topology
        N = self._infer_N(inp)
        
        if self.homo:
            # Homo branch (Legacy support)
            edge_index = sc
            W = to_dense_adj(edge_index.long(), max_num_nodes=N)
            x_btn = self._to_BTN(inp, N=N)
            
            x_gcn, _ = self._to_x_gcn(inp, edge_index)
            c = self.c_gcn1(x_gcn, edge_index)
            c = self.c_norm1(c)
            c = F.relu(c)
            c = self.dropout(c)
            
            c = self.c_gcn2(c, edge_index)
            c = self.c_norm2(c)
            c = F.relu(c)
            c = self.dropout(c)
            c = c.t().unsqueeze(0) # [1, ch, N]
            
            graph_struct = edge_index
        else:
            # Brain branch (Standard)
            W = self._ensure_dense_adj(sc, N) # [B, N, N]
            graph_struct = W
            x_btn = self._to_BTN(inp, N=N)    # [B, T, N]

            # Control Pattern c via MultiConv
            c_raw = self.c_conv(x_btn)
            c_bcn = self._multiconv_to_BCN(c_raw, N)
            c = self.c_proj(c_bcn)            # [B, ch, N]
            c = self.dropout(c)

        # Oscillator Init x0 via GST
        gst_out = self.gst(x_btn, W)          # [B, N, T, dim]
        gst_flat = self._flatten_gst(gst_out)
        x0 = self.patchfy_x(gst_flat.transpose(1, 2))

        # Initialization Strategy
        if self.init_x_from == "c":
            x = c.clone()
        else:
            x = x0

        if self.fuse_y_into_x:
            x = x + self.alpha * c

        saved_c = c.clone()
        pre_ode = x.detach()

        # Dynamics Evolution
        xs, es = [], []
        for i, (kblock, ro) in enumerate(self.layers):
            _xs, _es, _ = kblock(x, c, graph_struct, T=self.steps[i], gamma=self.gamma)
            x = _xs[-1]
            x = ro(x, graph_struct)
            xs.append(_xs)
            es.append(_es)

        post_ode = x.detach()
        return c, x, xs, es, saved_c, pre_ode, post_ode

    def forward(self, input, input_fc, input_sc, return_xs=False, return_es=False, return_features=False):
        """
        Args:
            input: Node features [B, N, F] or [B, F, N]
            input_sc: Adjacency [B, N, N]
        Returns:
            logits: [B, num_class]
        """
        c, x, xs, es, saved_c, pre_ode, post_ode = self.feature(input, input_sc)

        # Graph Classification Logic
        # 1. Choose Embedding
        node_emb = x if self.pred_from == "x" else c  # [B, ch, N]
        
        # 2. Global Pooling [B, ch, N] -> [B, ch, 1]
        g = self.pool(node_emb).squeeze(-1)           # [B, ch]
        
        # 3. MLP Classifier -> [B, num_class]
        logits = self.graph_head(g)

        if not (return_xs or return_es or return_features):
            return logits, x, saved_c

        ret = [logits]
        if return_xs: ret.append(xs)
        if return_es: ret.append(es)

        if return_features:
            return ret, x, saved_c, pre_ode, post_ode
        return ret, x, saved_c
