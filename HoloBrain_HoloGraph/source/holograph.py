import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj

from source.modules.kuramoto_layer import Kuramoto_Solver

from source.modules.common_layers_node import ReadOutConv, MultiConv1D
from source.modules.GST import GSTWavelet

class HoloGraph(nn.Module):
    """
    HoloGraph: Physics-Informed Graph Neural Network.
    
    Theoretical Alignment:
      1. Attending Memory 'y' (Control Pattern): Encodes structural/feature context.
      2. Initialization 'x(0)' [Eq. 4]: Generated via Geometric Scattering Transform (GST).
      3. Dynamics [Eq. 6]: Evolved via Kuramoto Dynamics for Q steps.
    """
    def __init__(
        self,
        n=4,
        ch=256, #256
        L=1,
        Q=8,             
        num_class=4,
        feature_dim=None,
        J="attn",
        ksize=1,
        c_norm="gn",
        gamma=1.0,
        use_omega=False,
        init_omg=1.0,
        global_omg=False,
        maxpool=True,
        heads=8,
        use_residual=True,
        dropout=0.5,    
        learn_omg=False,
        homo=False,
        level=3,
        order=0,
        gst_total=4,
    ):
        super().__init__()
        self.n = n
        self.ch = ch
        self.L = L
        
        if isinstance(Q, int):
            self.Q = [Q] * L
        else:
            self.Q = Q
            
        if isinstance(J, str):
            self.J = [J] * L
        else:
            self.J = J
        self.gamma = gamma
        self.feature_dim = feature_dim
        self.num_class = num_class
        self.feature_dim = feature_dim
        self.homo = homo
        self.gst_total = gst_total
        self.encoder_y1 = GCNConv(feature_dim, ch)
        self.encoder_y2 = GCNConv(ch, ch)
        self.norm1 = nn.LayerNorm(ch)
        self.norm2 = nn.LayerNorm(ch)
        self.encoder_y = MultiConv1D(in_channels=self.feature_dim, out_channels=self.feature_dim, num_convs=self.gst_total)
        self.proj_y_linear = nn.Sequential(
            # nn.Conv1d(self.feature_dim*4, ch, kernel_size=1, stride=1, padding=0),
            nn.Linear(self.feature_dim, ch),
            # nn.Linear(self.feature_dim*4, ch),
            # nn.LayerNorm(ch),
            # nn.ReLU(),
            # nn.Linear(ch, ch),
            # nn.Dropout(p=0.5)
        )
        self.proj_y_flat = nn.Linear(self.feature_dim*self.gst_total, ch)
        self.proj_y = nn.Conv1d(self.feature_dim*self.gst_total, ch, kernel_size=1, stride=1, padding=0)

        # self.proj_x0 = GCNConv(self.feature_dim*4, ch)
        # self.proj_x0 = nn.Linear(self.feature_dim, ch)
        # self.proj_x0 = nn.Sequential(
        #     nn.Linear(self.feature_dim*4, ch),
        #     # nn.LayerNorm(ch),
        #     nn.ReLU(),
        #     nn.Linear(ch, ch),
        #     # nn.Dropout(p=0.5)
        # ) 
        
        # self.proj_x0 = nn.Sequential(
        #     nn.Conv1d(self.feature_dim, ch, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv1d(ch, ch, kernel_size=3, stride=1, padding=1)
        # ) 
        self.proj_x0 = nn.Conv1d(self.feature_dim*self.gst_total, ch, kernel_size=1, stride=1, padding=0)
        self.proj_x0_aux = nn.Sequential(
            nn.Conv1d(self.feature_dim*self.gst_total, ch, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv1d(ch, ch, kernel_size=3, stride=1, padding=1)
        )

        self.order = order
        self.level = level
        self.layers = nn.ModuleList()
        self.gst = GSTWavelet(list(range(self.order + 1)), self.level)
        
        chs = [ch] * (self.L + 1)

        for l in range(self.L):
            ch = chs[l]
            if l == self.L - 1:
                ch_next = chs[l + 1]
            else:
                ch_next = chs[l + 1]

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
                feature_dim=self.feature_dim,
                return_energy=False,
            )
            
            readout = ReadOutConv(ch, ch_next, self.n, 1, 1, 0, self.homo)

            self.layers.append(nn.ModuleList([klayer, readout]))
        ch = ch_next

        
        if maxpool:
            pool = nn.AdaptiveMaxPool1d(1) 
        else:
            pool = nn.AdaptiveAvgPool1d(1)
        self.out_pred_graph = nn.Sequential(
            nn.Identity(),
            pool,  
            nn.Flatten(start_dim=1),  # [B, C]
            nn.Linear(ch, 4 * ch), 
            nn.ReLU(),
            nn.Linear(4 * ch, self.num_class),  
        )

        self.out_pred_node = nn.Sequential( 
            nn.Identity(),
            # nn.Linear(ch, 4*ch),  
            # nn.ReLU(), 
            nn.Linear(ch, self.num_class)  
        )
        
        self.use_residual = use_residual
        
        self.dropout = nn.Dropout(p=dropout)
    
    def feature(self, inp, inp_fc, inp_sc):

        if self.homo:
            y = self.encoder_y1(inp.squeeze(0).transpose(0, 1), inp_sc)
            y = self.norm1(y)
            y = F.relu(y)
            y = self.dropout(y)
            
            identity = y
            y = self.encoder_y2(y, inp_sc)
            y = self.norm2(y)
            if self.use_residual:
                y = y + identity
            y = F.relu(y)
            y = self.dropout(y)

            y = y.t().unsqueeze(0)  # [1, C, N]
        else:
            # # print("encoder_y")
            # # # New processing using encoder_y (1D CNN) for c
            y = self.encoder_y(inp)  # Use MultiConv1D to get y
            # # print(f"Debug feature - after encoder_y, c shape: {c.shape}")
            # # c = F.relu(c)
            # # c = self.dropout(c)
            # # c = c.transpose(1,2).flatten(start_dim=2)
            # # # c = self.proj_y_linear(c.transpose(1,2)) # 1-d conv
            # # print(f"Debug feature - after transpose/flatten, c shape: {c.shape}")
            y = y.flatten(start_dim=2)
            # c = self.proj_y_flat(c).transpose(1,2) # linear



            y = self.proj_y(y.transpose(1,2))
            # print(f"Debug feature - after transpose/flatten, c shape: {c.shape}")


            # c = self.proj_y_linear(inp.transpose(1, 2)).transpose(1, 2)  # Linear


        
        
        
        # # Use GST to get x
        # Need to know N to properly format input for GST
        if inp.dim() == 2:
            N = inp.size(0)
            x_gst_in = inp.t().unsqueeze(0) # [1, F, N]
        elif inp.dim() == 3:
            # Heuristic for Planetoid [1, F, N] or [1, N, F]
            if inp.size(1) == self.feature_dim: 
                x_gst_in = inp # Assume [1, F, N]
                N = inp.size(2)
            else:
                x_gst_in = inp.transpose(1, 2) # Assume [1, N, F] -> [1, F, N]
                N = inp.size(1)
        else:
            # Fallback if sc is provided
            N = inp_sc.shape[-1]
            x_gst_in = inp.contiguous()

        # Ensure adjacency is dense [1, N, N]
        if inp_sc.dim() == 2 and inp_sc.size(0) == 2:
            sc_dense = to_dense_adj(inp_sc.long(), max_num_nodes=N)
        elif inp_sc.dim() == 2:
            sc_dense = inp_sc.unsqueeze(0)
        else:
            sc_dense = inp_sc

        # GST Forward
        x = self.gst(x_gst_in, sc_dense)  # Use Wavelet (GST) to get x
        x = x.flatten(start_dim=2)
        
        x = self.proj_x0(x.transpose(1,2)).transpose(1, 2)
        
        # # x = self.proj_x0(x).transpose(1, 2)  # Linear
        # # print("x shape: ", x.shape, "c shape: ", c.shape)
        # # x = self.proj_x0(inp.transpose(1, 2)).transpose(1, 2)

        # # x = self.proj_x0(inp)
        # # x = self.proj_x0(inp.transpose(1, 2)).transpose(1, 2)
        # # print("x shape: ", x.shape, "c shape: ", c.shape)

        # # brain
        # # Use GST to get x
        # x = self.gst(inp, inp_sc)  # Use Wavelet (GST) to get x
        # x = x.flatten(start_dim=2)
        # # x = self.proj_x0(x.transpose(1,2))
        # # x = self.proj_x0(x).transpose(1, 2)  # Linear
        # # print("x shape: ", x.shape, "c shape: ", c.shape)
        # x = self.proj_x0_aux(x.transpose(1, 2))
        # # print("x shape: ", x.shape, "c shape: ", c.shape)
        
        # Original x assignment (commented out)
        # x = c.clone()  

        
        saved_y = y.clone()
        saved_x = x.clone()  # Save initial x


        xs = []
        es = []
        x_L = []
        pre_ode_features = x.detach()

        for i, (kblock, ro) in enumerate(self.layers):
            # identity = x
            # [Rename]: Pass Q instead of T
            _xs, _es, save_x = kblock(x, y, inp_sc, Q=self.Q[i], gamma=self.gamma)
            x = _xs[-1]  # [1, N, C]
            
            # # Apply layer normalization
            # batch_size, num_nodes, channels = x.size()
            # x = x.reshape(-1, channels)
            # x = self.layer_norms[i](x)
            # x = x.reshape(batch_size, num_nodes, channels)
            
            # if x.shape == identity.shape:
            #     x = x + identity
            
            x = ro(x, inp_sc)
            
            xs.append(_xs)
            es.append(_es)
        
        post_ode_features = x.detach()
        return y, x, xs, es, x_L, saved_y, pre_ode_features, post_ode_features
    

    def forward(self, input, input_fc, input_sc, return_xs=False, return_es=False, return_features=False):

        final_y, x, xs, es, x_L, saved_y, pre_ode_features, post_ode_features = self.feature(input, input_fc, input_sc)
        
        # # For node classification, we use the node features directly
        node_features = final_y.transpose(1, 2)  # [B, N, C]
        logits = self.out_pred_node(node_features)  # [B, N, num_classes]
        # logits = self.out_pred_graph(final_y)  # This is for graph-level prediction, not node-level
        # print("final_c shape: ", final_c.shape)
        # logits = self.out_pred_graph(final_c)
        
        ret = [logits]
        if return_xs:
            ret.append(xs)
        if return_es:
            ret.append(es)


        # print(len(ret), ret[0].shape)
        if len(ret) == 1:
            if return_features:
                return ret[0], x, saved_y, pre_ode_features, post_ode_features
            else:
                return ret[0], x, saved_y
        
        if return_features:
            return ret, x, saved_y, pre_ode_features, post_ode_features # [B, 116, T*L]
        else:
            return ret, x, saved_y
