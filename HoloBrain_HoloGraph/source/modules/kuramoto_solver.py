import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv

class OmegaModule(nn.Module):
    def __init__(self, hidden_dim):
        super(OmegaModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.omega_param = nn.Parameter((1.0 / np.sqrt(2)) * torch.ones(hidden_dim // 2, 2))

    def forward(self, x):
        B, T, C = x.shape
        x_reshaped = x.transpose(1, 2).unflatten(1, (C // 2, 2))
        omega = torch.linalg.norm(self.omega_param, dim=1)  
        omega = omega.unsqueeze(0)  
        while omega.ndim < x_reshaped.ndim:
            omega = omega.unsqueeze(-1)
        omega_x = torch.stack([omega * x_reshaped[:, :, 1], -omega * x_reshaped[:, :, 0]], dim=2)
        omega_x = omega_x.flatten(1, 2).transpose(1, 2)
        return omega_x

class SyncModule(nn.Module):
    def __init__(self, num_nodes):
        super(SyncModule, self).__init__()
        self.param = nn.Parameter(torch.empty(num_nodes, num_nodes))
        nn.init.xavier_uniform_(self.param)

    def forward(self, adj, x):
        sym_param = (self.param + self.param.t()) / 2
        P_new = sym_param * adj  
        output = F.relu(torch.matmul(P_new, x))
        return output
    
class f_phi(nn.Module):
    def __init__(self, in_channels,  mapping_type, N):
        super(f_phi, self).__init__()
        self.mapping_type = mapping_type
        self.N = N
        if self.mapping_type == 'conv':
            self.mapping_conv = nn.Conv1d(in_channels, in_channels * N, kernel_size=1)
        elif self.mapping_type == 'gconv':
            self.mapping_gconv = GCNConv(in_channels, in_channels * N)
        self.bias = nn.Parameter(torch.zeros(in_channels))
        
    def forward(self, x, adj):
        if self.mapping_type == 'conv':
            x = x.permute(0, 2, 1)  
            x = self.mapping_conv(x)  
        elif self.mapping_type == 'gconv':
            edge_index = self._get_edge_index(adj.squeeze())
            x = self.mapping_gconv(x.squeeze(0), edge_index).T.unsqueeze(0)
        x = x.unflatten(1, (self.N, -1))
        x = torch.linalg.norm(x, dim=2)
        x = x + self.bias.unsqueeze(0).unsqueeze(-1)
        return x
    
    def _get_edge_index(self, adj):
        edge_index = torch.nonzero(adj, as_tuple=False).T
        return edge_index

class Kuramoto_Solver(nn.Module):  
    def __init__(self, N, hidden_dim, beta, T, L, mapping_type='conv', num_modes=116):
        super().__init__()
        self.N = N
        self.T = T
        self.L = L
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.mapping_type = mapping_type
        self.num_modes = num_modes
        
        self.omega_module = OmegaModule(hidden_dim)
        self.sync_module = SyncModule(num_modes)
        self.norm_y = nn.GroupNorm(hidden_dim // N, hidden_dim, affine=True)
        self.f_phi = f_phi(in_channels=hidden_dim, out_channels=hidden_dim, mapping_type=mapping_type, N=N)
        
    def surrounding_osc(self, x: torch.Tensor, y: torch.Tensor, adj: torch.Tensor, memory_level=1):
        wx = self.sync_module(adj, x)
        z = wx + memory_level * y
        return z
    
    def project_osc(self, x, z):
        B, T, C = x.shape
        x = x.transpose(1, 2).unflatten(1, (self.N, C // self.N))
        z = z.transpose(1, 2).unflatten(1, (self.N, C // self.N))
        phi_z = z - torch.sum(x * z, dim=-1, keepdim=True) * x
        phi_z = phi_z.flatten(1, 2).transpose(1, 2)
        return phi_z
    
    def update_osc(self, omega_x, phi_z):
        delta_x = omega_x + self.beta * phi_z.flatten(1, 2).transpose(1, 2)
        return delta_x
    
    def map_to_sphere(self, x):
        x = x.transpose(1, 2).unflatten(1, (-1, self.N))
        x = F.normalize(x, dim=2)
        x = x.flatten(1, 2).transpose(1, 2)
        return x

    
    def forward(self, x, y, adj):
        x_L = []
        for _ in range(self.L):
            y = self.norm_y(y)
            y = y.transpose(1, 2) if y.shape[1]==self.hidden else y #[B, T, C]
            x = x.transpose(1, 2) if x.shape[1]==self.hidden else x #[B, T, C]
            x = self.map_to_sphere(x)
            for _ in range(self.T):
                omega = self.omega_module(x)
                Z = self.surrounding_osc(x, y, adj)
                Phi_Z = self.project_osc(x, Z)
                Delta_X = self.update_osc(omega, Phi_Z)
                x = self.map_to_sphere(Delta_X) 
                x_L.append(x.unsqueeze(1))
                
            y = self.f_phi(x, adj)
        return x, y, x_L
