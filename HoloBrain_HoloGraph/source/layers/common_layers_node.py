import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch_geometric.nn import GCNConv

# ==========================================
# 1. Core Layers
# ==========================================

class ReadOutConv(nn.Module):
    def __init__(self, inch, outch, out_dim, kernel_size=3, stride=1, padding=1, homo=False):
        super().__init__()
        self.outch = outch
        self.out_dim = out_dim
        self.homo = homo
        if self.homo:
            self.invconv = GCNConv(inch, outch * out_dim)
        else:
            self.invconv = nn.Linear(inch, outch * out_dim)
        
        self.bias = nn.Parameter(torch.zeros(outch))

    def forward(self, x, sc):
        # x shape: [B, Q, C] (assumed based on KMLayer output)
        if self.homo:
            # GCN usually expects [N, F]
            # Assuming x is [1, N, C], squeeze to [N, C]
            x_in = x.squeeze(0) 
            x_out = self.invconv(x_in, sc) # [N, outch*out_dim]
            x = x_out.unsqueeze(0) # [1, N, outch*out_dim]
            x = x.transpose(1, 2) # [1, outch*out_dim, N]
        else:
            # x: [B, Q, C] -> Linear -> [B, Q, outch*out_dim]
            # transpose -> [B, outch*out_dim, Q]
            x = self.invconv(x).transpose(1, 2) 
            
        # Reshape and Norm logic
        x = x.unflatten(1, (self.outch, -1))  # [B, outch, out_dim, Q]
        x = torch.linalg.norm(x, dim=2)       # [B, outch, Q]
        x = x + self.bias[None, :, None]
        return x 

class MultiConv1D(nn.Module):
    """Used for generating control pattern y in non-homo graphs."""
    def __init__(self, in_channels, out_channels, num_convs=4):
        super(MultiConv1D, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
            for _ in range(num_convs)
        ])
    
    def forward(self, x):
        # Input x: [B, F, N] (treated as [B, Channels, Length])
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x).unsqueeze(-1)) 
        output = torch.cat(outputs, dim=-1)  # [B, F, N, num_convs]
        output = output.permute(0, 2, 1, 3)  # [B, N, F, num_convs] ? Check dimensions
        # Based on HoloGraph usage:
        # returns typically [B, N, F, K] or similar to be flattened.
        return output

class Attention(nn.Module):
    """Used in KLayer when J='attn'."""
    def __init__(self, ch, heads=8, weight="fc", kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.heads = heads
        self.head_dim = ch // heads
        self.scale = self.head_dim ** -0.5
        
        if weight == "fc":
            self.qkv = nn.Linear(ch, 3 * ch, bias=False)
            self.proj = nn.Linear(ch, ch, bias=False)
        else:
            raise ValueError("Only fc weight type is supported")

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

class adj_connectivity(nn.Module):
    """Used in KLayer when J='adj' (Dense Learnable Adjacency)."""
    def __init__(self, num_nodes):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_nodes, num_nodes))
        nn.init.xavier_uniform_(self.weight)
        self._latest_A_weighted = None

    def forward(self, x, A):
        A_weighted = A * self.weight
        # Save for visualization/regularization
        if self.training: 
            self._latest_A_weighted = A_weighted.detach() 
        out = A_weighted @ x
        return F.relu(out)
    
    def update_latest_A_weighted(self, A_weighted):
        self._latest_A_weighted = A_weighted.detach().clone()

class ScaleAndBias(nn.Module):
    """Used in KLayer when c_norm='sandb'."""
    def __init__(self, num_channels, token_input=True):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.token_input = token_input

    def forward(self, x):
        if self.token_input:
            shape = [1, 1, -1]
        else:
            shape = [1, -1] + [1] * (x.dim() - 2)
        return x * self.scale.view(*shape) + self.bias.view(*shape)

# ==========================================
# 2. Training Utils
# ==========================================

def compute_weighted_metrics(preds, gts, num_classes=None):
    if num_classes is None:
        num_classes = len(torch.unique(gts))
    
    device = preds.device
    class_counts = torch.zeros(num_classes, device=device)
    for cls in range(num_classes):
        class_counts[cls] = (gts == cls).sum()

    correct = (preds == gts).sum().item()
    acc = correct / gts.size(0)

    precision_list, recall_list, f1_list = [], [], []
    for cls in range(num_classes):
        tp = ((preds == cls) & (gts == cls)).sum().item()
        fp = ((preds == cls) & (gts != cls)).sum().item()
        fn = ((preds != cls) & (gts == cls)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        weight = class_counts[cls] / class_counts.sum()
        precision_list.append(precision * weight)
        recall_list.append(recall * weight)
        f1_list.append(f1 * weight)

    pre = sum(precision_list)
    recall = sum(recall_list)
    f1 = sum(f1_list)

    return acc, pre, recall, f1