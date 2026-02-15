import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import einops
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
from torch.optim.lr_scheduler import _LRScheduler
import logging
import os

class LinearWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, last_iter=-1):
        self.warmup_iters = warmup_iters
        self.current_iter = 0 if last_iter == -1 else last_iter
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        super(LinearWarmupScheduler, self).__init__(optimizer, last_epoch=last_iter)

    def get_lr(self):
        if self.current_iter < self.warmup_iters:
            # Linear warmup phase
            return [
                base_lr * (self.current_iter + 1) / self.warmup_iters
                for base_lr in self.base_lrs
            ]
        else:
            # Maintain the base learning rate after warmup
            return [base_lr for base_lr in self.base_lrs]

    def step(self, it=None):
        if it is None:
            it = self.current_iter + 1
        self.current_iter = it
        super(LinearWarmupScheduler, self).step(it)

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class ResBlock(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class ReadOutConv(nn.Module):
    def __init__(
        self,
        inch,
        outch,
        out_dim,
        kernel_size=3,
        stride=1,
        padding=1,
        homo=False,
    ):
        super().__init__()
        self.outch = outch
        self.out_dim = out_dim
        self.homo = homo
        if self.homo:
            self.invconv = GCNConv(inch, outch * out_dim)
        else:
            # self.invconv = nn.Conv1d(
            #     inch,
            #     outch * out_dim,
            #     kernel_size=kernel_size,
            #     stride=stride,
            #     padding=padding,
            # )
            self.invconv = nn.Linear(
                inch,
                outch * out_dim,
            )
        
        self.bias = nn.Parameter(torch.zeros(outch))

    def forward(self, x ,sc):
        # x shape: [B, T, C]
        if self.homo:
            x = self.invconv(x.squeeze(0), sc).T.unsqueeze(0)  # Apply GCN, output shape: [B, outch * out_dim, T_out]
        else:
            # x = x.permute(0, 2, 1)
            # x = self.invconv(x)  # Apply 1D convolution, output shape: [B, outch * out_dim, T_out]

            x = self.invconv(x).transpose(1,2) # Linear
        x = x.unflatten(1, (self.outch, -1))  # Split channels into [B, outch, out_dim, T_out]
        x = torch.linalg.norm(x, dim=2)  # Compute norm across the `out_dim` dimension
        x = x + self.bias[None, :, None]  # Add bias
        return x  # Output shape: [B, outch, T_out]

class ScaleAndBias(nn.Module):
    def __init__(self, num_channels, token_input=True):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.token_input = token_input

    def forward(self, x):
        # Determine the shape for scale and bias based on input dimensions
        if self.token_input:
            # token input
            shape = [1, 1, -1]
            scale = self.scale.view(*shape)
            bias = self.bias.view(*shape)
        else:
            # image input
            shape = [1, -1] + [1] * (x.dim() - 2)
            scale = self.scale.view(*shape)
            bias = self.bias.view(*shape)
        return x * scale + bias

class Attention(nn.Module):
    def __init__(
        self,
        ch,
        heads=8,
        weight="fc",
        kernel_size=1,
        stride=1,
        padding=0,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = ch // heads
        self.scale = self.head_dim ** -0.5
        self.weight = weight

        # Use a single linear layer for QKV projection
        if weight == "fc":
            self.qkv = nn.Linear(ch, 3 * ch, bias=False)
            self.proj = nn.Linear(ch, ch, bias=False)
        else:
            raise ValueError("Only fc weight type is supported for optimized attention")

    def forward(self, x):
        B, T, C = x.shape
        
        # Fused QKV projection
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]


        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)

        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

    
def compute_weighted_metrics(preds, gts, num_classes=None):
    """
    Args:
        preds (torch.Tensor)
        gts (torch.Tensor)
        num_classes (int, optional)
    """
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

    pre = sum(precision_list).item()
    recall = sum(recall_list).item()
    f1 = sum(f1_list).item()

    return acc, pre, recall, f1


class MultiConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_convs=4):
        super(MultiConv1D, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
            # nn.Linear(in_channels, out_channels)
            for _ in range(num_convs)
        ])
    
    def forward(self, x):
        # Input x: [B, N, T]
        outputs = []
        # x=x.transpose(1,2)
        for conv in self.convs:
            outputs.append(conv(x).unsqueeze(-1))  # [B, N, T] -> [B, N, T, 1]
        output = torch.cat(outputs, dim=-1)  # [B, N, T, num_convs]
        output = output.permute(0,2,1,3)

        # print("multi conv output shape: ", output.shape)
        return output
 
class adj_connectivity(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_nodes, num_nodes))
        nn.init.xavier_uniform_(self.weight)
        self._latest_A_weighted = None

    def forward(self, x, A):
        A_weighted = A * self.weight
        if self._latest_A_weighted is None:
            self._latest_A_weighted = A_weighted.detach().clone()
        out = A_weighted @ x
        return F.relu(out)
    
    def get_learned_weights(self):
        return {
            'weight': self.weight.detach().clone(),
            'A_weighted': self._latest_A_weighted.clone() if self._latest_A_weighted is not None else None
        }
    
    def update_latest_A_weighted(self, A_weighted):
        self._latest_A_weighted = A_weighted.detach().clone()


def effective_rank(X, eps=1e-8):
    """
    Compute the effective rank of a feature matrix using PyTorch.
    
    Parameters:
        X: torch.Tensor of shape (N, d), on any device
        eps: float, small constant to avoid log(0)
    
    Returns:
        torch scalar: effective rank
    """
    # Compute singular values
    U, S, V = torch.linalg.svd(X, full_matrices=False)
    S = S[S > eps]  # filter out near-zero singular values

    p = S / S.sum()
    entropy = -(p * (p + eps).log()).sum()
    return entropy.exp()

def class_mix_score(X, y, delta=1e-6, eps=1e-8, X0=None, distance_type='euclidean'):
    """
    Compute the class-mix convergence score S^(l) in PyTorch.
    
    Parameters:
        X: torch.Tensor of shape (N, d), feature matrix
        y: torch.Tensor of shape (N,), integer class labels
        delta: float, stabilization constant
        eps: float, to avoid division by 0
        X0: torch.Tensor or None, initial features X^(0) for normalization
        distance_type: str, 'euclidean' or 'cosine'
    
    Returns:
        torch scalar: normalized score S^(l) if X0 is provided; else rho^(l)
    """

    def pairwise_energy(X, y, same_class=True):
        if distance_type == 'cosine':
            X = torch.nn.functional.normalize(X, dim=1, eps=eps)
            dist = 1 - torch.matmul(X, X.T).clamp(-1 + eps, 1 - eps)  # cosine distance: 1 - cosine similarity
        else:
            diff = X.unsqueeze(1) - X.unsqueeze(0)  # (N, N, d)
            dist = (diff ** 2).sum(dim=-1)         # Euclidean squared distance

        # Build mask
        same = (y.unsqueeze(0) == y.unsqueeze(1))  # (N, N)
        mask = same if same_class else (~same)

        # Exclude diagonal and duplicate pairs
        tril_mask = torch.tril(torch.ones_like(mask), diagonal=-1).bool()
        valid_mask = mask & tril_mask

        total = dist[valid_mask].sum()
        count = valid_mask.sum().float()
        return total / (count + eps)

    E_w = pairwise_energy(X, y, same_class=True)
    E_b = pairwise_energy(X, y, same_class=False)
    rho = E_w / (E_b + eps)

    if X0 is not None:
        E0_w = pairwise_energy(X0, y, same_class=True)
        E0_b = pairwise_energy(X0, y, same_class=False)
        rho0 = E0_w / (E0_b + eps)
        S = 1.0 - torch.abs(rho - 1.0) / (torch.abs(rho0 - 1.0) + delta)
        return S
    else:
        return rho

def create_logger(logging_dir):
    """
    Creates a logger that writes to both file and stdout.
    """
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt", mode='w'),
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    return logger

class SpectralNetLoss(nn.Module):
    def __init__(self):
        super(SpectralNetLoss, self).__init__()

    def forward(self, W, Y):
        """
        Args:
            W: Affinity matrix (Adjacency) [Batch, N, N] or [N, N]
            Y: Output embeddings [Batch, N, K] or [N, K]
        Returns:
            Spectral Loss: Trace(Y.T * L * Y) / Batch_Size
            (Minimizing this is equivalent to Spectral Clustering objective)
        """
        # Ensure batch dimension
        if W.dim() == 2:
            W = W.unsqueeze(0)
        if Y.dim() == 2:
            Y = Y.unsqueeze(0)
            
        B, N, _ = W.shape
        
        # 1. Compute Degree Matrix D
        # D is diagonal matrix where D_ii = sum_j(W_ij)
        D = torch.diag_embed(W.sum(dim=1))
        
        # 2. Compute Laplacian L = D - W
        L = D - W
        
        # 3. Compute Trace(Y^T * L * Y)
        # Y: [B, N, K] -> Y.T: [B, K, N]
        # L: [B, N, N]
        # term: Y.T @ L @ Y -> [B, K, K]
        
        # Note: Usually SpectralNet also enforces Y^T * Y = I (Orthonormality).
        # Assuming the model architecture (like HoloGraph) or explicit constraints handle scale,
        # we focus on the Laplacian Trace minimization here.
        
        loss = 0
        for i in range(B):
            y_i = Y[i] # [N, K]
            l_i = L[i] # [N, N]
            # loss += Trace(y_i.T @ l_i @ y_i)
            # More efficient: sum((y_i.T @ l_i) * y_i.T)
            term = torch.matmul(y_i.t(), torch.matmul(l_i, y_i))
            loss += torch.trace(term)
            
        # Normalize by batch size and nodes to keep loss magnitude reasonable
        return loss / (B * N * N)
