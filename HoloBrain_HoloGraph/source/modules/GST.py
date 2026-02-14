import torch
import torch.nn as nn

def diffusion_P_from_L_heat(W, tau=1.0, normalized=True, eps=1e-12):
    # W: [B,N,N]
    B, N, _ = W.shape
    W = 0.5*(W + W.transpose(-1,-2))
    deg = W.sum(-1).clamp_min(eps)
    D = torch.diag_embed(deg)

    if normalized:
        Dm12 = torch.diag_embed(torch.rsqrt(deg))
        L = torch.eye(N, device=W.device).unsqueeze(0).expand(B,-1,-1) - Dm12 @ W @ Dm12
    else:
        L = D - W

    # P = exp(-tau L)
    P = torch.matrix_exp(-tau * L)
    return P, L
class GSTWavelet(nn.Module):
    """
    Clean GST implementation aligned with paper Eq.(4):

    Lazy random walk:
        P = 0.5 * (I + W D^{-1})

    Wavelets:
        psi^0 = I - P
        psi^h = P^{2^{h-1}} - P^{2^{h}}   for h>=1

    Low-pass:
        Phi = P^{2^J}

    Scattering (windowed):
        Apply wavelets at each layer, take abs, collect outputs,
        then apply Phi on concatenated basis.
    """
    def __init__(
        self,
        wavelet_orders=(0, 1, 2),
        level=2,
        J_lowpass=None,
        eps=1e-12,
        make_symmetric=True,
        add_self_loops_if_isolated=True,
    ):
        super().__init__()
        self.wavelet_orders = [int(o) for o in wavelet_orders]
        self.level = int(level)
        self.eps = float(eps)
        self.make_symmetric = bool(make_symmetric)
        self.add_self_loops_if_isolated = bool(add_self_loops_if_isolated)
        self.J_lowpass = int(J_lowpass) if J_lowpass is not None else max(self.wavelet_orders)

    # ---------- core math ----------
    def _sanitize_W(self, W):
        """
        W: [B,N,N] dense adjacency (can be weighted).
        Minimal fixes:
          - optional symmetrize
          - if isolated nodes exist: add self-loops
        """
        if self.make_symmetric:
            W = 0.5 * (W + W.transpose(-1, -2))

        if self.add_self_loops_if_isolated:
            deg = W.sum(dim=-1)  # [B,N]
            isolated = deg <= self.eps
            if isolated.any():
                B, N, _ = W.shape
                eye = torch.eye(N, device=W.device, dtype=W.dtype).unsqueeze(0).expand(B, -1, -1)
                # set diagonal to 1 only at isolated nodes
                # mask: [B,N] -> [B,N,1] -> broadcast on diag via eye
                mask = isolated.unsqueeze(-1)
                W = W + eye * mask.to(W.dtype)

        return W

    def _lazy_random_walk_P(self, W):
        """
        Paper Eq.(4):
            P = 0.5 * (I + W D^{-1})
        """
        B, N, _ = W.shape
        deg = W.sum(dim=-1).clamp_min(self.eps)   # [B,N]
        Dinv = torch.diag_embed(1.0 / deg)        # [B,N,N]
        I = torch.eye(N, device=W.device, dtype=W.dtype).unsqueeze(0).expand(B, -1, -1)
        P = 0.5 * (I + torch.bmm(W, Dinv))
        return P

    def _heat_kernel_P(self, W, tau=1.0, normalized=True):
        P, L = diffusion_P_from_L_heat(W, tau=tau, normalized=normalized, eps=self.eps)
        return P, L


    # def _lazy_random_walk_P(self, W):
    #     P, L = diffusion_P_from_L_heat(W, tau=1.0, normalized=True, eps=self.eps)
    #     return P
    def construct_wavelet(self, adj):
        """
        adj: [B,N,N] or [N,N]
        Returns:
          wavelets: list of [B,N,N] matrices [psi^order]
          low_pass: [B,N,N] = P^(2^J_lowpass)
          P: [B,N,N] (optional useful for debug)
        """
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        elif adj.dim() != 3:
            raise ValueError(f"adj must be [N,N] or [B,N,N], got {tuple(adj.shape)}")

        W = adj.float()
        W = self._sanitize_W(W)
        P = self._lazy_random_walk_P(W) # use W directly
        # P, L = diffusion_P_from_L_heat(W, tau=1.0, normalized=True, eps=self.eps) # use Laplacian

        B, N, _ = P.shape
        I = torch.eye(N, device=P.device, dtype=P.dtype).unsqueeze(0).expand(B, -1, -1)

        # Precompute P^(2^h) for needed h
        max_h = max(max(self.wavelet_orders), self.J_lowpass)
        P_pows = {0: P}  # P^(2^0) = P
        cur = P
        for h in range(1, max_h + 1):
            cur = torch.bmm(cur, cur)     # square -> P^(2^h)
            P_pows[h] = cur

        wavelets = []
        for order in self.wavelet_orders:
            if order == 0:
                psi = I - P
            else:
                # psi^h = P^(2^(h-1)) - P^(2^h)
                psi = P_pows[order - 1] - P_pows[order]
            wavelets.append(psi)

        # Phi = P^(2^J)
        low_pass = P_pows[self.J_lowpass]
        return wavelets, low_pass, P

    # ---------- scattering ----------
    def windowed(self, x, adj):
        """
        x: [B,F,N] 
        adj: [B,N,N] or [N,N]
        Returns:
          scattering_coeff: [B,N,F,dim] where dim depends on level and wavelet_orders
        """
        wavelets, Phi, _ = self.construct_wavelet(adj)

        # Start with node-signal in shape [B,N,F]
        if x.dim() != 3:
            raise ValueError(f"x must be [B,F,N], got {tuple(x.shape)}")
        x0 = x.transpose(1, 2).contiguous()  # [B,N,F]

        outputs = [[x0]]  # list of layers, each layer is list of tensors [B,N,F]
        for _layer in range(self.level):
            layer_output = []
            for inp in outputs[-1]:
                for psi in wavelets:
                    out = torch.matmul(psi, inp)  # [B,N,N] @ [B,N,F] -> [B,N,F]
                    out = torch.abs(out)
                    layer_output.append(out)
            outputs.append(layer_output)

        # basis: concat all layers along "feature bank" dimension
        # each layer: stack(..., dim=-1) -> [B,N,F,K]
        # then cat over layers -> [B,N,F,dim]
        basis = torch.cat([torch.stack(layer, dim=-1) for layer in outputs], dim=-1)

        # Apply low-pass Phi on node dimension for each basis channel
        B, N, F, D = basis.shape
        basis_flat = basis.view(B, N, F * D)              # [B,N,F*D]
        sc_flat = torch.matmul(Phi, basis_flat)           # [B,N,N]@[B,N,F*D] -> [B,N,F*D]
        scattering = sc_flat.view(B, N, F, D)             # [B,N,F,D]
        return scattering

    def nonwindowed(self, x, adj):
        """
        return the multi-layer basis without low-pass.
        """
        wavelets, _, _ = self.construct_wavelet(adj)
        x0 = x.transpose(1, 2).contiguous()  # [B,N,F]
        outputs = [[x0]]
        for _layer in range(self.level):
            layer_output = []
            for inp in outputs[-1]:
                for psi in wavelets:
                    out = torch.matmul(psi, inp)
                    out = torch.abs(out)
                    layer_output.append(out)
            outputs.append(layer_output)
        basis = torch.cat([torch.stack(layer, dim=-1) for layer in outputs], dim=-1)  # [B,N,F,dim]
        return basis

    def forward(self, x, adj, windowed=True):
        if windowed:
            return self.windowed(x, adj)
        return self.nonwindowed(x, adj)
