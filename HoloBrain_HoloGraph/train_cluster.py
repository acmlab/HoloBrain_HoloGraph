import argparse
import os
import logging
import random
import time
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.distributed.nn.functional import all_gather

import accelerate
from accelerate import Accelerator

from ema_pytorch import EMA

from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

from source.utils import LinearWarmupScheduler
from source.data.create_dataset import create_dataset

from spectralnet._losses._spectralnet_loss import SpectralNetLoss

from source.holograph_holobrain import HoloBrain

def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
    )
    return logging.getLogger(__name__)


def compute_purity(labels_true, labels_pred):
    matrix = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)


def _unpack_batch(batch):

    if len(batch) == 3:
        features, adj, target = batch
        return features, adj, target
    if len(batch) == 4:
        features, _, adj, target = batch
        return features, adj, target
    raise ValueError(f"Unexpected batch length={len(batch)}")


def _to_batched_dense_adj(adj):

    if adj.dim() == 2:
        return adj.unsqueeze(0)
    if adj.dim() == 3:
        return adj
    raise ValueError(f"adj must be [N,N] or [B,N,N], got {tuple(adj.shape)}")


def train_one_epoch(net, ema, opt, scheduler, loader, epoch, accelerator, logger):
    net.train()
    loss_fn = SpectralNetLoss()
    running = 0.0

    for batch in loader:
        features, adj, target = _unpack_batch(batch)

        features = features.to(accelerator.device)
        adj = adj.to(accelerator.device)

        if features.dim() == 2:
            features = features.unsqueeze(0)

        w_dense = _to_batched_dense_adj(adj).float()  # [B,N,N]

        opt.zero_grad(set_to_none=True)

        # HoloBrain forward: logits [B,N,K], x [B,N,ch], y [B,ch,N]
        # logits, x_feature, y_feature = net(features, features, w_dense)

        # outputs = logits  # [B,N,K]

        if accelerator.num_processes > 1:
            outputs = torch.cat(all_gather(outputs), dim=0)   # [B_all,N,K]
            w_dense = torch.cat(all_gather(w_dense), dim=0)   # [B_all,N,N]

        _, x_feature, y_feature = net(features, features, w_dense)

        # node-level outputs: [B,N,K]
        node_feats = y_feature.transpose(1, 2).contiguous()   # [B,N,C]
        if hasattr(net, "module"):  # DDP
            outputs = net.module.out_pred_node(node_feats)
        else:
            outputs = net.out_pred_node(node_feats)

        if accelerator.num_processes > 1:
            outputs = torch.cat(all_gather(outputs), dim=0)   # [B_all,N,K]
            w_dense = torch.cat(all_gather(w_dense), dim=0)   # [B_all,N,N]

        out_2d = outputs[0]     # [N,K] 
        w_2d = w_dense[0]       # [N,N]
        N = w_2d.size(-1)

        loss = loss_fn(w_2d, out_2d) / float(N)

        accelerator.backward(loss)
        opt.step()
        scheduler.step()
        ema.update()

        running += loss.item()

    avg = running / len(loader)
    if accelerator.is_main_process:
        logger.info(f"[Epoch {epoch+1}] loss: {avg:.4f}")
    return avg


@torch.no_grad()
def evaluate(net, loader, accelerator):
    net.eval()

    preds_list, gts_list = [], []
    x_list, y_list, inp_list = [], [], []

    for batch in loader:
        features, adj, target = _unpack_batch(batch)

        features = features.to(accelerator.device)
        adj = adj.to(accelerator.device)
        target = target.to(accelerator.device)

        if features.dim() == 2:
            features = features.unsqueeze(0)
        if target.dim() == 2:
            target = target.squeeze(1)

        w_dense = _to_batched_dense_adj(adj).float()

        logits, x_feature, y_feature = net(features, features, w_dense)

        if accelerator.num_processes > 1:
            logits = torch.cat(all_gather(logits), dim=0)
            target = torch.cat(all_gather(target), dim=0)
            x_feature = torch.cat(all_gather(x_feature), dim=0)
            y_feature = torch.cat(all_gather(y_feature), dim=0)
            features = torch.cat(all_gather(features), dim=0)

        preds_list.append(logits)
        gts_list.append(target)
        x_list.append(x_feature)
        y_list.append(y_feature)
        inp_list.append(features)

    preds = torch.cat(preds_list, dim=0)   # [B,N,K]
    gts = torch.cat(gts_list, dim=0)
    # x_feats = torch.cat(x_list, dim=0)     # [B,N,ch]
    # y_feats = torch.cat(y_list, dim=0)     # [B,ch,N]
    # inputs = torch.cat(inp_list, dim=0)

    # y_bnch = y_feats.transpose(1, 2).contiguous()  # [B,N,ch]
    # feats = torch.cat([y_bnch, x_feats], dim=-1)   # [B,N,2ch]
    x_feats = torch.cat(x_list, dim=0)
    y_feats = torch.cat(y_list, dim=0)

    # y: [B,ch,N] -> [B,N,ch]
    y_bnch = y_feats.transpose(1, 2).contiguous()
    N = y_bnch.size(1)

    if x_feats.dim() == 3:
        if x_feats.size(1) == N:
            x_bnch = x_feats.contiguous()              # already [B,N,ch]
        elif x_feats.size(2) == N:
            x_bnch = x_feats.transpose(1, 2).contiguous()  # [B,ch,N] -> [B,N,ch]
        else:
            raise RuntimeError(f"x_feats shape not compatible: {tuple(x_feats.shape)}, expected N={N}")
    else:
        raise RuntimeError(f"x_feats must be 3D, got {tuple(x_feats.shape)}")

    feats = torch.cat([y_bnch, x_bnch], dim=-1)   # [B,N,2ch]

    sample_pred = preds[0].detach().cpu().numpy()  # [N,K]
    sample_gt = gts[0].detach().cpu().numpy() if gts.dim() > 1 else gts.detach().cpu().numpy()
    inputs = torch.cat(inp_list, dim=0)
    nmi, purity = 0.0, 0.0
    try:
        sc = SpectralClustering(
            n_clusters=8,
            affinity="nearest_neighbors",
            n_neighbors=10,
            assign_labels="kmeans",
            random_state=0,
        )
        pred_labels = sc.fit_predict(sample_pred)
        nmi = normalized_mutual_info_score(sample_gt, pred_labels)
        purity = compute_purity(sample_gt, pred_labels)
    except Exception:
        pass

    return nmi, purity, feats, inputs, gts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data", type=str, default="HCP-A")

    # HoloBrain params
    parser.add_argument("--num_nodes", type=int, default=116)
    parser.add_argument("--feature_dim", type=int, default=300)
    parser.add_argument("--num_class", type=int, default=4)
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--h", type=int, default=256)
    parser.add_argument("--Q", type=int, default=8)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--J", type=str, default="adj")  # dense-only
    args = parser.parse_args()

    accelerator = Accelerator()
    accelerate.utils.set_seed(args.seed + accelerator.process_index)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if accelerator.is_main_process:
        os.makedirs("runs_cluster_safe", exist_ok=True)
        logger = create_logger("runs_cluster_safe")
        logger.info("Init successfully")
        logger.info(f"Starting training for {args.epochs} epochs...")
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset = create_dataset(args.data)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batchsize // accelerator.num_processes),
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = HoloBrain(
        n=args.N,
        ch=args.h,
        L=args.L,
        Q=args.Q,
        num_class=args.num_class,
        feature_dim=args.feature_dim,
        num_nodes=args.num_nodes,
        gamma=args.gamma,
        J=args.J,         # "adj"
        homo=False,
        gst_total=4,
        dropout=0.5,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = LinearWarmupScheduler(optimizer, warmup_iters=args.warmup_iters)
    ema = EMA(model, beta=args.ema_decay, update_every=10, update_after_step=200)

    best_nmi = -1.0
    for epoch in range(args.epochs):
        model.to(accelerator.device)

        train_one_epoch(model, ema, optimizer, scheduler, loader, epoch, accelerator, logger)

        nmi, purity, feats, inputs, gts = evaluate(model, loader, accelerator)

        if accelerator.is_main_process:
            logger.info(f"Epoch {epoch+1}: NMI={nmi:.4f}, Purity={purity:.4f}")
            if nmi > best_nmi:
                best_nmi = nmi
                torch.save(model.state_dict(), "runs_cluster_safe/model_best.pth")

    if accelerator.is_main_process:
        torch.save(model.state_dict(), "runs_cluster_safe/model_final.pth")


if __name__ == "__main__":
    main()
