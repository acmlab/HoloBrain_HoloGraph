import argparse
import os
import logging
import random
import time
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from accelerate import utils

from source.utils import LinearWarmupScheduler, logger
from source.data.create_dataset import create_dataset
from source.brick import BRICK  
from spectralnet._losses._spectralnet_loss import SpectralNetLoss
from ema_pytorch import EMA  

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import SpectralClustering

def compute_purity(labels_true, labels_pred):
    matrix = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)

def train_unsupervised(model, ema, optimizer, scheduler, loader, epoch, device, log):

    model.train()
    total_loss = 0.0
    criterion = SpectralNetLoss()
    
    for batch_idx, (features, adj, targets) in enumerate(loader):
        features = features.to(device)
        adj = adj.to(device)
        if features.dim() == 2:
            features = features.unsqueeze(0)
        optimizer.zero_grad()
        outputs, w, x_features, c_features = model(features, adj)
        loss = criterion(w, outputs) 
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    log.info(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate_unsupervised(model, loader, device, log, test_mode=True):
    model.eval()
    preds_list = []
    targets_list = []
    x_feats_list = []
    c_feats_list = []
    inputs_list = []
    
    with torch.no_grad():
        for batch_idx, (features, adj, targets) in enumerate(loader):
            features = features.to(device)
            adj = adj.to(device)
            if features.dim() == 2:
                features = features.unsqueeze(0)
            targets = targets.to(device)
            if targets.dim() == 2:
                targets = targets.squeeze(1)
            outputs, _, x_feature, c_feature = model(features, adj, adj, test=test_mode)
            preds_list.append(outputs)
            targets_list.append(targets)
            x_feats_list.append(x_feature)
            c_feats_list.append(c_feature)
            inputs_list.append(features)
    
    preds = torch.cat(preds_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    x_feats = torch.cat(x_feats_list, dim=0)
    c_feats = torch.cat(c_feats_list, dim=0)

    feats = torch.cat([c_feats.transpose(1, 2).unsqueeze(1), x_feats], dim=1)
    
    predictions_np = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    targets_np = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets
    
    spectral_clustering = SpectralClustering(
        n_clusters=8,  
        affinity='nearest_neighbors',
        n_neighbors=10,
        assign_labels='kmeans',
        random_state=0
    )
    pred_labels = spectral_clustering.fit_predict(predictions_np)
    nmi_score = normalized_mutual_info_score(targets_np, pred_labels)
    purity = compute_purity(targets_np, pred_labels)
    
    log.info(f"Evaluation: NMI: {nmi_score:.4f}, Purity: {purity:.4f}")
    return nmi_score, purity, feats, torch.cat(inputs_list, dim=0), targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="GPU id to use")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay factor")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--warmup_iters", type=int, default=10)
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data", type=str, default="HCP-A", help="Dataset name")
    parser.add_argument("--num_nodes", type=int, default=116, help="Number of nodes")
    parser.add_argument("--feature_dim", type=int, default=300, help="Input feature dimension")
    parser.add_argument("--num_class", type=int, default=4, help="Number of classes")
    parser.add_argument("--L", type=int, default=1, help="Number of Kuramoto solvers")
    parser.add_argument("--h", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--T", type=int, default=8, help="Number of time steps")
    parser.add_argument("--N", type=int, default=4, help="oscillator dimensions")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta for Kuramoto solver")
    parser.add_argument("--use_pe", action="store_true", help="Use positional encoding")
    parser.add_argument("--node_cls", action="store_false", help="Node classification mode")
    parser.add_argument("--y_type", type=str, default="linear", choices=["conv", "linear"], help="y computation type ")
    parser.add_argument("--mapping_type", type=str, default="conv", choices=["conv", "gconv"], help="Mapping type for y")
    parser.add_argument("--parcellation", action="store_false", help="Implement parcellation or not")
    
    args = parser.parse_args()
    utils.set_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger = logger()
    logger.info("Init successfully")
    
    dataset = create_dataset(args.data)
    loader = DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    model = BRICK(
        N=args.N,
        hidden_dim=args.h,
        L=args.L,
        T=args.T,
        num_class=args.num_class,
        beta=args.beta,
        feature_dim=args.feature_dim,
        num_nodes=args.num_nodes,
        use_pe=args.use_pe,
        node_cls=args.node_cls,
        y_type=args.y_type,
        mapping_type=args.mapping_type,
        parcellation=args.parcellation,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = LinearWarmupScheduler(optimizer, warmup_iters=args.warmup_iters)
    ema = EMA(model, beta=args.ema_decay, update_every=10, update_after_step=200)
    
    logger.info(f"Starting unsupervised training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss = train_unsupervised(model, ema, optimizer, scheduler, loader, epoch, device, logger)
        start_time = time.time()
        metrics, features, inputs_data, gt = evaluate_unsupervised(model, loader, device, logger)
        elapsed_ms = (time.time() - start_time) * 1000 / len(gt)
        test_acc, pre, rec, f1 = metrics
        logger.info(f"Epoch {epoch+1}: 
                    Test Acc: {test_acc:.4f}, 
                    Precision: {pre:.4f}, 
                    Recall: {rec:.4f}, 
                    F1: {f1:.4f} 
                    (Avg inference time: {elapsed_ms:.2f} ms)")
    
    torch.save(model.state_dict(), os.path.join(".", "model_final.pth"))
    torch.save(ema.state_dict(), os.path.join(".", "ema_model_final.pth"))
    
if __name__ == "__main__":
    main()
