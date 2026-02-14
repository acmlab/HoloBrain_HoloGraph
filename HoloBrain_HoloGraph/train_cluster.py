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

from source.utils import LinearWarmupScheduler, create_logger, SpectralNetLoss
# from spectralnet._losses._spectralnet_loss import SpectralNetLoss
from source.data.create_dataset import create_dataset
from source.holograph import HoloGraph 
from ema_pytorch import EMA  

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import SpectralClustering

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
        force=True # Force re-config to avoid conflicts
    )
    logger = logging.getLogger(__name__)
    return logger

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
        
        # HoloGraph expects [B, N, F] or [B, F, N]. Ensure batch dim.
        if features.dim() == 2:
            features = features.unsqueeze(0)
        
        optimizer.zero_grad()
        
        # HoloGraph forward signature: returns logits, x (oscillator), y (control/memory)
        outputs, x_features, y_features = model(features, features, adj)
        
        # Unpack list if necessary (HoloGraph returns [logits])
        if isinstance(outputs, list):
            outputs = outputs[0]
            
        # Spectral Loss: (Affinity Matrix, Embeddings)
        # Using 'adj' as the Affinity Matrix 'w'
        loss = criterion(adj, outputs) 
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    log.info(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate_unsupervised(model, loader, device, log):
    model.eval()
    preds_list = []
    targets_list = []
    x_feats_list = []
    y_feats_list = [] 
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
            
            # HoloGraph Inference
            outputs, x_feature, y_feature = model(features, features, adj)
            
            if isinstance(outputs, list):
                outputs = outputs[0]

            preds_list.append(outputs)
            targets_list.append(targets)
            x_feats_list.append(x_feature)
            y_feats_list.append(y_feature)
            inputs_list.append(features)
    
    preds = torch.cat(preds_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    x_feats = torch.cat(x_feats_list, dim=0)
    y_feats = torch.cat(y_feats_list, dim=0)

    # Feature Concatenation for Analysis [y + x]
    feats = torch.cat([y_feats.transpose(1, 2), x_feats.transpose(1, 2)], dim=-1)
    
    predictions_np = preds.detach().cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Perform Spectral Clustering
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
    
    return nmi_score, purity, feats, torch.cat(inputs_list, dim=0), targets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_iters", type=int, default=10)
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data", type=str, default="HCP-A", help="Dataset name")
    
    # HoloGraph Specific Args
    parser.add_argument("--num_nodes", type=int, default=116, help="Mapped to imsize")
    parser.add_argument("--feature_dim", type=int, default=300)
    parser.add_argument("--num_class", type=int, default=4, help="Clusters")
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--h", type=int, default=256, help="Mapped to ch")
    parser.add_argument("--T", type=int, default=8, help="Mapped to Q")
    parser.add_argument("--N", type=int, default=4, help="Oscillator dimension")
    parser.add_argument("--gamma", type=float, default=1.0, help="Step size")
    
    args = parser.parse_args()
    
    utils.set_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    log = create_logger("logs") 
    log.info("Init successfully")
    
    dataset = create_dataset(args.data)
    loader = DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Initialize HoloGraph
    model = HoloGraph(
        n=args.N,
        ch=args.h,              # args.h -> ch
        L=args.L,
        Q=args.T,               # args.T -> Q
        num_class=args.num_class,
        feature_dim=args.feature_dim,
        imsize=args.num_nodes,  # Brain graph size
        gamma=args.gamma,
        homo=False,             # Brain networks are non-homophilic
        gst_total=4,            
        dropout=0.5
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total trainable parameters: {total_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = LinearWarmupScheduler(optimizer, warmup_iters=args.warmup_iters)
    ema = EMA(model, beta=args.ema_decay, update_every=10, update_after_step=200)
    
    log.info(f"Starting unsupervised training for {args.epochs} epochs...")
    
    best_nmi = 0.0
    
    for epoch in range(args.epochs):
        train_loss = train_unsupervised(model, ema, optimizer, scheduler, loader, epoch, device, log)
        
        start_time = time.time()
        # Evaluate
        nmi, purity, features, inputs_data, gt = evaluate_unsupervised(model, loader, device, log)
        elapsed_ms = (time.time() - start_time) * 1000 / len(gt)
        
        if nmi > best_nmi:
            best_nmi = nmi
            torch.save(model.state_dict(), os.path.join(".", "model_best.pth"))
            
        log.info(f"Epoch {epoch+1}: NMI: {nmi:.4f}, Purity: {purity:.4f} (Best NMI: {best_nmi:.4f})")
    
    torch.save(model.state_dict(), os.path.join(".", "model_final.pth"))
    
if __name__ == "__main__":
    main()
