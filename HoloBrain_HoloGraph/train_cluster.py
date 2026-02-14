import argparse
import os
import logging
import random
import time
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from accelerate import Accelerator, utils
from torch.distributed.nn.functional import all_gather

from source.utils import LinearWarmupScheduler, create_logger, SpectralNetLoss
# from spectralnet._losses._spectralnet_loss import SpectralNetLoss
from source.data.create_dataset import create_dataset
from source.holograph_holobrain import HoloBrain 
from ema_pytorch import EMA  

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import SpectralClustering

def compute_purity(labels_true, labels_pred):
    matrix = contingency_matrix(labels_true, labels_pred)
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)

def train_unsupervised(model, ema, optimizer, scheduler, loader, epoch, device, log, accelerator):
    model.train()
    total_loss = 0.0
    criterion = SpectralNetLoss()
    
    for batch_idx, (features, adj, targets) in enumerate(loader):
        features = features.to(device)
        adj = adj.to(device)
        
        if features.dim() == 2:
            features = features.unsqueeze(0)
        
        optimizer.zero_grad()
        
        # HoloBrain forward: (logits_graph, x, y)
        _, x_features, y_features = model(features, features, adj)
        
        node_feats = y_features.transpose(1, 2)
        
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            outputs_node = model.module.out_pred_node(node_feats)
        else:
            outputs_node = model.out_pred_node(node_feats)

        if accelerator.num_processes > 1:
            outputs_node_gathered = torch.cat(all_gather(outputs_node), dim=0)
            adj_gathered = torch.cat(all_gather(adj), dim=0)
            loss = criterion(adj_gathered, outputs_node_gathered)
        else:
            loss = criterion(adj, outputs_node)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema.update()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(loader)
    if accelerator.is_main_process:
        log.info(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate_unsupervised(model, loader, device, log, accelerator):
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
            
            # Forward
            _, x_features, y_features = model(features, features, adj)
            
            node_feats = y_features.transpose(1, 2)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                outputs_node = model.module.out_pred_node(node_feats)
            else:
                outputs_node = model.out_pred_node(node_feats)

            # Gather for evaluation
            if accelerator.num_processes > 1:
                outputs_node = torch.cat(all_gather(outputs_node), dim=0)
                targets = torch.cat(all_gather(targets), dim=0)
                x_features = torch.cat(all_gather(x_features), dim=0)
                y_features = torch.cat(all_gather(y_features), dim=0)
                features = torch.cat(all_gather(features), dim=0)

            preds_list.append(outputs_node)
            targets_list.append(targets)
            x_feats_list.append(x_features)
            y_feats_list.append(y_features)
            inputs_list.append(features)
    
    preds = torch.cat(preds_list, dim=0) 
    targets = torch.cat(targets_list, dim=0)
    x_feats = torch.cat(x_feats_list, dim=0)
    y_feats = torch.cat(y_feats_list, dim=0)

    # Feature Concatenation [y + x]
    feats = torch.cat([y_feats.transpose(1, 2), x_feats.transpose(1, 2)], dim=-1)
    
    predictions_np = preds.detach().cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    nmi_score, purity = 0.0, 0.0
    try:
        sample_pred = predictions_np[0] # [Num_Nodes, K]
        sample_target = targets_np[0]   # [Num_Nodes] (For Parcellation Task)
        
        spectral_clustering = SpectralClustering(
            n_clusters=8, 
            affinity='nearest_neighbors',
            n_neighbors=10,
            assign_labels='kmeans',
            random_state=0
        )
        pred_labels = spectral_clustering.fit_predict(sample_pred) 
        nmi_score = normalized_mutual_info_score(sample_target, pred_labels)
        purity = compute_purity(sample_target, pred_labels)
    except Exception as e:
        # log.warning(f"Clustering eval failed: {e}")
        pass

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
    
    # HoloBrain Params
    parser.add_argument("--num_nodes", type=int, default=116, help="Mapped to brain region")
    parser.add_argument("--feature_dim", type=int, default=300) 
    parser.add_argument("--num_class", type=int, default=4, help="Clusters")
    parser.add_argument("--L", type=int, default=1)
    parser.add_argument("--h", type=int, default=256, help="Mapped to ch")
    parser.add_argument("--Q", type=int, default=8, help="Mapped to Q")
    parser.add_argument("--N", type=int, default=4, help="Oscillator dimension")
    parser.add_argument("--gamma", type=float, default=1.0, help="Step size")
    
    args = parser.parse_args()
    
    # Initialize Accelerator first
    accelerator = Accelerator()
    device = accelerator.device
    utils.set_seed(args.seed + accelerator.process_index)

    if accelerator.is_main_process:
        if not os.path.exists("logs_cluster"):
            os.makedirs("logs_cluster")
        log = create_logger("logs_cluster") 
        log.info("Init successfully")
    else:
        log = logging.getLogger(__name__)
        log.addHandler(logging.NullHandler())
    
    dataset = create_dataset(args.data)
    
    # Auto-detect Feature Dim
    try:
        if len(dataset) > 0:
            sample = dataset[0]
            if isinstance(sample, (tuple, list)):
                real_dim = sample[0].shape[0]
                if args.feature_dim != real_dim:
                    if accelerator.is_main_process:
                        log.info(f"Auto-detected feature_dim: {real_dim}")
                    args.feature_dim = real_dim
    except:
        pass

    loader = DataLoader(
        dataset,
        batch_size=args.batchsize,
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
            homo=False,             
            gst_total=4,            
            dropout=0.5
        ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = LinearWarmupScheduler(optimizer, warmup_iters=args.warmup_iters)
    ema = EMA(model, beta=args.ema_decay, update_every=10, update_after_step=200)
    
    # Prepare with Accelerator
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, loader, scheduler
    )

    if accelerator.is_main_process:
        log.info(f"Starting unsupervised training for {args.epochs} epochs...")
    
    best_nmi = 0.0
    
    for epoch in range(args.epochs):
        train_loss = train_unsupervised(model, ema, optimizer, scheduler, loader, epoch, device, log, accelerator)
        
        # Evaluation is heavy, maybe run less frequently
        if (epoch + 1) % 10 == 0:
            start_time = time.time()
            nmi, purity, features, inputs_data, gt = evaluate_unsupervised(model, loader, device, log, accelerator)
            
            if accelerator.is_main_process:
                if nmi > best_nmi:
                    best_nmi = nmi
                    torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(".", "model_best.pth"))
                log.info(f"Epoch {epoch+1}: NMI: {nmi:.4f}, Purity: {purity:.4f} (Best NMI: {best_nmi:.4f})")
    
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(".", "model_final.pth"))
    
if __name__ == "__main__":
    main()
