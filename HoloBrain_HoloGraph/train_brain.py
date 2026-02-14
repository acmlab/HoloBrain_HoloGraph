import argparse
import os
import logging
import random
import time
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold

from accelerate import Accelerator, utils
from torch.distributed.nn.functional import all_gather

from source.utils import LinearWarmupScheduler, compute_weighted_metrics
from source.data.create_dataset import create_dataset
from source.holograph import HoloGraph 
from ema_pytorch import EMA  

def create_logger(logging_dir):
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

def train_one_epoch(model, ema, optimizer, scheduler, train_loader, epoch, device, accelerator, logger):
    model.train()
    total_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (features, adj, targets) in enumerate(train_loader):
        features = features.to(device)
        adj = adj.to(device)
        
        # Ensure batch dim [B, N, F]
        if features.dim() == 2:
            features = features.unsqueeze(0)
            
        targets = targets.to(device)
        targets = targets.squeeze(1) if targets.dim() == 2 else targets

        optimizer.zero_grad()
        
        outputs, x_features, y_features = model(features, features, adj)
        
        if isinstance(outputs, list):
            outputs = outputs[0]

        if accelerator.num_processes > 1:
            outputs = torch.cat(all_gather(outputs), dim=0)
            targets = torch.cat(all_gather(targets), dim=0)

        loss = criterion(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        ema.update() 

    avg_loss = total_loss / len(train_loader)
    if accelerator.is_main_process:
        logger.info(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, accelerator, test_loader, device, logger):
    model.eval()
    all_preds = []
    all_targets = []
    all_x_feats = []
    all_y_feats = []
    all_inputs = []
    
    with torch.no_grad():
        for batch_idx, (features, adj, targets) in enumerate(test_loader):
            features = features.to(device)
            adj = adj.to(device)
            if features.dim() == 2:
                features = features.unsqueeze(0)
            targets = targets.to(device)
            targets = targets.squeeze(1) if targets.dim() == 2 else targets

            # HoloBrain Inference
            outputs, x_features, y_features = model(features, features, adj)
            
            if isinstance(outputs, list):
                outputs = outputs[0]

            if accelerator.num_processes > 1:
                outputs = torch.cat(all_gather(outputs), dim=0)
                targets = torch.cat(all_gather(targets), dim=0)
                x_features = torch.cat(all_gather(x_features), dim=0)
                y_features = torch.cat(all_gather(y_features), dim=0)
                features = torch.cat(all_gather(features), dim=0)

            preds = outputs.argmax(dim=-1)
            all_preds.append(preds)
            all_targets.append(targets)
            all_x_feats.append(x_features)
            all_y_feats.append(y_features)
            all_inputs.append(features)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    

    y_feats_t = torch.cat(all_y_feats, dim=0).transpose(1, 2) 
    x_feats_t = torch.cat(all_x_feats, dim=0).transpose(1, 2)
    
    all_feats = torch.cat([y_feats_t, x_feats_t], dim=-1)
    
    metrics = compute_weighted_metrics(all_preds, all_targets)
    return metrics, all_feats, torch.cat(all_inputs, dim=0), all_targets


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

    parser.add_argument("--data", type=str, default="HCP-YA", help="Dataset name")
    
    # HoloBrain Params
    parser.add_argument("--num_nodes", type=int, default=116, help="Number of nodes (imsize)")
    parser.add_argument("--feature_dim", type=int, default=175, help="Input feature dimension")
    parser.add_argument("--num_class", type=int, default=4, help="Number of classes")
    parser.add_argument("--L", type=int, default=1, help="Number of Kuramoto solvers")
    parser.add_argument("--h", type=int, default=256, help="Hidden dimension (ch)")
    parser.add_argument("--T", type=int, default=8, help="Number of time steps (Q)")
    parser.add_argument("--N", type=int, default=4, help="Oscillator dimensions")
    parser.add_argument("--gamma", type=float, default=1.0, help="Step size")
    
    # Removed irrelevant args (beta, use_pe, node_cls, y_type, etc.)
    
    args = parser.parse_args()
    
    utils.set_seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    accelerator = Accelerator()
    device = accelerator.device

    # Initialize Logger
    log_dir = "logs_brain"
    logger = create_logger(log_dir)
    logger.info("Init successfully")
    
    dataset = create_dataset(args.data)

    splits = 5
    kfold = KFold(n_splits=splits, shuffle=True, random_state=args.seed)
    all_fold_acc = []
    all_fold_pre = []
    all_fold_f1 = []

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        logger.info(f"Fold {fold_idx}:")
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        
        if accelerator.is_main_process:
            logger.info(f"Train samples: {len(train_subset):,}, Test samples: {len(test_subset):,}")
            
        train_loader = DataLoader(
            train_subset,
            batch_size=args.batchsize // accelerator.num_processes,
            shuffle=True,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_subset,
            batch_size=args.batchsize // accelerator.num_processes,
            shuffle=False,
            num_workers=args.num_workers,
        )

        # [Change 5]: Initialize HoloGraph
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
            gst_total=4,            # Default value
            dropout=0.5
        ).to(device)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {total_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
        scheduler = LinearWarmupScheduler(optimizer, warmup_iters=args.warmup_iters)
        ema = EMA(model, beta=args.ema_decay, update_every=10, update_after_step=200)

        if accelerator.is_main_process:
            logger.info(f"Starting training for {args.epochs} epochs...")

        best_test_acc, best_pre, best_f1 = 0, 0, 0
        
        for epoch in range(args.epochs):
            epoch_loss = train_one_epoch(model, ema, optimizer, scheduler, train_loader, epoch, device, accelerator, logger)
            
            start_time = time.time()
            metrics, features, inputs_data, gt = evaluate(model, accelerator, test_loader, device, logger)
            elapsed_ms = (time.time() - start_time) * 1000 / len(gt)
            
            test_acc, pre, rec, f1 = metrics
            
            logger.info(f"Epoch {epoch+1}: Test Acc: {test_acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f} "
                        f"(Avg inference time: {elapsed_ms:.2f} ms)")
            
            if test_acc > best_test_acc:
                best_test_acc, best_pre, best_f1 = test_acc, pre, f1
                # Optional: Save numpy files for analysis
                # np.save(os.path.join(".", f"fold_{fold_idx}_features.npy"), features.cpu().detach().numpy())
                # np.save(os.path.join(".", f"fold_{fold_idx}_inputs.npy"), inputs_data.cpu().detach().numpy())
                # np.save(os.path.join(".", f"fold_{fold_idx}_gt.npy"), gt.cpu().detach().numpy())

        if accelerator.is_main_process:
            torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(".", "model.pth"))
            torch.save(ema.state_dict(), os.path.join(".", "ema_model.pth"))

        logger.info(f"Fold {fold_idx}: Best Test Acc: {best_test_acc:.4f}, Precision: {best_pre:.4f}, F1: {best_f1:.4f}")
        all_fold_acc.append(best_test_acc)
        all_fold_pre.append(best_pre)
        all_fold_f1.append(best_f1)

    avg_acc = np.mean(all_fold_acc)
    avg_pre = np.mean(all_fold_pre)
    avg_f1 = np.mean(all_fold_f1)
    logger.info(f"Final Results -- Average Test Acc: {avg_acc:.4f}, Precision: {avg_pre:.4f}, F1: {avg_f1:.4f}")
    logger.info(f"All folds Accuracies: {all_fold_acc}")
    logger.info(f"All folds Precisions: {all_fold_pre}")
    logger.info(f"All folds F1 scores: {all_fold_f1}")


if __name__ == "__main__":
    main()
