# train_brick_diffdata.py
import argparse
import os
import random
import time
import numpy as np

import torch
import torch.nn as nn
from torch import optim

from torch_geometric.datasets import WebKB, Actor, WikipediaNetwork
from torch_geometric.utils import to_dense_adj

from source.utils import LinearWarmupScheduler, compute_weighted_metrics, logger
from accelerate import utils
from source.brick import BRICK
from ema_pytorch import EMA


def train_one_epoch(model, ema, optimizer, scheduler, data, train_mask, device, epoch, logger):
    model.train()
    epoch_loss = 0.0
    
    criterion = nn.CrossEntropyLoss()
    data = data.to(device)
    inputs = data.x.unsqueeze(0)
    adj = to_dense_adj(data.edge_index)[0]
    if adj.dim() == 2:
        adj = adj.unsqueeze(0)

    outputs, x_feat, y_feat = model(inputs, adj)
    outputs = outputs.squeeze(0)[train_mask]
    targets = data.y.to(device)
    if targets.dim() == 2:
        targets = targets.squeeze(1)
    targets = targets[train_mask]

    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    ema.update()

    epoch_loss += loss.item()
    logger.info(f"[Epoch {epoch+1}] Training Loss: {epoch_loss:.4f}")
    return epoch_loss

def evaluate(model, data, train_mask, val_mask, test_mask, device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        inputs = data.x.unsqueeze(0) 
        adj = to_dense_adj(data.edge_index)[0]
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        
        outputs, x_feat, y_feat = model(inputs, adj)
        outputs = outputs.squeeze(0)
        targets = data.y.to(device)
        if targets.dim() == 2:
            targets = targets.squeeze(1)
        
        results = {}
        for mask_type, mask in zip(["val", "test"], [val_mask, test_mask]):
            preds = outputs[mask].argmax(dim=-1)
            metrics = compute_weighted_metrics(preds, targets[mask])
            results[mask_type] = {
                "accuracy": metrics[0],
                "precision": metrics[1],
                "f1": metrics[3]
            }
    return results

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

    parser.add_argument("--data", type=str, default="Cornell", help="Dataset name")
    parser.add_argument("--num_nodes", type=int, default=116, help="Number of nodes")
    parser.add_argument("--feature_dim", type=int, default=39, help="Input feature dimension")
    parser.add_argument("--num_class", type=int, default=4, help="Number of classes")
    parser.add_argument("--L", type=int, default=1, help="Number of Kuramoto sovlers")
    parser.add_argument("--h", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--T", type=int, default=8, help="Number of time steps")
    parser.add_argument("--N", type=int, default=4, help="oscillator dimensions")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta for Kuramoto solver")

    parser.add_argument("--use_pe", action="store_false", help="Use positional encoding")
    parser.add_argument("--node_cls", action="store_true", help="Node classification mode")
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

    datasets = {
        "Cornell": WebKB(root="data/Cornell", name="Cornell"),
        "Texas": WebKB(root="data/Texas", name="Texas"),
        "Wisconsin": WebKB(root="data/Wisconsin", name="Wisconsin"),
        "Chameleon": WikipediaNetwork(root="data/Chameleon", name="Chameleon"),
        "Squirrel": WikipediaNetwork(root="data/Squirrel", name="Squirrel"),
        "Actor": Actor(root="data/Actor"),
    }
    if args.data not in datasets:
        raise ValueError(f"Dataset {args.data} not supported. Available: {list(datasets.keys())}")

    logger.info(f"Running on dataset {args.data} ...")
    dataset_obj = datasets[args.data]
    data = dataset_obj[0]  
    data = data.to(device)

    metrics_summary = {"accuracy": [], "precision": [], "f1": []}

    num_splits = data.train_mask.shape[1] if hasattr(data, "train_mask") else 1
    for split_idx in range(num_splits):
        logger.info(f"Split {split_idx}:")
        train_mask = data.train_mask[:, split_idx]
        val_mask = data.val_mask[:, split_idx]
        test_mask = data.test_mask[:, split_idx]

        model_imsize = data.num_features if args.model_imsize is None else args.model_imsize
        net = BRICK(
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
        ).to(device)

        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {total_params:,}")

        optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0001)
        scheduler = LinearWarmupScheduler(optimizer, warmup_iters=args.warmup_iters)
        ema = EMA(net, beta=args.ema_decay, update_every=10, update_after_step=200)

        best_val_metrics = None
        best_test_metrics = None

        for epoch in range(args.epochs):
            loss = train_one_epoch(net, ema, optimizer, scheduler, data, train_mask, device, epoch, logger)
            t0 = time.time()
            results = evaluate(net, data, train_mask, val_mask, test_mask, device)
            elapsed_ms = (time.time() - t0) * 1000
            if best_val_metrics is None or results["val"]["accuracy"] > best_val_metrics["accuracy"]:
                best_val_metrics = results["val"]
                best_test_metrics = results["test"]

            if (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch+1:03d}, Loss: {loss:.4f}, Val Acc: {results['val']['accuracy']:.4f}, "
                    f"Test Acc: {results['test']['accuracy']:.4f}, "
                    f"Test Pre: {results['test']['precision']:.4f}, Test F1: {results['test']['f1']:.4f} "
                    f"(Inference time: {elapsed_ms:.2f} ms)"
                )

        metrics_summary["accuracy"].append(best_test_metrics["accuracy"])
        metrics_summary["precision"].append(best_test_metrics["precision"])
        metrics_summary["f1"].append(best_test_metrics["f1"])
        logger.info(f"Best Test Metrics for split {split_idx+1}: {best_test_metrics}")

    avg_acc = np.mean(metrics_summary["accuracy"])
    avg_pre = np.mean(metrics_summary["precision"])
    avg_f1 = np.mean(metrics_summary["f1"])
    logger.info(f"Final Results -- Average Test Acc: {avg_acc:.4f}, Precision: {avg_pre:.4f}, F1: {avg_f1:.4f}")
    logger.info(f"All splits Accuracies: {metrics_summary['accuracy']}")
    logger.info(f"All splits Precisions: {metrics_summary['precision']}")
    logger.info(f"All splits F1 scores: {metrics_summary['f1']}")

if __name__ == "__main__":
    main()
