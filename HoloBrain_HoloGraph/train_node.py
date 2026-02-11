import argparse
import os
import logging
import copy
import numpy as np

import torch
import torch.nn as nn
from torch import optim

from source.training_utils import LinearWarmupScheduler


import accelerate
from accelerate import Accelerator

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WebKB, Actor, WikipediaNetwork, Planetoid
from source.layers.common_layers_node import compute_weighted_metrics
from source.models.net_node import HoloGraph
from ema_pytorch import EMA


def create_logger(logging_dir: str):
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt"),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger

def str2bool(x):
    if isinstance(x, bool):
        return x
    x = x.lower()
    if x[0] in ["0", "n", "f"]:
        return False
    elif x[0] in ["1", "y", "t"]:
        return True
    raise ValueError("Invalid value: {}".format(x))

def build_arg_parser():

    
    parser = argparse.ArgumentParser()

    # Training options
    parser.add_argument("--gpu", type=str, default='0')
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--exp_name", type=str, help="expname")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--beta", type=float, default=0.999, help="ema decay")
    parser.add_argument("--epochs", type=int, default=300, help="num of epochs")
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=50,
        help="save checkpoint every specified epochs",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="lr")
    parser.add_argument("--warmup_iters", type=int, default=20)
    parser.add_argument(
        "--finetune",
        type=str,
        default=None,
        help="path to the checkpoint. Training starts from that checkpoint",
    )

    # Data loading
    parser.add_argument("--data", type=str, default="Cora")
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--data_imsize",
        type=int,
        default=None,
        help="Image size. If None, use the default size of each dataset",
    )

    # General model options
    parser.add_argument("--L", type=int, default=1, help="num of layers")
    parser.add_argument("--ch", type=int, default=256, help="num of channels")
    parser.add_argument(
        "--imsize",
        type=int,
        default=None,
        help=(
            "Model's imsize. This is used when you want finetune a pretrained model "
            "that was trained on images with different resolution than the finetune image dataset."
        ),
    )
    parser.add_argument("--ksize", type=int, default=1, help="kernel size")
    parser.add_argument("--Q", type=int, default=8, help="num of recurrence")
    parser.add_argument("--num_class", type=int, default=6, help="num of class")
    parser.add_argument("--feature_dim", type=int, default=None, help="num of channels of features")
    parser.add_argument(
        "--maxpool", type=str2bool, default=True, help="max pooling or avg pooling"
    )
    parser.add_argument(
        "--heads", type=int, default=8, help="num of heads in self-attention"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")


    parser.add_argument("--N", type=int, default=4, help="num of rotating dimensions")
    parser.add_argument("--gamma", type=float, default=1.0, help="step size")
    parser.add_argument("--J", type=str, default="attn", help="connectivity")
    parser.add_argument("--use_omega", type=str2bool, default=False)
    parser.add_argument("--global_omg", type=str2bool, default=False)
    parser.add_argument(
        "--c_norm",
        type=str,
        default="sandb",
        help="normalization. gn(GroupNorm), sandb(scale and bias), or none",
    )
    parser.add_argument(
        "--init_omg", type=float, default=0.01, help="initial omega length"
    )
    parser.add_argument("--learn_omg", type=str2bool, default=True)
    parser.add_argument("--homo", type=str2bool, default=True)

    # Extra training options (from train_node1.py)
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay factor")
    parser.add_argument(
        "--use_scheduler", type=str2bool, default=True, help="Use learning rate scheduler"
    )
    return parser


def run_planetoid_public(args, accelerator: Accelerator, device: torch.device):


    datasets = {
        "Cora": Planetoid(root="data/Planetoid", name="Cora", split="public"),
        "Citeseer": Planetoid(root="data/Planetoid", name="Citeseer", split="public"),
        "Pubmed": Planetoid(root="data/Planetoid", name="Pubmed", split="public"),
    }

    if args.data not in datasets:
        raise ValueError(f"Dataset {args.data} is not a Planetoid public-split dataset")

    dataset = datasets[args.data]
    data = dataset[0].to(device)

    def train_fn(net, ema, opt, scheduler, data, epoch, train_mask, val_mask, test_mask):
        net.train()
        running_loss = 0.0

        inputs = data.x.T
        sc = data.edge_index
        target = data.y.squeeze()
        inputs = inputs.unsqueeze(0) if inputs.dim() == 2 else inputs

        outputs, x_fea, c_fea = net(inputs, sc, sc)

        target = target.squeeze(1) if target.dim() == 2 else target

        outputs_train = outputs.squeeze(0)[train_mask]
        target_train = target[train_mask]

        loss = nn.CrossEntropyLoss()(outputs_train, target_train.to(device))

        opt.zero_grad()
        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()
        if args.use_scheduler:
            scheduler(epoch)
        ema.update()

        running_loss += loss.item()
        return running_loss

    def test_fn(net, data, train_mask, val_mask, test_mask):
        net.eval()

        inputs = data.x.T
        sc = data.edge_index
        inputs = inputs.unsqueeze(0) if inputs.dim() == 2 else inputs
        target = data.y.squeeze()
        target = target.squeeze(1) if target.dim() == 2 else target

        outputs, _, _ = net(inputs, sc, sc)

        results = {}
        for mask_name, mask in zip(["val", "test"], [val_mask, test_mask]):
            pred = outputs.squeeze(0)[mask].argmax(dim=-1)
            gt = target[mask]
            acc, pre, recall, f1 = compute_weighted_metrics(pred, gt)
            results[mask_name] = {
                "accuracy": acc,
                "precision": pre,
                "f1": f1,
            }

        return results, None, target

    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask

    assert train_mask.dim() == 1, "Expected 1D mask for standard split"

    if args.data == "Cora":
        args.num_class = 7
        args.feature_dim = data.x.shape[1]
    elif args.data == "Citeseer":
        args.num_class = 6
        args.feature_dim = data.x.shape[1]
    elif args.data == "Pubmed":
        args.num_class = 3
        args.feature_dim = data.x.shape[1]

    print(f"Standard semi-supervised split:")
    print(f"  Training nodes: {train_mask.sum().item()} ({train_mask.sum().item()/data.num_nodes*100:.2f}%)")
    print(f"  Validation nodes: {val_mask.sum().item()} ({val_mask.sum().item()/data.num_nodes*100:.2f}%)")
    print(f"  Test nodes: {test_mask.sum().item()} ({test_mask.sum().item()/data.num_nodes*100:.2f}%)")

    print(
        f"Dataset info - num_nodes: {data.x.shape[0]}, feature_dim: {data.x.shape[1]}, num_classes: {args.num_class}"
    )
    print(
        f"Parameters: ch={args.ch}, L={args.L}, Q={args.Q}, gamma={args.gamma}, dropout={args.dropout}"
    )

    net = HoloGraph(
        n=args.N,
        ch=args.ch,
        L=args.L,              
        Q=args.Q,
        gamma=args.gamma,
        J=args.J,
        use_omega=args.use_omega,
        global_omg=args.global_omg,
        c_norm=args.c_norm,
        num_class=data.y.max().item() + 1, 
        feature_dim=data.x.size(-1),        
        dropout=args.dropout,
        homo=True,

        gst_level=1,           
        gst_wavelet=(0,1,2),   

        init_x_from="gst",    
        fuse_y_into_x=False,  
        pred_from="y",         
    ).to(device)


    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total number of basemodel parameters: {total_params}")

    if args.finetune:
        logging.info("Loading checkpoint...")
        state = torch.load(args.finetune, map_location=device)
        net.load_state_dict(state["model_state_dict"])

    optimizer = optim.AdamW(
        net.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    if args.finetune:
        logging.info("Loading optimizer state...")
        state = torch.load(args.finetune, map_location=device)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr

    ema = EMA(net, beta=0.995, update_every=1, update_after_step=50)

    def lr_scheduler(epoch):
        if epoch < args.warmup_iters:
            lr_scale = min(1.0, (epoch + 1) / args.warmup_iters)
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * lr_scale
        else:
            import numpy as np

            decay_factor = 0.5 * (
                1
                + np.cos(
                    np.pi * (epoch - args.warmup_iters)
                    / (args.epochs - args.warmup_iters)
                )
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr * decay_factor

    best_val_metrics = None
    best_test_metrics = None
    best_val_epoch = 0

    all_epochs = []
    all_val_accs = []
    all_test_accs = []

    best_test_acc_overall = 0
    best_test_epoch_overall = 0

    best_val_state_dict = None
    best_test_state_dict = None

    log_interval = 10 if args.epochs > 100 else 5

    for epoch in range(0, args.epochs):
        loss = train_fn(
            net, ema, optimizer, lr_scheduler, data, epoch, train_mask, val_mask, test_mask
        )
        results, _, _ = test_fn(net, data, train_mask, val_mask, test_mask)

        all_epochs.append(epoch + 1)
        all_val_accs.append(results["val"]["accuracy"])
        all_test_accs.append(results["test"]["accuracy"])

        if best_val_metrics is None or results["val"]["accuracy"] > best_val_metrics["accuracy"]:
            best_val_metrics = results["val"].copy()
            best_test_metrics = results["test"].copy()
            best_val_epoch = epoch + 1
            best_val_state_dict = copy.deepcopy(net.state_dict())

        if results["test"]["accuracy"] > best_test_acc_overall:
            best_test_acc_overall = results["test"]["accuracy"]
            best_test_epoch_overall = epoch + 1
            best_test_state_dict = copy.deepcopy(net.state_dict())

        if (epoch + 1) % log_interval == 0 or epoch == 0 or epoch == args.epochs - 1:
            print(
                f"Epoch {epoch + 1:03d}, Loss: {loss:.4f}, "
                f"Val Acc: {results['val']['accuracy']:.4f}, "
                f"Test Acc: {results['test']['accuracy']:.4f}, "
                f"Test Pre: {results['test']['precision']:.4f}, "
                f"Test F1: {results['test']['f1']:.4f}, "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    print("\n" + "=" * 80)
    print(f"Best test accuracy: {best_test_acc_overall:.4f} (Epoch {best_test_epoch_overall})")
    print(f"Acc at best validation: {best_test_metrics['accuracy']:.4f} (Epoch {best_val_epoch})")
    print(f"F1 at best validation: {best_test_metrics['f1']:.4f}")
    print(f"Precision at best validation: {best_test_metrics['precision']:.4f}")
    print("=" * 80)


def run_heterophilic(args, accelerator: Accelerator, device: torch.device): 

    datasets = {
        "Cornell": WebKB(root="data/Cornell", name="Cornell"),
        "Texas": WebKB(root="data/Texas", name="Texas"),
        "Wisconsin": WebKB(root="data/Wisconsin", name="Wisconsin"),
        "Actor": Actor(root="data/Actor"),
        "Cora_geom": Planetoid(root="data/Cora", name="Cora", split="geom-gcn"),
        "Citeseer_geom": Planetoid(root="data/Citeseer", name="Citeseer", split="geom-gcn"),
    }

    if args.data not in datasets:
        raise ValueError(f"Dataset {args.data} is not supported in heterophilic branch")

    dataset = datasets[args.data].to(device)

    metrics = {"accuracy": [], "precision": [], "f1": []}

    data0 = dataset[0]
    num_classes = dataset.num_classes if hasattr(dataset, "num_classes") else int(data0.y.max().item() + 1)
    num_features = dataset.num_features if hasattr(dataset, "num_features") else data0.x.size(1)

    if args.feature_dim is None or args.feature_dim <= 0:
        args.feature_dim = data0.num_nodes
    if args.imsize is None or args.imsize <= 0:
        args.imsize = num_features
    args.num_class = num_classes
    print("args.feature_dim: ", args.feature_dim)

    print(
        f"Dataset {args.data}: num_nodes={data0.num_nodes}, feature_dim={num_features}, num_classes={num_classes}"
    )

    def make_model():
        return HoloGraph(
        n=args.N,
        ch=args.ch,
        L=args.L,              
        Q=args.Q,
        gamma=args.gamma,
        J=args.J,
        use_omega=args.use_omega,
        global_omg=args.global_omg,
        c_norm=args.c_norm,
        num_class=data.y.max().item() + 1, 
        feature_dim=data.x.size(-1),        
        dropout=args.dropout,
        homo=False,

        gst_level=1,           
        gst_wavelet=(0,1,2),   

        init_x_from="gst",    
        fuse_y_into_x=False,  
        pred_from="y",         
    ).to(device)

    def build_optimizer(model):
        return optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    data = dataset[0]
    if data.train_mask.dim() == 2 and data.train_mask.size(1) == 10:
        split_range = range(10)
    else:
        split_range = range(1)

    for split_idx in split_range:
        if data.train_mask.dim() == 2:
            train_mask = data.train_mask[:, split_idx]
            val_mask = data.val_mask[:, split_idx]
            test_mask = data.test_mask[:, split_idx]
        else:
            train_mask = data.train_mask
            val_mask = data.val_mask
            test_mask = data.test_mask

        net = make_model()
        optimizer = build_optimizer(net)
        # ema = EMA(net, beta=args.beta, update_every=10, update_after_step=100)
        scheduler = LinearWarmupScheduler(optimizer, warmup_iters=args.warmup_iters)

        print("params: ", sum(p.numel() for p in net.parameters() if p.requires_grad))

        def train_once(epoch):
            net.train()
            running_loss = 0.0

            d = data.to(device)
            inputs = d.x.T
            sc = d.edge_index
            target = d.y

            inputs = inputs.unsqueeze(0) if inputs.dim() == 2 else inputs

            outputs, x_fea, c_fea = net(inputs, sc, sc)

            target_ = target.squeeze(1) if target.dim() == 2 else target
            target_ = target_[train_mask]
            outputs_ = outputs.squeeze(0)[train_mask]

            loss = nn.CrossEntropyLoss(label_smoothing=0.1)(outputs_, target_.to(device))
            # loss = nn.CrossEntropyLoss()(outputs_, target_.to(device))

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            if args.use_scheduler:
                scheduler.step()

            # ema.update()
            running_loss += loss.item()
            return running_loss

        def eval_once():
            net.eval()
            d = data.to(device)
            inputs = d.x.T
            sc = d.edge_index
            inputs = inputs.unsqueeze(0) if inputs.dim() == 2 else inputs
            target = d.y
            target = target.squeeze(1) if target.dim() == 2 else target

            outputs, _, _ = net(inputs, sc, sc)
            outputs = outputs.squeeze(0)

            results = {}
            for mask_name, mask in zip(["val", "test"], [val_mask, test_mask]):
                preds = outputs[mask].argmax(dim=-1)
                gt = target[mask]
                acc, pre, rec, f1 = compute_weighted_metrics(preds, gt)
                results[mask_name] = {
                    "accuracy": acc,
                    "precision": pre,
                    "f1": f1,
                }
            return results

        best_val_metrics = None
        best_test_metrics = None

        for epoch in range(args.epochs):
            loss = train_once(epoch)
            results = eval_once()

            if best_val_metrics is None or results["val"]["accuracy"] > best_val_metrics["accuracy"]:
                best_val_metrics = results["val"]
                best_test_metrics = results["test"]

            if (epoch + 1) % 10 == 0:
                print(
                    f"[split {split_idx}] Epoch {epoch + 1:03d}, Loss: {loss:.4f}, "
                    f"Val Acc: {results['val']['accuracy']:.4f}, "
                    f"Test Acc: {results['test']['accuracy']:.4f}, "
                    f"Test Pre: {results['test']['precision']:.4f}, "
                    f"Test F1: {results['test']['f1']:.4f}"
                )

        print(f"Best Test Metrics for split {split_idx}: {best_test_metrics}")
        metrics["accuracy"].append(best_test_metrics["accuracy"])
        metrics["precision"].append(best_test_metrics["precision"])
        metrics["f1"].append(best_test_metrics["f1"])

    for metric in metrics:
        mean = np.mean(metrics[metric])
        std = np.std(metrics[metric])
        print(f"{metric.capitalize()} - Mean: {mean:.4f}, Std: {std:.4f}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(enabled=True)

    accelerator = Accelerator()
    device = torch.device(
        "cuda:" + args.gpu if torch.cuda.is_available() else "cpu"
    )
    accelerate.utils.set_seed(args.seed + accelerator.process_index)

    jobdir = f"runs/{args.exp_name}/"
    if accelerator.is_main_process:
        if not os.path.exists(jobdir):
            os.makedirs(jobdir)
        create_logger(jobdir)


    planetoid_public = {"Cora", "Citeseer", "Pubmed"}
    heterophilic = {
        "Cornell",
        "Texas",
        "Wisconsin",
        "Actor",
    }

    if args.data in planetoid_public:
        run_planetoid_public(args, accelerator, device)
    elif args.data in heterophilic:
        run_heterophilic(args, accelerator, device)
    else:
        raise ValueError(
            f"Unsupported dataset '{args.data}'. "
            f"Planetoid branch: {sorted(planetoid_public)}, "
            f"Heterophilic branch: {sorted(heterophilic)}"
        )


if __name__ == "__main__":
    main()

