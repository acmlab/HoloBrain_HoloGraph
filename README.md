## From HoloBrain to HoloGraph: A Very First Step to Explore Machine Intelligence for Connecting Dots on Graphs

```plaintext
HoloBrain_HoloGraph/
 ├─ Holobrain.py                  # Script for computing Holobrain (CFC)
 ├─ source/
 │   ├─ data/
 │   │   ├─ create_dataset.py    
 │   │   └─ dataset.py            # Data loading for different brain data
 │   ├─ modules/
 │   │   ├─ GST.py                # GST module (Graph Sattering Transform)
 │   │   └─ kuramoto_solver.py    # Kuramoto solver for oscillator synchronization
 │   ├─ brick.py                  # The main BRICK model
 │   └─ utils.py                  
 ├─ train_brain.py                # Script for brain data
 ├─ train_cluster.py              # Script for unsupervised clustering
 └─ train_node.py                 # Script for node-level prediction
```
---

## ⚙️ Installation

1. **Create environment**

   ```bash
   conda create -n holobrain python=3.10 -y
   conda activate holobrain
   ```

2. **Install PyTorch**

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Configure Accelerate**

   ```bash
   accelerate config
   ```


---

## 📂 Data

The script uses:

```python
from source.data.create_dataset import create_dataset
dataset = create_dataset(args.data)
```

* Supported datasets are defined in your `create_dataset` implementation.
* Example: `"HCP-YA"`.
* Each dataset must yield tuples of the form:

  ```python
  (features, adjacency_matrix, target)
  ```

---

## 🚀 Running

### Single-GPU / CPU

```bash
python train_brain.py --L 2 --N 4 --batchsize 256 --T 8 --h 256 --epochs 200 --data HCP-YA --gpu 0
```

### Multi-GPU with Accelerate

```bash
accelerate launch --multi_gpu --num_processes 2 --gpu_ids 0,1 --main_process_port 29500 train_brain.py --L 2 --N 4 --batchsize 256 --T 8 --ch 256 --epochs 200 --data HCP-YA  
```

### Key Arguments

* **Training**:
  `--epochs`, `--lr`, `--ema_decay`, `--warmup_iters`, `--batchsize`, `--num_workers`
* **Data/Model**:
  `--data`, `--num_nodes`, `--feature_dim`, `--num_class`,
  `--L` (# solvers), `--T` (# time steps), `--N` (oscillator dim), `--h` (hidden dim)
* **Options**:
  `--use_pe` (positional encoding),
  `--node_cls` (node classification mode),
  `--parcellation` (parcellation mode),
  `--y_type` (`linear|conv`),
  `--mapping_type` (`conv|gconv`)

For full list:

```bash
python train_brain.py -h
```

---

## 📊 Training Details

* **Cross-validation**: 5-fold (default).
* **Optimization**: Adam + linear warmup scheduler.
* **EMA**: model weights updated with decay factor (`--ema_decay`).
* **Metrics**: Accuracy, Precision, Recall, F1 (weighted).

At the end of training:

* **Best metrics per fold** are logged.
* **Average results across folds** are reported.


