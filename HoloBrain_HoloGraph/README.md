## From HoloBrain to HoloGraph: A Very First Step to Explore Machine Intelligence for Connecting Dots on Graphs

```plaintext
BRICK/
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
