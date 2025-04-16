import os
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.utils import to_dense_adj


class HCPA_BoldSCDataset(Dataset):
    def __init__(self, bold_dir, sc_dir):
        self.bold_dir = bold_dir
        self.sc_dir = sc_dir
        self.label_mapping = {
            "REST": 0,
            "CARIT": 1,
            "FACENAME": 2,
            "VISMOTOR": 3,
        }
        self.data = self._load_data()

    def _load_data(self):
        data = []
        bold_files = [f for f in os.listdir(self.bold_dir) if f.endswith(".csv")]
        for bold_file in bold_files:
            try:
                parts = bold_file.split("_")
                subject_id = parts[0]
                task_type = parts[1].split("-")[1]
                if task_type not in self.label_mapping:
                    continue
                sc_path = os.path.join(self.sc_dir, subject_id, f"{subject_id}_space-T1w_desc-preproc_msmtconnectome.mat")
                if not os.path.exists(sc_path):
                    continue
                mat = loadmat(sc_path)
                if 'aal116_radius2_count_connectivity' not in mat:
                    raise KeyError(f"AAL not found in SC file: {sc_path}")
                sc = mat['aal116_radius2_count_connectivity'].astype(np.float32)

                bold_path = os.path.join(self.bold_dir, bold_file)
                bold_data = pd.read_csv(bold_path).values[:, 1:]
                bold_data = self.pad_sentences(bold_data) if bold_data.shape[0] < 300 else bold_data[:300]
                label = self.label_mapping[task_type]
                data.append((bold_data, sc, label))
            except Exception as e:
                print(f"Error processing file {bold_file}: {e}")
        return data

    def pad_sentences(self, sentence):
        pad_data = torch.cat((torch.tensor(sentence), torch.zeros(300 - sentence.shape[0], sentence.shape[1])), dim=0)
        return pad_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bold, sc, label = self.data[idx]
        bold_tensor = torch.tensor(bold, dtype=torch.float32)
        fc = torch.corrcoef(bold_tensor.T)
        fc_tensor = torch.nan_to_num(fc)
        sc_tensor = torch.tensor(sc, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return bold_tensor, fc_tensor, label_tensor


class TUD(Dataset):
    def __init__(self, root: str, name: str):
        self.dataset = TUDataset(root=root, name=name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        x = data.x.T  
        adj = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0)
        label = data.y
        return x, adj, label


class HCP_YA_SCDataset(Dataset):
    def __init__(self, bold_dir, sc_dir, label_path, scan="LR"):
        self.bold_dir = bold_dir
        self.sc_dir = sc_dir
        self.label_path = label_path
        self.labels = pd.read_csv(label_path).values.flatten()
        self.bold_files = [f for f in os.listdir(bold_dir) if "WM" in f and scan in f]
        self.data = []
        for bold_file in self.bold_files:
            subject_id = bold_file.split("_")[0].split("-")[1]
            sc_path = os.path.join(sc_dir, f"sub-{subject_id}", f"sub-{subject_id}_space-T1w_desc-preproc_msmtconnectome.mat")
            if not os.path.exists(sc_path):
                continue
            sc_dict = loadmat(sc_path)
            if 'brainnetome246_radius2_count_connectivity' not in sc_dict:
                continue
            sc_matrix = sc_dict['brainnetome246_radius2_count_connectivity']
            bold_path = os.path.join(bold_dir, bold_file)
            bold_data = pd.read_csv(bold_path).values[:, 1:]
            segments, segment_labels = self._process_bold_segments(bold_data, self.labels)
            for segment, label in zip(segments, segment_labels):
                self.data.append((segment, sc_matrix, label))

    def _process_bold_segments(self, bold_data, labels):
        segments = []
        segment_labels = []
        current_label = labels[0]
        start_idx = 0
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                if 1 <= current_label <= 8:
                    segments.append(bold_data[start_idx:i])
                    segment_labels.append(current_label - 1)
                start_idx = i
                current_label = labels[i]
        if 1 <= current_label <= 8:
            segments.append(bold_data[start_idx:])
            segment_labels.append(current_label - 1)
        assert len(segments) == len(segment_labels) == 8
        return segments, segment_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bold_segment, sc_matrix, label = self.data[idx]
        bold_tensor = torch.tensor(bold_segment, dtype=torch.float32)
        fc = torch.corrcoef(bold_tensor.T)
        fc_tensor = torch.nan_to_num(fc)
        sc_tensor = torch.tensor(sc_matrix, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return bold_tensor, fc_tensor, label_tensor


class HCPYA_BoldSCDataset(Dataset):
    def __init__(self, bold_dir, sc_dir):
        self.bold_dir = bold_dir
        self.sc_dir = sc_dir
        self.label_mapping = {
            "EMOTION": 0,
            "GAMBLING": 1,
            "LANGUAGE": 2,
            "MOTOR": 3,
            "RELATIONAL": 4,
            "SOCIAL": 5,
            "WM": 6,
        }
        self.data = self._load_data()

    def _load_data(self):
        data = []
        bold_files = [f for f in os.listdir(self.bold_dir) if f.endswith(".csv") and "LR" in f]
        for bold_file in bold_files:
            try:
                parts = bold_file.split("_")
                subject_id = parts[0]
                task_type = parts[1].split("-")[1]
                if task_type not in self.label_mapping:
                    continue
                sc_path = os.path.join(self.sc_dir, subject_id, f"{subject_id}_space-T1w_desc-preproc_msmtconnectome.mat")
                if not os.path.exists(sc_path):
                    continue
                mat = loadmat(sc_path)
                if 'aal116_radius2_count_connectivity' not in mat:
                    raise KeyError(f"AAL not found in SC file: {sc_path}")
                sc = mat['aal116_radius2_count_connectivity'].astype(np.float32)
                bold_path = os.path.join(self.bold_dir, bold_file)
                bold = pd.read_csv(bold_path, header=0).values[:, 1:]
                bold = self.pad_sentences(bold) if bold.shape[0] < 175 else bold[:175]
                label = self.label_mapping[task_type]
                data.append((bold, sc, label))
            except Exception as e:
                print(f"Error processing file {bold_file}: {e}")
        return data

    def pad_sentences(self, sentence):
        pad_data = torch.cat((torch.tensor(sentence), torch.zeros(175 - sentence.shape[0], sentence.shape[1])), dim=0)
        return pad_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bold, sc, label = self.data[idx]
        bold_tensor = torch.tensor(bold, dtype=torch.float32)
        fc = torch.corrcoef(bold_tensor.T)
        fc_tensor = torch.nan_to_num(fc)
        sc_tensor = torch.tensor(sc, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return bold_tensor, fc_tensor, label_tensor


class HCPYA_byregion(Dataset):
    def __init__(self, bold_dir, label_path):
        self.bold_dir = bold_dir
        self.labels = self._load_labels(label_path)
        self.num_regions = 116
        self.data = self._load_and_cache_bold_data()

    def _load_and_cache_bold_data(self):
        bold_files = [os.path.join(self.bold_dir, f) for f in os.listdir(self.bold_dir)
                      if f.endswith(".csv") and "LR" in f]
        all_bold_data = []
        for bold_file in bold_files:
            bold = pd.read_csv(bold_file, header=0).values[:, 1:]  # [T, 116]
            bold_tensor = torch.tensor(bold, dtype=torch.float32)
            bold_tensor = self.pad_sentences(bold_tensor) if bold_tensor.shape[0] < 175 else bold_tensor[:175]
            all_bold_data.append(bold_tensor)
        return torch.stack(all_bold_data, dim=0)[:900]  # [num_subjects, 175, num_regions]

    def pad_sentences(self, sentence):
        pad_data = torch.cat((sentence, torch.zeros(175 - sentence.shape[0], sentence.shape[1])), dim=0)
        return pad_data

    def _load_labels(self, label_path):
        labels = np.loadtxt(label_path)
        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.num_regions

    def __getitem__(self, region_idx):
        bold_region = self.data[:, :, region_idx] 
        bold_region = bold_region.permute(1, 0)  
        label = self.labels[region_idx] - 1
        return bold_region, bold_region, label
    

class HCPA_byregion(Dataset):
    def __init__(self, bold_dir, label_path):

        self.bold_dir = bold_dir
        self.labels = self._load_labels(label_path) 
        self.num_regions = 116                      
        self.data = self._load_and_cache_bold_data() 

    def _load_and_cache_bold_data(self):

        bold_files = [os.path.join(self.bold_dir, f) for f in os.listdir(self.bold_dir) if f.endswith(".csv") and "AP" in f and "REST" in f]
        all_bold_data = []

        for bold_file in bold_files:
            bold = pd.read_csv(bold_file, header=0).values[:, 1:]  # [T, 116]
            bold_tensor = torch.tensor(bold, dtype=torch.float32)
            bold_tensor = self.pad_sentences(bold_tensor) if bold_tensor.shape[0] < 300 else bold_tensor[:300] 
            all_bold_data.append(bold_tensor)

        return torch.stack(all_bold_data, dim=0)[:900] 
    
    def pad_sentences(self, sentence):
        pad_data = torch.cat((sentence, torch.zeros(300 - sentence.shape[0], sentence.shape[1])), dim=0)
        return pad_data

    def _load_labels(self, label_path):
        labels = np.loadtxt(label_path)
        return torch.tensor(labels, dtype=torch.long) 

    def __len__(self):
        return self.num_regions

    def __getitem__(self, region_idx):

        bold_region = self.data[:, :, region_idx]  
        bold_region = bold_region.permute(1, 0)  
        label = self.labels[region_idx]-1      

        return bold_region, bold_region, label

