from source.data.dataset import *

def create_dataset(data):
    if data == "HCP-A":
        dataset = HCPA_BoldSCDataset(
            bold_dir='./HCP-A-SC_FC/AAL_116/BOLD', 
            sc_dir='./HCP-A-SC_FC/ALL_SC'
        )
    elif data == "HCP-YA-WM":
        dataset = HCP_YA_SCDataset(
            bold_dir='./HCP-YA-SC_FC/Brainnetome_264/BOLD_interpolated', 
            sc_dir='./HCP-YA-SC_FC/HCP-YA-SC',
            label_path='../LR_label/LR_label.csv',
            scan="LR"
        )
    elif data == "HCP-YA-WM-RL":
        dataset = HCP_YA_SCDataset(
            bold_dir='./HCP-YA-SC_FC/Brainnetome_264/BOLD_interpolated', 
            sc_dir='./HCP-YA-SC_FC/HCP-YA-SC',
            label_path='../RL_label/RL_label.csv',
            scan="RL"
        )
    elif data == "HCP-YA":
        dataset = HCPYA_BoldSCDataset(
            bold_dir='./HCP-YA-SC_FC/AAL_116/BOLD', 
            sc_dir='./HCP-YA-SC_FC/HCP-YA-SC'
        )
    elif data == "HCP-YA-region":
        dataset = HCPYA_byregion(
            bold_dir='./HCP-YA-SC_FC/AAL_116/BOLD', 
            label_path='./region_label.txt'
        )
    elif data == "HCPA-region":
        dataset = HCPA_byregion(
            bold_dir='./HCP-A-SC_FC/AAL_116/BOLD', 
            label_path='./region_label.txt'
        )
    return dataset
