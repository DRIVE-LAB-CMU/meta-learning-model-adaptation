import os
import json
import torch
import time
import wandb
import numpy as np

from offroad import OFFROAD_MODEL_DIR, OFFROAD_DATA_DIR
from offroad.utils import load_state
from car_dynamics.analysis import pos2vel_savgol, calc_delta_v
from car_dynamics.scripts.data_config import DatasetConfig
from car_dynamics.models_torch import MLP, train_MLP_pipeline, parse_data, parse_data_end2end, parse_data_heading_psi, create_missing_folder
from torch.utils.data import DataLoader, TensorDataset, random_split
from termcolor import colored
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE", DEVICE)

LF = .16
LR = .15
L = LF+LR


def train_model(H, BATCH_SIZE, hidden_size, NUM_EPOCHS, LR, data_config, model_struct='heading',tune=False):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    experiment_dir = timestr
    
    run = wandb.init(project="offroad",
                     config={
                        'dir': experiment_dir,
                        'lr': LR,
                        'H': H,
                        'batch_size': BATCH_SIZE,
                        'num_epochs': NUM_EPOCHS,
                        'data_config': data_config,
                        'hidden_size': hidden_size,
                        'model_struct': model_struct,
                        'tune': tune,
                     })
    
    print(f"H{H}-BS{BATCH_SIZE}-E{NUM_EPOCHS}-LR{LR}")
    
    # EXP_DIR = f"H{H}-BS{BATCH_SIZE}-E{NUM_EPOCHS}-LR{LR}"
    model_dir = os.path.join(OFFROAD_MODEL_DIR, experiment_dir, f'model.pt')
    create_missing_folder(model_dir, is_file_name=True)
    
    if model_struct == 'end2end':
        model = MLP(input_size=6*H, hidden_size=hidden_size, output_size=4, last_layer_activation='none')
        if tune:
            raise NotImplementedError
        data_parser = parse_data_end2end
    elif model_struct == 'heading':
        model = MLP(input_size=6*H, hidden_size=hidden_size, output_size=3, last_layer_activation='none')
        data_parser = parse_data
    elif model_struct == 'heading-psi':
        model = MLP(input_size=6*H, hidden_size=hidden_size, output_size=5, last_layer_activation='none')
        data_parser = parse_data_heading_psi
    else:
        raise Exception(f"model_struct {model_struct} not supported!")

    model.to(DEVICE)    
    
    
    all_data_list = DatasetConfig[data_config]['train_data_list']
    X_all, y_all = data_parser(H, all_data_list, load_state, OFFROAD_DATA_DIR, LF, LR)
    total_samples = len(X_all)
    train_size = int(0.7 * total_samples)
    test_size = total_samples - train_size
    print(colored(f"Total samples: {total_samples}, train_size: {train_size}, test_size: {test_size}", 'green'))

    full_dataset = TensorDataset(X_all, y_all)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=4096, shuffle=True)
        
    train_info = train_MLP_pipeline(run, model, train_loader, test_loader, DEVICE, model_dir, num_epochs=NUM_EPOCHS, lr=LR)
    run.finish()
    
if __name__ == "__main__":
    for LR in [1e-3]:
        for BATCH_SIZE in [512]:
            for hidden_size in [32, 64]:
                for NUM_EPOCHS in [2000]:
                    for H in [2, 4, 8]:
                        for data_config in ['real_dataset_1']:
                            for model_struct in ['end2end',]:
                                for tune in [False]:
                                    train_model(H, BATCH_SIZE, hidden_size, NUM_EPOCHS, LR, data_config, model_struct, tune)