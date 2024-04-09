import os
import json
import torch
import time
import wandb
import numpy as np

from offroad.utils import load_state
from car_dynamics import CAR_DYNAMICS_ROOT_DIR
from offroad import OFFROAD_MODEL_DIR, OFFROAD_DATA_DIR
from car_dynamics.analysis import pos2vel_savgol, calc_delta_v
from car_dynamics.scripts.data_config import DatasetConfig
from car_dynamics.models_torch import MLP, train_mlp_dbm_pipeline, parse_data_end2end_8dim, create_missing_folder, normalize_dataset
from torch.utils.data import DataLoader, TensorDataset, random_split
from car_dynamics.envs import DynamicParams, DynamicBicycleModel

from termcolor import colored
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("DEVICE", DEVICE)

LF = .16
LR = .15
L = LF+LR


def train_model(H, BATCH_SIZE, hidden_size, NUM_EPOCHS, LR, data_config, model_struct='',tune=False):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    experiment_dir = timestr
    dt = 0.05
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
                        'dt': dt,
                     })
    
    print(f"H{H}-BS{BATCH_SIZE}-E{NUM_EPOCHS}-LR{LR}")
    
    
    # EXP_DIR = f"H{H}-BS{BATCH_SIZE}-E{NUM_EPOCHS}-LR{LR}"
    model_dir = os.path.join(OFFROAD_MODEL_DIR, experiment_dir, f'model.pt')
    create_missing_folder(model_dir, is_file_name=True)
    
    
    model = MLP(input_size=8*H, hidden_size=hidden_size, output_size=4, last_layer_activation='none')
    # data_parser = parse_data_end2end_norm   
    #TODO: implement parsing dataset for phyx_dbm
    data_parser = parse_data_end2end_8dim
    model.to(DEVICE)

    
    all_data_list = DatasetConfig[data_config]['train_data_list']
    
    log_dir = os.path.join(OFFROAD_DATA_DIR, 'vicon-data-clean')
    X_all, y_all = data_parser(H, all_data_list, log_dir,)
    total_samples = len(X_all)
    train_size = int(0.7 * total_samples)
    test_size = total_samples - train_size
    print(colored(f"Total samples: {total_samples}, train_size: {train_size}, test_size: {test_size}", 'green'))

    # mean = X_all.mean()
    # std = X_all.std()
    # X_all = (X_all - mean) / std
    full_dataset = TensorDataset(X_all, y_all)
    # normalize_dataset(full_dataset)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=4096, shuffle=True)
        
    model_params = DynamicParams(num_envs=BATCH_SIZE, DT=dt,Sa=0.45, Sb=0.0, K_FFY=15, K_RFY=15, Ta=8.)
    dynamics = DynamicBicycleModel(model_params, device=DEVICE)
    train_info = train_mlp_dbm_pipeline(run, model, dynamics, train_loader, test_loader, BATCH_SIZE, DEVICE, model_dir, num_epochs=NUM_EPOCHS, lr=LR)
    run.finish()
    
if __name__ == "__main__":
    for LR in [1e-4,]:
        for BATCH_SIZE in [32]:
            for hidden_size in [64]:
                for NUM_EPOCHS in [2000]:
                    for H in [1]:
                        for data_config in ['real_dataset_3']:
                            for model_struct in ['phyx_dbm',]:
                                for tune in [False]:
                                    train_model(H, BATCH_SIZE, hidden_size, NUM_EPOCHS, LR, data_config, model_struct, tune)