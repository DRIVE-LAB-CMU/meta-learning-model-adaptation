import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

a_max = 0.5
v_max = 20.
mu = 0.5
g = 9.81
mass = 1.

def add_columns_to_csv(input_file: str, output_file: str):
    """
    Reads a CSV file with 2 columns, adds two new columns with specified names
    and a constant value, and saves the modified DataFrame to a new CSV file.
    
    :param input_file: Path to the input CSV file
    :param output_file: Path to the output CSV file
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Check if the CSV has exactly two columns
    if df.shape[1] != 2:
        raise ValueError("The input CSV must have exactly two columns")
    
    # Add new columns with specified names and a constant value
    df['w_tr_right_m'] = 0.2
    df['w_tr_left_m'] = 0.2
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

def add_ggv(input_file: str, output_file: str) :
    df = pd.read_csv(input_file)
    df.iloc[:, 1] = a_max
    df.iloc[:, 2] = mu*mass*g

    df.to_csv(output_file, index=False)

def add_speeds(input_file: str, output_file: str):
    df = pd.read_csv(input_file)
    df.iloc[:,1] = a_max
    df.iloc[:,0] *= v_max / df.iloc[-1,0]
    df.to_csv(output_file, index=False)
    
    
# Example usage
input_track = 'berlin_2018.csv'
output_file = '../global_racetrajectory_optimization/inputs/tracks/input.csv'

add_columns_to_csv(input_track, output_file)
add_ggv('../global_racetrajectory_optimization/inputs/veh_dyn_info/ggv_1.csv', '../global_racetrajectory_optimization/inputs/veh_dyn_info/ggv.csv')
add_speeds('../global_racetrajectory_optimization/inputs/veh_dyn_info/ax_max_machines_1.csv', '../global_racetrajectory_optimization/inputs/veh_dyn_info/ax_max_machines.csv')
os.system('python3 ../global_racetrajectory_optimization/main_globaltraj.py')

df = pd.read_csv('../global_racetrajectory_optimization/outputs/traj_race_cl.csv',skiprows=2,delimiter=';')
print(df.iloc[:,:])
df_new = df.iloc[:,[0, 1,2,5]]
df_new.to_csv(input_track[:-4]+'_with_speeds'+'.csv', index=False)
plt.plot(df_new.iloc[:,1],df_new.iloc[:,2])
plt.axis('equal')
plt.show()