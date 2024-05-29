import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pickle

exp_name = 'without_lstm_pre'
filename = 'data/' + exp_name + '.pickle'
with open(filename, 'rb') as file:
    data = pickle.load(file)
if os.path.exists('data/' + exp_name) == False:
    os.makedirs('data/' + exp_name)

print(data)
plt.plot(data['losses'], label='loss')
plt.xlabel('Iter')
plt.ylabel('Loss')
plt.title("Training losses")
# plt.ylim(0, 7.5)
plt.plot([data['online_transition']-data['buffer'],data['online_transition']-data['buffer']], [0, np.max(data['losses'])], label='online transition', color='red', linestyle='--')
plt.legend()
plt.savefig('data/' + exp_name + '/loss.png')

plt.figure()
plt.plot(data['lat_errs'],label='lateral error')
plt.xlabel('Iter')
plt.ylabel('Lateral error')
plt.title("Lateral error (in m)")
plt.plot([data['buffer'],data['buffer']], [0, np.max(data['lat_errs'])], label='start training offline', color='red', linestyle='--')
plt.plot([data['online_transition'],data['online_transition']], [0, np.max(data['lat_errs'])], label='online transition', color='red', linestyle='--')
plt.legend()
plt.savefig('data/' + exp_name + '/lat_errs.png')

plt.figure()
plt.plot(data['ws_gt'],label='GT w')
plt.plot(data['ws_'],label='Predicted w after 5 steps')
plt.plot(data['ws'],label='Predicted w with -steering after 20 steps')
plt.xlabel('Iter')
plt.ylabel('w (in rad/s)')
plt.title("w")
plt.plot([data['buffer'],data['buffer']], [np.min(data['ws']), np.max(data['ws_'])], label='start training offline', color='red', linestyle='--')
plt.plot([data['online_transition'],data['online_transition']], [np.min(data['ws']), np.max(data['ws_'])], label='online transition', color='red', linestyle='--')
plt.legend()
plt.savefig('data/' + exp_name + '/ws.png')

