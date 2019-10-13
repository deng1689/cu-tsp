import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pdb import set_trace

files = ['groundtruth.csv'] + ['model{}.csv'.format(i) for i in range(5)]
#files = ['groundtruth.csv'] * 6
protein_lens = {'2n64': 75, '3jb5': 300, '5fhy': 229, '5fjl': 136, '5j4a': 116, 
                '5jo9': 239}
output_dists = {}
LOG_SCALE = False

def findnth(string, substring, n):
    parts = string.split(substring, n + 1)
    if len(parts) <= n + 1:
        return -1
    return len(string) - len(parts[-1]) - len(substring)

for (k, v) in protein_lens.items():
    output_dists[k] = [np.zeros((v, v)) for _ in range(len(files))]
for (idx, f) in enumerate(files):
    df = pd.read_csv(f)
    for i in range(len(df)):
        pid_i_j = df['Id'][i]
        pid = pid_i_j[:4]
        x_idx = findnth(pid_i_j, '_', 1)
        y_idx = findnth(pid_i_j, '_', 2)
        x = int(pid_i_j[x_idx+1:y_idx]) - 1
        y = int(pid_i_j[y_idx+1:]) - 1
        val = df['Predicted'][i]
        output_dists[pid][idx][x, y] = val

#pickle.dump( output_dists, open( "output.p", "wb" ) )
#output_dists = pickle.load( open( "output.p", "rb" ) )

for p in protein_lens.keys():
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')
    ax1.imshow(output_dists[p][0], cmap='hot')
    ax2.imshow(output_dists[p][1], cmap='hot')
    ax3.imshow(output_dists[p][2], cmap='hot')
    ax4.imshow(output_dists[p][3], cmap='hot')
    ax5.imshow(output_dists[p][4], cmap='hot')
    ax6.imshow(output_dists[p][5], cmap='hot')
    fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(ax6)
    plt.show()
    
