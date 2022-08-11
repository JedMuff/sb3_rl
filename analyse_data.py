import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, nargs='+', default=["PPO_RandomPegCylinder-v0"], help="directory file is in")
parser.add_argument("-f", "--file_name", type=str, nargs='+', default=[], help="name of file without .csv ending")
args = parser.parse_args()

dirs = args.model
file_names = args.file_name

def plot_data(df, xcol=None, title=''):

    cols = df.columns.tolist()
    if 'Unnamed: 0' in cols:
        cols.remove('Unnamed: 0')

    length = len(cols)

    sizes = {1: (1,1), 2: (1, 2), 3: (1, 3),
             4: (2,2), 5: (2, 3), 6: (2, 3),
             7: (3,2), 8: (3, 3), 9: (3, 3),
             10: (3,4), 11: (3, 4), 12: (3, 4),
             13: (4,4), 14: (4, 4), 15: (4, 4),
             16: (4,4)}

    size = sizes[length]
    
    for idx, col in enumerate(cols):
        plt.subplot(size[0], size[1], idx+1)
        if xcol == None:
            df.plot(y=col, kind = 'line', ax=plt.gca(), use_index=True)
        else:
            df.plot(x=xcol, y=col, kind = 'line', ax=plt.gca())

        plt.minorticks_on()
        plt.grid(which='major', linestyle='-')
        plt.grid(which='minor', linestyle='--')
        plt.suptitle(title)

for idx in range(len(dirs)):
    print(idx)
    if len(file_names) -1 < idx:
        file_name = 'data'
    else:
        file_name = file_names[idx]
        
    df = pd.read_csv('models/' + dirs[idx] + '/' + file_name + '.csv')

    plt.figure(idx)
    plot_data(df, title=dirs[idx])

plt.show()
