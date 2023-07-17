import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot(X, Y, save_path=None):
    if X.shape[1] == 2:
        # plot 3d
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(X[:, 0], X[:, 1], Y, cmap=plt.get_cmap('rainbow'), c=Y, s=1)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')
        if save_path:
            plt.savefig(save_path)

if __name__ == '__main__':
    data = pd.read_csv('data/cmdkv/test_data.csv')
    X = data[['x', 't']].values
    Y = data['u'].values
    plot(X, Y, 'data/cmdkv/test_data.png')
