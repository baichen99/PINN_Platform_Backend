import pandas as pd
import matplotlib.pyplot as plt
from utils.csv import get_cols


def plot_history(df: pd.DataFrame, columns: list, save_path: str):
    """plot value history from log file, columns is a list of column names
        compare different columns of the same log file
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    ax.set_xlabel('epoch')
    ax.set_ylabel('value')
    for column in columns:
        x, y = get_cols(df, column)
        ax.plot(x, y, label=column)
    ax.legend()
    fig.savefig(save_path)
    plt.close()

def plot_xy(x, y, save_path: str, x_label: str = 'x', y_label: str = 'y'):
    """plot x, y
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.plot(x, y)
    fig.savefig(save_path)
    plt.close()
    
def compare_history(log_filenames: list, column: str, save_path: str):
    """compare same columns of different log files
    """
    dfs = [pd.read_csv(log_filename) for log_filename in log_filenames]
    fig, axs = plt.subplots()
    fig.set_size_inches(12, 8)
    axs.set_xlabel('epoch')
    axs.set_ylabel(column)
    
    for idx, df in enumerate(dfs):
        df = df[['epoch', column]].dropna(axis=0, how='any', inplace=False)
        if not column in df.columns:
            raise ValueError(f"Column {column} not found in {log_filenames[idx]}")
        axs.plot(df['epoch'], df[column], label=log_filenames[idx])
    axs.legend()
    fig.savefig(save_path)
    

if __name__ == '__main__':
    ...