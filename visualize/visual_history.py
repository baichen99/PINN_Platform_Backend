import pandas as pd
import matplotlib.pyplot as plt


def plot_history(log_filename: str, columns: list, save_path: str):
    """plot value history from log file, columns is a list of column names
        compare different columns of the same log file
    """
    df = pd.read_csv(log_filename)
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 8)
    ax.set_xlabel('epoch')
    ax.set_ylabel('value')
    for column in columns:
        ax.plot(df['epoch'], df[column], label=column)
    ax.legend()
    fig.savefig(save_path)
    

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
    # files = ['logs/adaptive loss.csv', 'logs/without adaptive loss.csv', 'logs/ns_RAR_D.csv']
    # compare_history(files, 'l2_err_0', 'l2_err_0.png')
    # compare_history(files, 'l2_err_1', 'l2_err_1.png')
    
    # plot_history('logs/adaptive loss.csv', ['w1', 'w2'], 'adaptive loss weights history.png')
    # plot_history('logs/adaptive loss.csv', ['log_sigma1', 'log_sigma2'], 'adaptive loss log_sigma history.png')
    # plot_history('logs/adaptive loss.csv', ['pde_loss_0', 'pde_loss_1'], 'adaptive loss pde_loss history.png')
    # plot_history('logs/adaptive loss.csv', ['boundary_loss_0', 'boundary_loss_1'], 'adaptive loss boundary_loss history.png')
    # plot_history('logs/adaptive loss.csv', ['loss'], 'adaptive loss history.png')
    
    
    files = ['logs/burgers_ada_loss.csv', 'logs/burgers_without_ada_loss.csv']
    # compare_history(files, 'l2_err_0', 'burgers_l2_err_0.png')
    
    plot_history('logs/burgers_ada_loss.csv', ['w1', 'w2'], 'burgers adaptive loss weights history.png')
    # plot_history('logs/burgers_ada_loss.csv', ['log_sigma1', 'log_sigma2'], 'burgers adaptive loss log_sigma history.png')
    # plot_history('logs/burgers_ada_loss.csv', ['pde_loss_0'], 'burgers adaptive loss pde_loss history.png')
    # plot_history('logs/burgers_ada_loss.csv', ['bc_loss_0'], 'burgers adaptive loss boundary_loss history.png')
    # plot_history('logs/burgers_ada_loss.csv', ['loss'], 'burgers adaptive loss history.png')
        