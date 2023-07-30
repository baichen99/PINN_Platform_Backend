import pandas as pd


def read_csv(filename: str) -> pd.DataFrame:
    """read csv file, return a pandas dataframe
    """
    return pd.read_csv(filename)

def get_cols(df: pd.DataFrame, col: str) -> (list, list):
    """get column names of a dataframe
    """
    if col not in df.columns:
        raise ValueError(f"Column {col} not found in dataframe")
    # remove nan values
    df = df[['epoch', col]].dropna(axis=0, how='any', inplace=False)
    return df['epoch'].tolist(), df[col].tolist()