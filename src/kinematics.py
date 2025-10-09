import pandas as pd


def load_dlc(path):
    df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    return df
