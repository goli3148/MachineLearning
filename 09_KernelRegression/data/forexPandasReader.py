import numpy as np
import pandas as pd

def forexHis(numbersOfData):
    df = pd.read_csv('data_/csv.csv')
    df = df.to_numpy()
    df = df[len(df)-numbersOfData:len(df)]
    df = df.transpose()
    df = df[1]
    df = df.transpose().tolist()
    x = [i for i in range(len(df))]
    return x, df
    