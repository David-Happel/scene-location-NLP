import os

import pandas as pd

script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir
data_path = os.path.join(directory_path, "data/parsed_transcripts.csv")

def read_data(process, nrows=None, chunksize=1000):
    results = []
    for chunk in pd.read_csv(data_path, index_col=0, nrows=nrows, chunksize=chunksize):
        results.append(process(chunk))
    return results