import os

import pandas as pd


def inspect_data():
    script_path = os.path.abspath(__file__)  # path to python script
    directory_path = os.path.dirname(os.path.split(
        script_path)[0])  # path to python script dir
    data_path = os.path.join(directory_path, "data/parsed_transcripts.csv")

    data = pd.read_csv(data_path, index_col=0)
    unique_locations = data['original_location'].unique()
    return


inspect_data()
