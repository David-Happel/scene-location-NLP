import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def balance_down(data):
    class_0 = data[data.location_int == 0]
    class_1 = data[data.location_int == 1]

    if len(class_0) < len(class_1):
        class_1 = resample(class_1, n_samples=len(class_0), random_state=42)
    elif len(class_1) < len(class_0):
        class_0 = resample(class_0, n_samples=len(class_1), random_state=42)

    result = pd.concat([class_0, class_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    return result


def split(data):
    return train_test_split(data, test_size=0.2, random_state=42)


def shuffle(data):
    return data.sample(frac=1, random_state=42).reset_index(drop=True)
