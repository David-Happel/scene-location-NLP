from sklearn.utils import resample
import pandas as pd

def int_to_one_hot(li):
    res = list()
    n = li.max()
    for i in li:
        encoding = [0 for _ in range(n+1)]
        encoding[i] = 1
        res.append(encoding)
    return res

def one_hot_to_int(li):
    res = list()
    for i in li:
        category = i.index(1)
        res.append(category)
    return res

def balance_data_down(data):
    class_0 = data[data.location==0]
    class_1 = data[data.location==1]

    if len(class_0) < len(class_1):
        class_1 = resample(class_1, 
                replace=True,     # sample with replacement
                n_samples=len(class_0),    # to match majority class
                random_state=42) # reproducible results
    elif len(class_1) < len(class_0):
        class_0 = resample(class_0, 
                replace=True,     # sample with replacement
                n_samples=len(class_1),    # to match majority class
                random_state=42) # reproducible results

    # Combine majority class with upsampled minority class
    result = pd.concat([class_0, class_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    return result