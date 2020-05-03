import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir

data_file = os.path.join(directory_path, "data/word_counts/test_data.pkl")
test_data = pd.read_pickle(data_file)

pickle_file = os.path.join(directory_path, "data/word_counts/word_counts.pkl")
counts = pd.read_pickle(pickle_file)

grouped_data = test_data.groupby('location')
locations = list(grouped_data.groups.keys())

test_preds = []
for line in test_data['clean_line']:
    words = line.split()
    offset = 0
    for word in words:
        if word in counts.index:
            word_offset = counts.loc[word]['offset']
            offset += word_offset
    if offset < 0:
        test_preds.append(locations[1])
    else:
        test_preds.append(locations[0])

print(f1_score(test_data['location'], test_preds, average='weighted'))
print(confusion_matrix(test_data['location'], test_preds))
plt.show()