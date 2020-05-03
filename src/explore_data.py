
import pandas as pd
import os
import matplotlib.pyplot as plt

script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir
data_dir = os.path.join(directory_path, "data/parsed_transcripts")
pickle_in_dir = os.path.join(directory_path, "data/embeddings")

def read_data():
    res = pd.DataFrame([])
    for filename in os.listdir(data_dir+'/season01')[0:5]:
        res = res.append(pd.read_csv(data_dir+'/season01/'+filename, '|'), ignore_index=True)
    return res


data = read_data()

plot = data.groupby('location').count().plot(kind='bar')
# plot2 = data.groupby('character').count().plot(kind='bar')
plt.show()
