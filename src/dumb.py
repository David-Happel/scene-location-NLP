import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from clean_data import clean_data
from read_data import read_data

script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir
data_dir = os.path.join(directory_path, "data/parsed_transcripts")
pickle_out_dir = os.path.join(directory_path, "data/embeddings")


data = pd.DataFrame()
def process(chunk):
    global data
    data = data.append(chunk)

read_data(process)

print('cleaning..')
data = clean_data(data)

data = data.drop(['line', 'character'], axis=1)

data, test = train_test_split(data, test_size=0.2)

print('counting words..')
grouped_data = data.groupby('location')
locations = list(grouped_data.groups.keys())

counts = {}

for location in locations:
    location_lines = grouped_data.get_group(location)['clean_line']
    location_counts = dict()
    total = 0
    for line in location_lines:
        words = line.split()
        for word in words:
            total += 1
            if word in location_counts:
                location_counts[word] += 1
            else:
                location_counts[word] = 1
    for word in location_counts.keys():
        location_counts[word] = location_counts[word] / total
    location_counts = pd.DataFrame.from_dict(location_counts, orient='index', columns=[location])
    location_counts = location_counts.sort_values(by=[location], ascending=False)
    # location_counts.head(30).plot(kind='bar', title=location)
    counts[location] = location_counts

counts = pd.concat(counts, join='outer', axis=1).fillna(0)
counts.columns = locations

def calc_offset(counts):
    if(counts[0] < counts[1]):
        return -(counts[0]/counts[1])
    else:
        return counts[1]/counts[0]

print('calculating offset..')
counts['offset'] = counts.apply(calc_offset, axis=1)

# counts[['offset']].sort_values(by=['offset']).head(30).plot(kind='bar', title='offset_central')
# counts[['offset']].sort_values(by=['offset'], ascending=False).head(30).plot(kind='bar', title='offset_appartment')

pickle_file = os.path.join(directory_path, "data/word_counts/word_counts.pkl")
counts.to_pickle(pickle_file)
test_data = os.path.join(directory_path, "data/word_counts/test_data.pkl")
test.to_pickle(test_data)

plt.show()
