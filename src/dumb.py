import os
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir
data_dir = os.path.join(directory_path, "data/parsed_transcripts")
pickle_out_dir = os.path.join(directory_path, "data/embeddings")

def read_data():
    res = pd.DataFrame([])
    for season_name in os.listdir(data_dir):
        print("--"+season_name)
        season_path = os.path.join(data_dir, season_name)
        for filename in os.listdir(season_path):
            episode_path = os.path.join(season_path, filename)
            print(episode_path)
            res = res.append(pd.read_csv(episode_path, '|'), ignore_index=True)
    return res

data = read_data()

punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`\'{|}~.,'
data['clean_line'] = data['line'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
data['clean_line'] = data['clean_line'].str.replace("[0-9]", " ")
data['clean_line'] = data['clean_line'].apply(lambda x:' '.join(x.split()))

# import spaCy's language model
nlp = spacy.load('en', disable=['parser', 'ner'])
# function to lemmatize text
def lemmatization(texts):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output

data['clean_line'] = lemmatization(data['clean_line'])
data['clean_line'] = data['clean_line'].str.replace("-PRON-", "")

data = data.drop(['line', 'character'], axis=1)

data, test = train_test_split(data, test_size=0.2)

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
    location_counts.head(30).plot(kind='bar', title=location)
    counts[location] = location_counts

counts = pd.concat(counts, join='outer', axis=1).fillna(0)
counts.columns = locations

def calc_offset(counts):
    if(counts[0] < counts[1]):
        return -(counts[0]/counts[1])
    else:
        return counts[1]/counts[0]

counts['offset_central'] = (counts[locations[0]] / counts[locations[1]]) - 1
counts['offset_appartment'] = (counts[locations[1]] / counts[locations[0]]) - 1
# max_value = counts['offset'].max()
# min_value = counts['offset'].min()
# counts['offset'] = (counts['offset'] - min_value) / (max_value - min_value)
counts['offset'] = counts.apply(calc_offset, axis=1)


# counts[['offset_central']].sort_values(by=['offset_central']).head(30).plot(kind='bar', title='offset_central')
# counts[['offset_appartment']].sort_values(by=['offset_appartment']).head(30).plot(kind='bar', title='offset_appartment')
counts[['offset']].sort_values(by=['offset']).head(30).plot(kind='bar', title='offset_central')
counts[['offset']].sort_values(by=['offset'], ascending=False).head(30).plot(kind='bar', title='offset_appartment')

pickle_file = os.path.join(directory_path, "data/word_counts/word_counts.pkl")
counts.to_pickle(pickle_file)
test_data = os.path.join(directory_path, "data/word_counts/test_data.pkl")
test.to_pickle(test_data)

plt.show()