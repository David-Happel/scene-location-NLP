import csv
import os
import spacy
import torch
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import pandas as pd
import pickle



tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir
data_dir = os.path.join(directory_path, "data/parsed_transcripts")
pickle_out_dir = os.path.join(directory_path, "data/embeddings")

def read_data():
    return pd.read_csv(data_dir+'/season01/0101.csv', '|')

data = read_data()

punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~.'
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
# print(data.sample(10))


def get_vectors(scentences):
    elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
    embeddings = elmo(
        scentences,
        signature="default",
        as_dict=True)["elmo"]
  
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings,1))

list_data = [data[i:i+100] for i in range(0,data.shape[0],100)]
elmo_data = [get_vectors(x['clean_line'].tolist()) for x in list_data]

elmo_data_new = np.concatenate(elmo_data, axis = 0)


pickle_out = open(pickle_out_dir+"/elmo_data_03032019.pickle","wb")
pickle.dump(elmo_data_new, pickle_out)
pickle_out.close()
