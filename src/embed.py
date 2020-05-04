import csv
import os
import spacy
import torch
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import pandas as pd
import pickle
from clean_data import clean_data
from timer import Timer


t = Timer()
t.start()

tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir
data_dir = os.path.join(directory_path, "data/parsed_transcripts")
pickle_out_dir = os.path.join(directory_path, "data/embeddings")

print('reading data..')
def read_data():
    res = pd.DataFrame([])
    for filename in os.listdir(data_dir+'/season01')[0:5]:
        res = res.append(pd.read_csv(data_dir+'/season01/'+filename, '|'), ignore_index=True)
    return res

data = read_data()

print('cleaning..')
data = clean_data(data)
data = data.drop(['line', 'character'], axis=1)

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

print('splitting data..')
list_data = [data[i:i+100] for i in range(0,data.shape[0],100)]
# elmo_data = [get_vectors(x['clean_line'].tolist()) for x in list_data]
elmo_data = []

print('starting embedding..')
t.log()
for i, x in enumerate(list_data):
    print('embedding:', i, '/', len(list_data))
    t.log()
    elmo_data.append(get_vectors(x['clean_line'].tolist()))

print('embedding done')
print('combining data..')
elmo_data_new = np.concatenate(elmo_data, axis = 0)

print('saving data..')
pickle_out = open(pickle_out_dir+"/elmo_data_03032019.pickle","wb")
pickle.dump(elmo_data_new, pickle_out)
pickle_out.close()
t.stop()
