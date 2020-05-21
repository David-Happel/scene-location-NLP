import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import getopt
import os
import sys
from clean_data import clean_data
from collections import Counter
from tensorflow.compat.v1.keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import time
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from models import classifier_model, logistic_regression
from helper import int_to_one_hot
from sklearn.manifold import TSNE
import seaborn as sns

script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir
source_guesses_path = '/home/david/Projects/NLP-project/reports/no_lemma_report/reclassify_report-2020-05-18-15-54-10/guesses.pkl'

data = pd.read_pickle(source_guesses_path)

tsne30 = TSNE(random_state=42,n_iter=1000,metric='cosine',n_components=2, perplexity=30)

embd_tr = tsne30.fit_transform(data['embedding'].to_list())
data['ts_x_axis'] = embd_tr[:,0]
data['ts_y_axis'] = embd_tr[:,1]

scatterplot = sns.scatterplot('ts_x_axis', 'ts_y_axis', hue='location', style='correct', size=1, data=data[['location', 'ts_x_axis', 'ts_y_axis', 'correct']], picker=4).set_title('perp: 30')

def onpick(event):
    ind = event.ind
    row = data.iloc[ind[0]]
    print('onpick3 scatter:', row['correct'], row['clean_line'])


scatterplot.figure.canvas.mpl_connect("pick_event", onpick)


plt.show()