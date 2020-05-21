import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import getopt
import os
import sys
from clean_data import clean_data
from collections import Counter
import tensorflow.keras as keras
from keras.layers import Input, Lambda, Dense
from keras.models import Model
from tensorflow.compat.v1.keras import backend as K
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import time
from datetime import datetime
import logging
import matplotlib.pyplot as plt
from models import classifier_model, elmo_model
from helper import balance_data_down

t = time.time()
now = datetime.now()

script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir
data_path = os.path.join(directory_path, "data/parsed_transcripts.csv")
report_dir = os.path.join(directory_path, 'reports', 'report-'+now.strftime("%Y-%m-%d-%H-%M-%S"))
os.mkdir(report_dir)

logging.basicConfig(filename=os.path.join(report_dir, 'log.log'), level=logging.DEBUG)

# Get full command-line arguments
full_cmd_arguments = sys.argv

# Keep all but the first
argument_list = full_cmd_arguments[1:]

short_options = "n:b:e:"
long_options = ["nrows=", "batchsize=", "epochs="]

try:
    arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
    # Output error, and return with an error code
    logging.error(str(err))
    sys.exit(2)

NROWS = None
BATCH_SIZE = 50
EPOCHS = 3

# Evaluate given options
for current_argument, current_value in arguments:
    if current_argument in ("-n", "--nrows"):
        NROWS = int(current_value)
    elif current_argument in ("-b", "--batchsize"):
        BATCH_SIZE = int(current_value)
    elif current_argument in ("-e", "--epochs"):
        EPOCHS = int(current_value)

logging.info('starting with nrows: %s batchsize: %s epochs: %s', NROWS, BATCH_SIZE, EPOCHS)

# inspiration: https://github.com/sambit9238/Deep-Learning/blob/master/elmo_embedding_tfhub.ipynb
# https://medium.com/@the1ju/simple-logistic-regression-using-keras-249e0cc9a970

tf.disable_eager_execution()
tf.get_logger().setLevel('INFO')
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

logging.info('Loading data... %s', time.time() - t)
data = pd.read_csv(data_path, index_col=0, nrows=NROWS)


locations = list(data.groupby('location').groups.keys())

data.loc[data["location"]==locations[0],"location"]=0
data.loc[data["location"]==locations[1],"location"]=1

logging.info('Cleaning data... %s', time.time() - t)
data = clean_data(data)

logging.info('location counts')
logging.info(data.location.value_counts())
logging.info('Balancing data... %s', time.time() - t)
data = balance_data_down(data)
logging.info('location counts after balance')
logging.info(data.location.value_counts())

X = np.array(data['clean_line'])
y = np.array(data['location'])


logging.info('Embedding data... %s', time.time() - t)

elmo = elmo_model()
with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())  
    session.run(tf.tables_initializer())
    X_embedded = elmo.predict(X, batch_size=BATCH_SIZE, verbose=1)

embeddings_path = os.path.join(report_dir, 'embeddings.pkl')
data['embedding'] = X_embedded.tolist()
data[['location', 'clean_line', 'embedding']].to_pickle(embeddings_path)

logging.info('Training classifier... %s', time.time() - t)
classifier = classifier_model()
classifier.summary(print_fn=logging.info)

with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())  
    session.run(tf.tables_initializer())
    history = classifier.fit(X_embedded, y, epochs=EPOCHS, batch_size=1000, validation_split=0.2)

    model_weights_path = os.path.join(report_dir, 'model_elmo_weights.h5')
    classifier.save_weights(model_weights_path)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'g', label='Training Acc')
plt.plot(epochs, val_acc, 'b', label='Validation Acc')
plt.title('Training and validation Acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()

training_plot_path = os.path.join(report_dir, 'training.png')
plt.savefig(training_plot_path)

logging.info('Validating model... %s', time.time() - t)

validation_path = os.path.join(directory_path, "data/validation_transcripts.csv")
validation_data = pd.read_csv(validation_path, index_col=0)
validation_data.loc[validation_data["location"]==locations[0],"location"]=0
validation_data.loc[validation_data["location"]==locations[1],"location"]=1
validation_data = clean_data(validation_data)
validation_data = balance_data_down(validation_data)
valid_X = np.array(validation_data['clean_line'])
valid_y = np.array(validation_data['location'])

with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())  
    session.run(tf.tables_initializer())
    valid_X_embedded = elmo.predict(valid_X, batch_size=BATCH_SIZE, verbose=1)

valid_embeddings_path = os.path.join(report_dir, 'valid_embeddings.pkl')
validation_data['embedding'] = valid_X_embedded.tolist()
validation_data[['location', 'clean_line', 'embedding']].to_pickle(valid_embeddings_path)

with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())  
    session.run(tf.tables_initializer())
    classifier.load_weights(model_weights_path, by_name=True)

    score = classifier.evaluate(valid_X_embedded, valid_y, batch_size=BATCH_SIZE)
    
    logging.info('valid score: %s', score[0]) 
    logging.info('valid accuracy:%s', score[1])

logging.info('Total time %s', time.time() - t)
