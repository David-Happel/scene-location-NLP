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

t = time.time()
now = datetime.now()

source_report = 'report-2020-05-12-18-36-10'

script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir
source_report_path = os.path.join(directory_path, 'reports', source_report)
embeddings_path = os.path.join(source_report_path, 'embeddings.pkl')
valid_embeddings_path = os.path.join(source_report_path, 'valid_embeddings.pkl')
report_dir = os.path.join(source_report_path, 'reclassify_report-'+now.strftime("%Y-%m-%d-%H-%M-%S"))
os.mkdir(report_dir)

logging.basicConfig(filename=os.path.join(report_dir, 'log.log'), level=logging.DEBUG)

# Get full command-line arguments
full_cmd_arguments = sys.argv

# Keep all but the first
argument_list = full_cmd_arguments[1:]

short_options = "b:e:"
long_options = ["batchsize=", "epochs="]

try:
    arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
    # Output error, and return with an error code
    logging.error(str(err))
    sys.exit(2)

BATCH_SIZE = 50
EPOCHS = 3

# Evaluate given options
for current_argument, current_value in arguments:
    if current_argument in ("-b", "--batchsize"):
        BATCH_SIZE = int(current_value)
    elif current_argument in ("-e", "--epochs"):
        EPOCHS = int(current_value)

logging.info('starting with batchsize: %s epochs: %s', BATCH_SIZE, EPOCHS)

# inspiration: https://github.com/sambit9238/Deep-Learning/blob/master/elmo_embedding_tfhub.ipynb
# https://medium.com/@the1ju/simple-logistic-regression-using-keras-249e0cc9a970

tf.disable_eager_execution()
tf.get_logger().setLevel('INFO')
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

logging.info('Loading data... %s', time.time() - t)

data = pd.read_pickle(embeddings_path)
X_embedded = np.array([np.array(x) for x in data['embedding']])
y = np.array(data['location'])
logging.debug(Counter(y))

print(X_embedded.shape)

logging.info('Training classifier... %s', time.time() - t)

classifier = classifier_model()
classifier.summary(print_fn=logging.info)
logging.info('loss_func %s', classifier.loss)
logging.info('optimizer_func %s', classifier.optimizer)

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

validation_data = pd.read_pickle(valid_embeddings_path)
valid_X_embedded = np.array([np.array(x) for x in validation_data['embedding']])
valid_y = np.array(validation_data['location'])

with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())  
    session.run(tf.tables_initializer())
    classifier.load_weights(model_weights_path)

    score = classifier.evaluate(valid_X_embedded, valid_y, batch_size=BATCH_SIZE)
    valid_pred = classifier.predict(valid_X_embedded, batch_size=BATCH_SIZE)

    logging.info('valid score: %s', score[0]) 
    logging.info('valid accuracy:%s', score[1])
    logging.info(confusion_matrix(valid_y.tolist(), valid_pred[:,0].round().astype(int).tolist()))

logging.info('Total time %s', time.time() - t)

validation_data['prediction'] = valid_pred[:,0].round().astype(int).tolist()

validation_data['correct'] = validation_data['location'] == validation_data['prediction']

tsne30 = TSNE(random_state=42,n_iter=2000,metric='cosine',n_components=2, perplexity=30)

embd_tr = tsne30.fit_transform(validation_data['embedding'].to_list())
validation_data['ts_x_axis'] = embd_tr[:,0]
validation_data['ts_y_axis'] = embd_tr[:,1]

plt.figure()
sns.scatterplot('ts_x_axis', 'ts_y_axis', hue='location', style='correct', size=1, data=validation_data[['location', 'ts_x_axis', 'ts_y_axis', 'correct']].sample(3000)).set_title('perp: 30')
tsne_plot_path = os.path.join(report_dir, 'tsne.png')
plt.savefig(tsne_plot_path)
plt.show()