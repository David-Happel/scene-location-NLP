import getopt
import os
import sys

import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub
from clean_data import clean_data
from read_data import read_data
from timer import Timer

# Get full command-line arguments
full_cmd_arguments = sys.argv

# Keep all but the first
argument_list = full_cmd_arguments[1:]

short_options = "n:c:"
long_options = ["nrows=", "chunksize="]

try:
    arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
    # Output error, and return with an error code
    print (str(err))
    sys.exit(2)

NROWS = 1000
CHUNKSIZE = 50

# Evaluate given options
for current_argument, current_value in arguments:
    if current_argument in ("-n", "--nrows"):
        NROWS = int(current_value)
    elif current_argument in ("-c", "--chunksize"):
        CHUNKSIZE = int(current_value)

print('starting with nrows:', NROWS, 'and chunksize:', CHUNKSIZE)

t = Timer()
t.start()

tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)

script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir
pickle_out_dir = os.path.join(directory_path, "data")

chunk_n = 0
embedded_data = None

def get_vectors(scentences):
    elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
    gpu_options = tf.GPUOptions(allow_growth=True)
    embeddings = elmo(
        scentences,
        signature="default",
        as_dict=True)["elmo"]

    with tf.Session(config = tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings, 1))

def process(chunk):
    global chunk_n
    chunk_n += 1
    print('chunk:', chunk_n)

    print('cleaning..')
    chunk = clean_data(chunk)
    chunk = chunk.drop(['line', 'character'], axis=1)

    print('starting embedding..')
    chunk['embedding'] = get_vectors(chunk['clean_line'].tolist()).tolist()

    print('embedding done')
    print('combining data..')
    global embedded_data
    if(embedded_data is None):
        embedded_data = chunk
    else:
        embedded_data = pd.concat([embedded_data, chunk])
    t.log()

t.log()
read_data(process, nrows=NROWS, chunksize=CHUNKSIZE)

embedded_data = embedded_data.reset_index(drop=True)

print('saving data..')
pd.to_pickle(embedded_data, pickle_out_dir+"/embeddings.pkl")
t.stop()
