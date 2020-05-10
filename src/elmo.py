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
    print (str(err))
    sys.exit(2)

NROWS = None
BATCH_SIZE = 50
EPOCHS = 5

# Evaluate given options
for current_argument, current_value in arguments:
    if current_argument in ("-n", "--nrows"):
        NROWS = int(current_value)
    elif current_argument in ("-b", "--batchsize"):
        BATCH_SIZE = int(current_value)
    elif current_argument in ("-e", "--epochs"):
        EPOCHS = int(current_value)

print('starting with nrows:', NROWS, 'batchsize:', BATCH_SIZE, 'epochs:', EPOCHS)


# inspiration: https://github.com/sambit9238/Deep-Learning/blob/master/elmo_embedding_tfhub.ipynb
# https://medium.com/@the1ju/simple-logistic-regression-using-keras-249e0cc9a970

tf.disable_eager_execution()

scentences = ["the cat is on the mat", "what are you doing in evening"]

script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir
data_path = os.path.join(directory_path, "data/parsed_transcripts.csv")
model_weights_path = os.path.join(directory_path, './model_elmo_weights.h5')

data = pd.read_csv(data_path, index_col=0, nrows=NROWS)

locations = list(data.groupby('location').groups.keys())

data.loc[data["location"]==locations[0],"location"]=0
data.loc[data["location"]==locations[1],"location"]=1

data = clean_data(data)
X = np.array(data['clean_line'])
y = np.array(data['location'])

print(Counter(y))

embed = hub.Module("https://tfhub.dev/google/elmo/3")
def ELMoEmbedding(x):
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

# def build_model(): 
#     input_text = Input(shape=(1,), dtype="string")
#     embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
#     dense = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(embedding)
#     pred = Dense(1, activation='sigmoid')(dense)
#     model = Model(inputs=[input_text], outputs=pred)
#     model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#     return model

def elmo_model():
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
    model = Model(inputs=[input_text], outputs=embedding)
    return model
    
def classifier_model():
    input_embeddings = Input(shape=(1024,))
    dense = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(input_embeddings)
    pred = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=[input_embeddings], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

elmo = elmo_model()

with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())  
    session.run(tf.tables_initializer())
    X_embedded = elmo.predict(X, batch_size=BATCH_SIZE, verbose=1)

classifier = classifier_model()
print(classifier.summary())

with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())  
    session.run(tf.tables_initializer())
    history = classifier.fit(X_embedded, y, epochs=EPOCHS, batch_size=1000, validation_split=0.2)
    classifier.save_weights(model_weights_path)

# import matplotlib.pyplot as plt

# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# plt.plot(epochs, acc, 'g', label='Training Acc')
# plt.plot(epochs, val_acc, 'b', label='Validation Acc')
# plt.title('Training and validation Acc')
# plt.xlabel('Epochs')
# plt.ylabel('Acc')
# plt.legend()

validation_path = os.path.join(directory_path, "data/validation_transcripts.csv")
validation_data = pd.read_csv(validation_path, index_col=0)
validation_data.loc[validation_data["location"]==locations[0],"location"]=0
validation_data.loc[validation_data["location"]==locations[1],"location"]=1
validation_data = clean_data(validation_data)
valid_X = np.array(validation_data['clean_line'])
valid_y = np.array(validation_data['location'])

with tf.Session() as session:
    K.set_session(session)
    session.run(tf.global_variables_initializer())  
    session.run(tf.tables_initializer())
    classifier.load_weights(model_weights_path)

    t = time.time()
    # predicts = model_elmo.predict(valid_X)
    valid_X_embedded = elmo.predict(valid_X, batch_size=BATCH_SIZE, verbose=1)
    score = classifier.evaluate(valid_X_embedded, valid_y, batch_size=BATCH_SIZE)
    print("time: ", time.time() - t)
    print('valid score:', score[0]) 
    print('valid accuracy:', score[1])

# plt.show()