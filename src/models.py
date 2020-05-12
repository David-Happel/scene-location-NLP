from keras.layers import Input, Lambda, Dense
from keras.models import Model, Sequential
import tensorflow.keras as keras
from keras.regularizers import L1L2
import tensorflow.compat.v1 as tf
import tensorflow_hub as hub

def classifier_model(reg=0.001):
    input_embeddings = Input(shape=(1024,))
    dense = Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(reg))(input_embeddings)
    pred = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs=[input_embeddings], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def logistic_regression():
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=L1L2(l1=0.01, l2=0.01), input_dim=1024))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def ELMoEmbedding(x):
    embed = hub.Module("https://tfhub.dev/google/elmo/3")
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

def elmo_model():
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
    model = Model(inputs=[input_text], outputs=embedding)
    return model