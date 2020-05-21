from tensorflow.compat.v1.keras import backend as K
import tensorflow.compat.v1 as tf
import numpy as np
from keras.layers import Input, Lambda, Dense
from keras.models import Model
import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer

tf.get_logger().setLevel('INFO')
tf.disable_eager_execution()

def elmo_embed(data, options={'batch_size':50}):
    clean_lines = np.array(data['clean_line'])

    elmo = elmo_model()
    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())  
        session.run(tf.tables_initializer())
        embedded_lines = elmo.predict(clean_lines, batch_size=options["batch_size"], verbose=1)
    data["embedding"] = embedded_lines.tolist()
    return data

def tfidf_vectorize(data, options={}):
    tfidf = TfidfVectorizer()
    data["embedding"] = tfidf.fit_transform(data["clean_line"]).toarray().tolist()
    return data


def ELMoEmbedding(x):
    embed = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
    return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

def elmo_model():
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
    model = Model(inputs=[input_text], outputs=embedding)
    return model