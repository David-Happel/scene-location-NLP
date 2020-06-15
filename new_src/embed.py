from tensorflow.compat.v1.keras import backend as K
import tensorflow.compat.v1 as tf
import numpy as np
from keras.layers import Input, Lambda, Dense
from keras.models import Model
import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer

tf.get_logger().setLevel('INFO')
tf.disable_eager_execution()
elmo = None


def elmo_embed(train, test, options={'batch_size': 50}):
    global elmo
    elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=False)

    train_clean_lines = np.array(train['clean_line'])
    test_clean_lines = np.array(test['clean_line'])

    model = elmo_model()
    with tf.Session() as session:
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        train_embedded_lines = model.predict(
            train_clean_lines, batch_size=options["batch_size"], verbose=1)
        test_embedded_lines = model.predict(
            test_clean_lines, batch_size=options["batch_size"], verbose=1)
    train["embedding"] = train_embedded_lines.tolist()
    test["embedding"] = test_embedded_lines.tolist()
    return train, test


def tfidf_vectorize(train, test, options={"stop_words": None}):
    tfidf = TfidfVectorizer(stop_words=options["stop_words"])
    train["embedding"] = tfidf.fit_transform(
        train["clean_line"]).toarray().tolist()
    test["embedding"] = tfidf.transform(
        test["clean_line"]).toarray().tolist()

    feature_names = tfidf.get_feature_names()
    return train, test, feature_names


def ELMoEmbedding(x):
    return elmo(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]


def elmo_model():
    input_text = Input(shape=(1,), dtype="string")
    embedding = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
    model = Model(inputs=[input_text], outputs=embedding)
    return model
