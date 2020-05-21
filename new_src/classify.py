from keras.regularizers import L1L2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
from keras.layers import Input, Lambda, Dense
from keras.models import Model, Sequential
from tensorflow.compat.v1.keras import backend as K
from sklearn.linear_model import LogisticRegression

tf.get_logger().setLevel('INFO')

def classify(data, options):
    train, test = split(data)
    train_embeddings = np.array([np.array(x) for x in train['embedding']])
    train_y = np.array(train['location_int'])
    test_embeddings = np.array([np.array(x) for x in test['embedding']])

    if "logistic_regression_sklearn" in options:
        model_config = options["logistic_regression_sklearn"]
        lr = LogisticRegression(max_iter=model_config["epochs"])
        lr.fit(train_embeddings, train_y)
        test['pred'] = lr.predict_proba(test_embeddings)[:,1]
        test['pred_int'] = lr.predict(test_embeddings)

    if "logistic_regression" in options:
        model_config = options["logistic_regression"]
        model = logistic_regression_model(model_config, input_dim=train_embeddings.shape[1])

        with tf.Session() as session:
            K.set_session(session)
            session.run(tf.global_variables_initializer())  
            session.run(tf.tables_initializer())
            history = model.fit(train_embeddings, train_y, epochs=model_config["epochs"], batch_size=1000, validation_split=0.2)
            preds = model.predict(test_embeddings, batch_size=1000)
            test['pred'] = preds[:,0]
            test['pred_int'] = test['pred'].round(0)
    
    return train, test

def logistic_regression_model(config = {"reg": 0.001}, input_dim=1024):
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=L1L2(l1=config["reg"], l2=config["reg"]), input_dim=input_dim))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def split(data):
    return train_test_split(data, test_size=0.2, random_state=42)
