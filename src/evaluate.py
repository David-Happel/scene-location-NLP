
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split

script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir
data_dir = os.path.join(directory_path, "data/parsed_transcripts")
pickle_in_dir = os.path.join(directory_path, "data/embeddings")

data = pd.read_pickle(pickle_in_dir+"/embeddings.pkl")
print(data.head())

xtrain, xvalid, ytrain, yvalid = train_test_split(data['embedding'].tolist(), 
                                                  data['location'].tolist(),  
                                                  random_state=42, 
                                                  test_size=0.2)

lreg = LogisticRegression(max_iter=1000)
lreg.fit(xtrain, ytrain)

preds_valid = lreg.predict(xvalid)

print(f1_score(yvalid, preds_valid, average='weighted'))
print(accuracy_score(yvalid, preds_valid))
plot_confusion_matrix(lreg, xvalid, yvalid)
plt.show()