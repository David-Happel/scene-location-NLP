
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt


script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir
data_dir = os.path.join(directory_path, "data/parsed_transcripts")
pickle_in_dir = os.path.join(directory_path, "data/embeddings")

def read_data():
    res = pd.DataFrame([])
    for filename in os.listdir(data_dir+'/season01')[0:5]:
        res = res.append(pd.read_csv(data_dir+'/season01/'+filename, '|'), ignore_index=True)
    return res

data = read_data()

# load elmo_train_new
pickle_in = open(pickle_in_dir+"/elmo_data_03032019.pickle","rb")
elmo_train_new = pickle.load(pickle_in)

xtrain, xvalid, ytrain, yvalid = train_test_split(elmo_train_new, 
                                                  data['location'],  
                                                  random_state=42, 
                                                  test_size=0.2)

lreg = LogisticRegression(max_iter=1000)
lreg.fit(xtrain, ytrain)

preds_valid = lreg.predict(xvalid)

print(f1_score(yvalid, preds_valid, average='weighted'))
print(accuracy_score(yvalid, preds_valid))
plot_confusion_matrix(lreg, xvalid, yvalid)
plt.show()