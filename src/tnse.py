import pandas as pd
from sklearn.manifold import TSNE
import os
import seaborn as sns
import matplotlib.pyplot as plt

# tuning https://distill.pub/2016/misread-tsne/


script_path = os.path.abspath(__file__)  # path to python script
directory_path = os.path.dirname(os.path.split(script_path)[0])  # path to python script dir
report_dir = os.path.join(directory_path, 'reports', 'report-2020-05-12-12-53-16')
data_path = os.path.join(report_dir, 'embeddings.pkl')

data = pd.read_pickle(data_path)
data = data.sample(2000, random_state = 42)
data = data[['location', 'embedding']]

tsne3 = TSNE(random_state=42,n_iter=3000,metric='cosine',n_components=2, perplexity=3)

embd_tr = tsne3.fit_transform(data['embedding'].to_list())
data['ts_x_axis'] = embd_tr[:,0]
data['ts_y_axis'] = embd_tr[:,1]

sns.scatterplot('ts_x_axis','ts_y_axis',hue='location', data=data[['location', 'ts_x_axis', 'ts_y_axis']]).set_title('perp: 3')
plt.figure()

tsne10 = TSNE(random_state=42,n_iter=3000,metric='cosine',n_components=2, perplexity=10)

embd_tr = tsne10.fit_transform(data['embedding'].to_list())
data['ts_x_axis'] = embd_tr[:,0]
data['ts_y_axis'] = embd_tr[:,1]

sns.scatterplot('ts_x_axis','ts_y_axis',hue='location', data=data[['location', 'ts_x_axis', 'ts_y_axis']]).set_title('perp: 10')
plt.figure()

tsne30 = TSNE(random_state=42,n_iter=3000,metric='cosine',n_components=2, perplexity=30)

embd_tr = tsne30.fit_transform(data['embedding'].to_list())
data['ts_x_axis'] = embd_tr[:,0]
data['ts_y_axis'] = embd_tr[:,1]

sns.scatterplot('ts_x_axis','ts_y_axis',hue='location', data=data[['location', 'ts_x_axis', 'ts_y_axis']]).set_title('perp: 30')
plt.figure()

tsne50 = TSNE(random_state=42,n_iter=3000,metric='cosine',n_components=2, perplexity=50)

embd_tr = tsne50.fit_transform(data['embedding'].to_list())
data['ts_x_axis'] = embd_tr[:,0]
data['ts_y_axis'] = embd_tr[:,1]

sns.scatterplot('ts_x_axis','ts_y_axis',hue='location', data=data[['location', 'ts_x_axis', 'ts_y_axis']]).set_title('perp: 50')
plt.figure()

tsne100 = TSNE(random_state=42,n_iter=3000,metric='cosine',n_components=2, perplexity=100)

embd_tr = tsne100.fit_transform(data['embedding'].to_list())
data['ts_x_axis'] = embd_tr[:,0]
data['ts_y_axis'] = embd_tr[:,1]

sns.scatterplot('ts_x_axis','ts_y_axis',hue='location', data=data[['location', 'ts_x_axis', 'ts_y_axis']]).set_title('perp: 100')

plt.show()
