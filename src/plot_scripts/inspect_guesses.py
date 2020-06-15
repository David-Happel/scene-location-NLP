import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE


def inspect_data():
    script_path = os.path.abspath(__file__)  # path to python script
    directory_path = os.path.dirname(os.path.split(
        script_path)[0])  # path to python script dir
    source_guesses_path = '/home/david/Projects/NLP-project/reports/report-2020-05-27-09-36-59/tfidf_test_results.pkl'

    data = pd.read_pickle(source_guesses_path)

    tsne30 = TSNE(random_state=42, n_iter=1000, metric='cosine',
                  n_components=2, perplexity=30)

    embd_tr = tsne30.fit_transform(data['embedding'].to_list())
    data['ts_x_axis'] = embd_tr[:, 0]
    data['ts_y_axis'] = embd_tr[:, 1]

    scatterplot = sns.scatterplot('ts_x_axis', 'ts_y_axis', hue='location', style='correct', size=1, data=data[[
                                  'location', 'ts_x_axis', 'ts_y_axis', 'correct']], picker=4).set_title('perp: 30')

    def onpick(event):
        ind = event.ind
        row = data.iloc[ind[0]]
        print(row['clean_line'])

    scatterplot.figure.canvas.mpl_connect("pick_event", onpick)

    scatterplot_path = os.path.join(
        directory_path, os.path.split(source_guesses_path)[0], "scatterplot.png")
    plt.savefig(scatterplot_path)
    plt.show()
    return


inspect_data()
