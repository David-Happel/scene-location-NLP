import os

import matplotlib.pyplot as plt
import pandas as pd

dic = pd.DataFrame()


def inspect():
    script_path = os.path.abspath(__file__)  # path to python script
    directory_path = os.path.dirname(os.path.split(
        script_path)[0])  # path to python script dir
    tfidf_path = '/home/david/Projects/NLP-project/reports/report-2020-06-09-10-51-14/tfidf_test_results.pkl'
    tfidf_coef_path = '/home/david/Projects/NLP-project/reports/report-2020-06-09-10-51-14/tfidf_coefficients.pkl'
    tfidf_coefficients = pd.read_pickle(tfidf_coef_path).set_index('feature_name')
    global dic
    dic = tfidf_coefficients

    tfidf = pd.read_pickle(tfidf_path).sort_values(by=['pred']).reset_index(drop=True).reset_index()
    low_vals = tfidf.sort_values(by=['pred'])[['pred', 'clean_line', 'location']].head(20)
    high_vals = tfidf.sort_values(by=['pred'])[['pred', 'clean_line', 'location']].tail(20)

    plt.tight_layout()
    plt.show()

    return


inspect()
