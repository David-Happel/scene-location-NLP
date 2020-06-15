import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def interpret():
    script_path = os.path.abspath(__file__)  # path to python script
    directory_path = os.path.dirname(os.path.split(
        script_path)[0])  # path to python script dir
    tfidf_path = '/home/david/Projects/NLP-project/reports/report-2020-06-09-10-51-14/tfidf_coefficients.pkl'
    elmo_path = '/home/david/Projects/NLP-project/reports/report-2020-06-09-11-27-37/elmo_coefficients.pkl'

    tfidf_coefficients = pd.read_pickle(tfidf_path).sort_values(by=['coef']).reset_index(drop=True).reset_index()
    elmo_coefficients = pd.read_pickle(elmo_path).sort_values(by=['coef']).reset_index(drop=True).reset_index()
    # coefficients = pd.DataFrame({"elmo_coef": elmo_coefficients['coef'], "tfidf_coef": tfidf_coefficients['coef']})

    ax = elmo_coefficients.plot.scatter(
        x='index', y='coef', fontsize=14, color='orange')
    ax.set_xlabel("Elmo Feature", fontsize=16)
    ax.set_ylabel("Classifier Coefficient", fontsize=16)
    plt.tight_layout()
    ax = tfidf_coefficients.plot.scatter(
        x='index', y='coef', fontsize=14, color='blue')
    ax.set_xlabel("TF-IDF Feature Word", fontsize=16)
    ax.set_ylabel("Classifier Coefficient", fontsize=16)

    plt.tight_layout()
    plt.show()

    return


interpret()
