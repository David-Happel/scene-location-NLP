import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def plot_clean():
    data_elmo = {
        "lemmatization\u2717\nrem. numbers\u2713\nrem. punctuation\u2713": [0.5949793930310978, 0.594230048707381, 0.5717497189958786],
        "lemmatization\u2713\nrem. numbers\u2713\nrem. punctuation\u2713": [0.5806233062330624, 0.5884146341463414, 0.5785907859078591],
        "lemmatization\u2717\nrem. numbers\u2717\nrem. punctuation\u2713": [0.5803671787186212, 0.5706257025103035, 0.5893593106032222],
        "lemmatization\u2717\nrem. numbers\u2713\nrem. punctuation\u2717": [0.5863619333083552, 0.590108654926939, 0.5934807043836643]
    }
    data_tfidf = {
        "lemmatization\u2717\nrem. numbers\u2713\nrem. punctuation\u2713": [0.6530535781191458, 0.6586736605470214, 0.6376920194829524],
        "lemmatization\u2713\nrem. numbers\u2713\nrem. punctuation\u2713": [0.6456639566395664, 0.6395663956639567, 0.6314363143631436],
        "lemmatization\u2717\nrem. numbers\u2717\nrem. punctuation\u2713": [0.6579243162233046, 0.6511802173098539, 0.664293742974897],
        "lemmatization\u2717\nrem. numbers\u2713\nrem. punctuation\u2717": [0.6508055451479955, 0.6395653802922443, 0.6489321843387036]
    }

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

    elmo_plot = plt.boxplot(data_elmo.values(), positions=np.array(range(len(data_elmo))) *
                            2.0-0.4, sym='', widths=0.6)

    tfidf_plot = plt.boxplot(data_tfidf.values(), positions=np.array(
        range(len(data_tfidf)))*2.0+0.4, sym='', widths=0.6)

    set_box_color(elmo_plot, '#D7191C')
    set_box_color(tfidf_plot, '#2C7BB6')

    plt.plot([], c='#D7191C', label='ELMo')
    plt.plot([], c='#2C7BB6', label='TF-IDF')

    ticks = data_elmo.keys()
    plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize=14)
    plt.yticks(fontsize=14)

    ax.set_ylabel("Accuracy Score", fontsize=20)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    fig.subplots_adjust(bottom=0.2)

    plt.show()


plot_clean()
