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


def plot_min_words():
    data_elmo = {
        "None": [0.5942199941020347, 0.5664995576526098, 0.578885284576821],
        "min. 2 words": [0.572498298162015, 0.5830496936691627, 0.5850918992511913],
        "min. 3 words": [0.5818658673660547, 0.5762457849381791, 0.5646309479205694],
        "min. 4 words": [0.5938553765455227, 0.5739977519670288, 0.5897339827650806],
        "min. 5 words": [0.5970081595648232, 0.5752493200362647, 0.599728014505893]
    }
    data_tfidf = {
        "None": [0.6446475965791801, 0.6163373636095547, 0.633146564435269],
        "min. 2 words": [0.6412525527569776, 0.6456773315180395, 0.6412525527569776],
        "min. 3 words": [0.6609216935181716, 0.643312101910828, 0.6436867740726864],
        "min. 4 words": [0.6440614462345448, 0.646309479205695, 0.6601723491944549],
        "min. 5 words": [0.6772438803263826, 0.6627379873073436, 0.6640979147778785]
    }

    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

    elmo_plot = plt.boxplot(data_elmo.values(), positions=np.array(range(len(data_elmo)))*2.0-0.4, sym='', widths=0.6)
    tfidf_plot = plt.boxplot(data_tfidf.values(), positions=np.array(
        range(len(data_tfidf)))*2.0+0.4, sym='', widths=0.6)
    set_box_color(elmo_plot, '#D7191C')
    set_box_color(tfidf_plot, '#2C7BB6')

    plt.plot([], c='#D7191C', label='ELMo')
    plt.plot([], c='#2C7BB6', label='TF-IDF')

    ticks = data_elmo.keys()
    plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize=14)
    plt.yticks(fontsize=14)

    ax.set_ylabel('Accuracy Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')

    plt.show()


plot_min_words()
