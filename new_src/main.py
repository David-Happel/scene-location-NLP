import pandas as pd
import os
import numpy as np
from clean import clean
from balance import balance_down, split
from embed import elmo_embed, tfidf_vectorize
from classify import classify
from evaluate import evaluate
from report import log, report_path

config = {
    # "elmo": {
    #     "batch_size": 50,
    #     "logistic_regression_sklearn": {
    #         "grid_search": {"C": np.logspace(-4, 5, 10), 'max_iter': [100, 300, 500]},
    #         'C': 1000,
    #         'max_iter': 100
    #     }
    # },
    "tfidf": {
        "stop_words": "english",
        "logistic_regression_sklearn": {
            # "grid_search": {"C": np.logspace(-2, 4, 7), 'max_iter': [100, 300, 500, 1000]},
            'C': 1000.0,
            'max_iter': 100
        }
    }
}
cleaning_options = {
    "split_lines": True,
    "remove_brackets": True,
    "remove_numbers": True,
    "minimum_words": 3,
    "remove_punctuation": True,
    "lemmatize": False,
    "remove_pron": False
}


def main():
    log('starting with: ' + str(config))
    log('cleaning options: ' + str(cleaning_options))
    script_path = os.path.abspath(__file__)  # path to python script
    directory_path = os.path.dirname(os.path.split(
        script_path)[0])  # path to python script dir
    data_path = os.path.join(directory_path, "data/parsed_transcripts.csv")

    log('reading data..')
    data = pd.read_csv(data_path, index_col=0)

    log('cleaning data..')
    data = clean(data, cleaning_options)
    train, test = split(data)
    train = balance_down(train)
    test = balance_down(test)
    log("training size: " + str(len(train)))
    log("testing size: " + str(len(test)))

    for technique, options in config.items():
        log('starting: ' + technique)
        if technique == "elmo":
            embedded_train, embedded_test = elmo_embed(train, test, options)

        elif technique == 'tfidf':
            embedded_train, embedded_test = tfidf_vectorize(train, test, options)
            log("tfidf feature length: " + str(embedded_train["embedding"].shape[1]))

        train, test = classify(embedded_train, embedded_test, options)
        test = evaluate(test)
        test.to_pickle(report_path(technique + '_test_results.pkl'))

    log('done')
    return


main()
