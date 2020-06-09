import pandas as pd
import os
import numpy as np
from clean import clean
from balance import balance_down, split, shuffle
from embed import elmo_embed, tfidf_vectorize
from classify import classify
from evaluate import evaluate
from report import log, report_path

config = {
    # "elmo": {
    #     "batch_size": 50,
    #     "logistic_regression_sklearn": {
    #         "grid_search": {"C": np.logspace(-4, 5, 10), 'max_iter': [5, 10, 20, 50, 100, 200, 300]},
    #         'C': 100,
    #         'max_iter': 300
    #     }
    # },
    "tfidf": {
        "stop_words": None,
        "logistic_regression_sklearn": {
            "grid_search": {"C": np.logspace(-4, 5, 10), 'max_iter': [5, 10, 20, 50, 100, 200, 300]},
            'C': 10000.0,
            'max_iter': 300
        }
    }
}
cleaning_options_list = [{
    "split_lines": True,
    "remove_brackets": True,
    "remove_numbers": True,
    "minimum_words": 2,
    "remove_punctuation": True,
    "lemmatize": False,
    "remove_pron": False}
    # , {
    #     "split_lines": True,
    #     "remove_brackets": True,
    #     "remove_numbers": True,
    #     "minimum_words": 2,
    #     "remove_punctuation": True,
    #     "lemmatize": False,
    #     "remove_pron": False
    # }, {
    #     "split_lines": True,
    #     "remove_brackets": True,
    #     "remove_numbers": True,
    #     "minimum_words": 3,
    #     "remove_punctuation": True,
    #     "lemmatize": False,
    #     "remove_pron": False
    # }, {
    #     "split_lines": True,
    #     "remove_brackets": True,
    #     "remove_numbers": True,
    #     "minimum_words": 3,
    #     "remove_punctuation": True,
    #     "lemmatize": False,
    #     "remove_pron": False
    # }, {
    #     "split_lines": True,
    #     "remove_brackets": True,
    #     "remove_numbers": True,
    #     "minimum_words": 5,
    #     "remove_punctuation": True,
    #     "lemmatize": False,
    #     "remove_pron": False
    # }
]


def main():
    log('starting with: ' + str(config))
    for cleaning_options in cleaning_options_list:
        log('cleaning options: ' + str(cleaning_options))
        script_path = os.path.abspath(__file__)  # path to python script
        directory_path = os.path.dirname(os.path.split(
            script_path)[0])  # path to python script dir
        data_path = os.path.join(directory_path, "data/parsed_transcripts.csv")

        log('reading data..')
        data = pd.read_csv(data_path, index_col=0)

        log('cleaning data..')
        data = clean(data, cleaning_options)
        data = balance_down(data)

        train, test = split(data)
        log("training size: " + str(len(train)))
        log("testing size: " + str(len(test)))

        for technique, options in config.items():
            log('starting: ' + technique)
            if technique == "elmo":
                embedded_train, embedded_test = elmo_embed(train, test, options)

            elif technique == 'tfidf':
                embedded_train, embedded_test = tfidf_vectorize(train, test, options)
                log("tfidf feature length: " + str(len(np.array(embedded_train["embedding"].iloc[1]))))

            train_res, test_res = classify(embedded_train, embedded_test, options, technique=technique)
            test_res = evaluate(test_res)
            test_res.to_pickle(report_path(technique + '_test_results.pkl'))

    log('done')
    return


main()
