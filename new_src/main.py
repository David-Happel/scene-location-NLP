import pandas as pd
import os
from clean import clean
from balance import balance_down
from embed import elmo_embed, tfidf_vectorize
from classify import classify
from evaluate import evaluate
from report import log, report_path

config = {
    "elmo": {
        "batch_size": 50,
        "logistic_regression_sklearn": {
            "epochs": 100,
            "reg": 0.001
        }
    },
    "tfidf": {
        "logistic_regression_sklearn": {
            "epochs": 100,
            "reg": 0.001
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
    data = balance_down(data)

    for technique, options in config.items():
        log('starting: ' + technique)
        if technique == "elmo":
            embedded_data = elmo_embed(data, options)

        elif technique == 'tfidf':
            embedded_data = tfidf_vectorize(data, options)

        train, test = classify(embedded_data, options)
        test = evaluate(test)
        test.to_pickle(report_path(technique + '_test_results.pkl'))

    log('done')
    return


main()
