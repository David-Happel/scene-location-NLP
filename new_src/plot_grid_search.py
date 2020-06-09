import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_grid_search():
    script_path = os.path.abspath(__file__)  # path to python script
    directory_path = os.path.dirname(os.path.split(
        script_path)[0])  # path to python script dir
    # source_path = '/home/david/Projects/NLP-project/reports/report-2020-06-01-17-46-32/tfidf_grid_search.pkl'
    source_path = '/home/david/Projects/NLP-project/reports/report-2020-06-01-17-30-34/elmo_grid_search.pkl'

    cv_results = pd.read_pickle(source_path)

    # Plot Grid search scores
    # _, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=100)

    # grid_params = {"C": np.logspace(-4, 5, 10), 'max_iter': [5, 10, 20, 50, 100, 200, 300]}

    # scores_mean = cv_results['mean_test_score']
    # scores_mean = np.array(scores_mean).reshape(len(grid_params["C"]), len(grid_params["max_iter"]))

    # scores_train = cv_results['mean_train_score']
    # scores_train = np.array(scores_train).reshape(len(grid_params["C"]), len(grid_params["max_iter"]))

    # # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    # for idx, val in enumerate(grid_params["C"]):
    #     ax.plot(grid_params["max_iter"], scores_mean[idx, :], '-o', label="C" + ': ' + str(val))

    # # for idx, val in enumerate(grid_params["C"]):
    # #     ax.plot(grid_params["max_iter"], scores_train[idx, :], linestyle="dotted")

    # ax.set_xlabel("Max Iterations", fontsize=16)
    # ax.set_ylabel('CV Average Score', fontsize=16)
    # ax.legend(loc="best", fontsize=16)
    # ax.grid('on')
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)

    # result_path = os.path.join(
    #     directory_path, os.path.split(source_path)[0], "grid_search.png")
    # plt.savefig(result_path)
    # plt.figure()

    # csv_path = os.path.join(
    #     directory_path, os.path.split(source_path)[0], "grid_search.csv")
    # cv_results.to_csv(csv_path)
    results = cv_results.loc[cv_results['param_max_iter'] == 200]
    results.set_index('param_C')
    # results.rename(columns={"mean_test_score": "Mean Test Score", "mean_train_score": "Mean Train Score"})
    ax = results.plot.bar(x='param_C', y=['mean_test_score', 'mean_train_score'], fontsize=14)
    ax.set_xlabel("C value", fontsize=16)
    ax.set_ylabel("5-fold Mean Accuracy", fontsize=16)
    ax.legend(["Test Score", "Train Score"], loc="lower right", fontsize=16)
    plt.tight_layout()
    plt.show()
    return


plot_grid_search()
