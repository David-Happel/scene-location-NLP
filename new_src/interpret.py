def interpret():
    script_path = os.path.abspath(__file__)  # path to python script
    directory_path = os.path.dirname(os.path.split(
        script_path)[0])  # path to python script dir
    # source_path = '/home/david/Projects/NLP-project/reports/report-2020-06-01-17-46-32/tfidf_grid_search.pkl'
    source_path = '/home/david/Projects/NLP-project/reports/report-2020-06-01-17-30-34/elmo_grid_search.pkl'

    cv_results = pd.read_pickle(source_path)
    return


interpret()
