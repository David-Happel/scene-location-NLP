from functools import reduce
import re
import numpy as np
import spacy


def splitkeepsep(s, sep):
    return reduce(lambda acc, elem: acc[:-1] + [acc[-1] + elem] if elem == sep else acc + [elem], re.split("(%s)" % re.escape(sep), s), [])


def clean(data, options={"split_lines": True, "remove_brackets": True, "remove_numbers": True, "minimum_words": 3, "remove_punctuation": True, "lemmatize": True, "remove_pron": True}):
    data["clean_line"] = data["line"]
    if options["split_lines"]:
        data = split_lines(data)
    if options["lemmatize"]:
        # import spaCy's language model
        nlp = spacy.load('en', disable=['parser', 'ner'])
        data['clean_line'] = lemmatization(data['clean_line'], nlp)

        if options["remove_pron"]:
            data['clean_line'] = data['clean_line'].str.replace("-PRON-", " ")

    if options["remove_brackets"]:
        data = remove_brackets(data)
    if options["remove_numbers"]:
        data = remove_numbers(data)
    data = remove_multiple_spaces(data)
    if options["minimum_words"]:
        data = remove_shorter_than(data, options["minimum_words"])
    if options["remove_punctuation"]:
        data = remove_punctuation(data)
    data = int_location(data).reset_index(drop=True)
    return data


def split_lines(data):
    data["clean_line"] = data["clean_line"].apply(
        lambda line: splitkeepsep(line, '[.?!]'))
    data = data.explode("clean_line").sample(frac=1).reset_index(drop=True)
    data = data.replace('', np.nan).dropna()
    return data


def remove_brackets(data):
    data['clean_line'] = data['clean_line'].str.replace(
        '\((.*?)\)', '', regex=True)
    return data


def remove_numbers(data):
    data['clean_line'] = data['clean_line'].str.replace("[0-9]", " ")
    return data


def remove_multiple_spaces(data):
    data['clean_line'] = data['clean_line'].apply(
        lambda x: ' '.join(x.split()))
    return data


def remove_shorter_than(data, minimum_words):
    data = data[data['clean_line'].str.count(' ') >= (minimum_words-1)]
    return data


def remove_punctuation(data):
    punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`\'’{|}~.,'
    data['clean_line'] = data['clean_line'].apply(
        lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
    return data


def int_location(data):
    locations = data.location.unique()

    data.loc[data["location"] == locations[0], "location_int"] = 0
    data.loc[data["location"] == locations[1], "location_int"] = 1
    return data


# function to lemmatize text
def lemmatization(texts, nlp):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output
