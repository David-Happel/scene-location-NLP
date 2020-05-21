from functools  import reduce
import re
import numpy as np

def splitkeepsep(s, sep):
    return reduce(lambda acc, elem: acc[:-1] + [acc[-1] + elem] if elem == sep else acc + [elem], re.split("(%s)" % re.escape(sep), s), [])

def clean(data, should_split_lines=True, should_remove_brackts=True, minimum_words=3, should_remove_punctuation=True):
    data["clean_line"] = data["line"]
    if should_split_lines:
        data = split_lines(data)
    if should_remove_brackts:
        data = remove_brackets(data)
    if minimum_words:
        data = remove_shorter_than(data, minimum_words)
    if should_remove_punctuation:
        data = remove_punctuation(data)
    data = int_location(data).reset_index(drop=True)
    return data

def split_lines(data):
    data["clean_line"] = data["clean_line"].apply(lambda line: splitkeepsep(line, '[.?!]'))
    data = data.explode("clean_line").sample(frac=1).reset_index(drop=True)
    data = data.replace('', np.nan).dropna()
    return data

def remove_brackets(data):
    data['clean_line'] = data['clean_line'].str.replace('\((.*?)\)', '', regex=True)
    return  data

def remove_numbers(data):
    data['clean_line'] = data['clean_line'].str.replace("[0-9]", " ")
    return data

def remove_multiple_spaces(data):
    data['clean_line'] = data['clean_line'].apply(lambda x: ' '.join(x.split()))
    return data

def remove_shorter_than(data, minimum_words):
    data = data[data['clean_line'].str.count(' ') >= (minimum_words-1)]
    return data

def remove_punctuation(data):
    punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`\'{|}~.,'
    data['clean_line'] = data['clean_line'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
    return data

def int_location(data):
    locations = data.location.unique()

    data.loc[data["location"]==locations[0],"location_int"]=0
    data.loc[data["location"]==locations[1],"location_int"]=1
    return data
