import spacy


def clean_data(data, remove_punctuation=True, remove_numbers=True, lemmatize=True, remove_pron=True, minimum_words=3):
    if remove_punctuation:
        punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`\'{|}~.,'
        data['clean_line'] = data['line'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

    if remove_numbers:
        data['clean_line'] = data['clean_line'].str.replace("[0-9]", " ")

    if lemmatize:
        # import spaCy's language model
        nlp = spacy.load('en', disable=['parser', 'ner'])
        data['clean_line'] = lemmatization(data['clean_line'], nlp)

        if remove_pron:
            data['clean_line'] = data['clean_line'].str.replace("-PRON-", "")

    data['clean_line'] = data['clean_line'].apply(lambda x: ' '.join(x.split()))
    data = data[data['clean_line'].str.count(' ') >= (minimum_words-1)]

    return data


# function to lemmatize text
def lemmatization(texts, nlp):
    output = []
    for i in texts:
        s = [token.lemma_ for token in nlp(i)]
        output.append(' '.join(s))
    return output
