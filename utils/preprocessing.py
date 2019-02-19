import nltk

def conll_dataset_to_word_AND_label_sents(dataset_name):
    """
    It transforms a dataset given by its name `dataset_name`
    (i.e. "train", "test" or "valid") into:
        - `word_sents` the list of sentences of words,
        - `label_sents` the list of sentences of the corresponding labels.
    """

    word_sents = []
    word_sent = []
    label_sents = []
    label_sent = []

    dataset_path = "storage/conll2003/{}.txt".format(dataset_name)
    with open(dataset_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()

            if len(line) == 0 or line.startswith('-DOCSTART'):
                if len(word_sent) > 0:
                    word_sents.append(word_sent)
                    word_sent = []
                    label_sents.append(label_sent)
                    label_sent = []
                continue

            parts = line.split(" ")
            word = parts[0]
            label = parts[-1]

            word_sent.append(word)
            label_sent.append(label)

    return word_sents, label_sents

def text_to_word_AND_pos_sents(text):
    sents = nltk.sent_tokenize(text)

    word_sents = []
    pos_sents = []
    offset = 0

    for sent in sents:
        word_sent = nltk.word_tokenize(sent)
        pos_sent = []

        for word in word_sent:
            start = text.find(word, offset)
            end = start + len(word)
            offset = end
            pos_sent.append([start, end])

        word_sents.append(word_sent)
        pos_sents.append(pos_sent)

    return word_sents, pos_sents