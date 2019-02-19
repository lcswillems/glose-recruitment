import json
import os
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from utils import load_vocab, conll_dataset_to_word_AND_label_sents, text_to_word_AND_pos_sents

def word_to_casing(word):
    # Function taken from: https://github.com/kamalkraj/Named-Entity-Recognition-with-Bidirectional-LSTM-CNNs/blob/master/prepro.py#L28

    num_digits = 0
    for char in word:
        if char.isdigit():
            num_digits += 1

    digit_fraction = num_digits / float(len(word))

    if word.isdigit(): # Is a digit
        casing = 'numeric'
    elif digit_fraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower(): # All lower case
        casing = 'all_lower'
    elif word.isupper(): # All upper case
        casing = 'all_upper'
    elif word[0].isupper(): # Is a title, initial char upper, then all lower
        casing = 'initial_upper'
    elif num_digits > 0:
        casing = 'contains_digit'
    else:
        casing = 'other'

    return casing

# Rk: Ids start at 1 because 0 is reserved for padding.
casing_to_id = {
    'numeric': 1,
    'all_lower': 2,
    'all_upper': 3,
    'initial_upper': 4,
    'other': 5,
    'mainly_numeric': 6,
    'contains_digit': 7
}

label_to_hot = {
    "O": [1, 0, 0, 0, 0, 0, 0, 0, 0],
    "B-ORG": [0, 1, 0, 0, 0, 0, 0, 0, 0],
    "I-ORG": [0, 0, 1, 0, 0, 0, 0, 0, 0],
    "B-MISC": [0, 0, 0, 1, 0, 0, 0, 0, 0],
    "I-MISC": [0, 0, 0, 0, 1, 0, 0, 0, 0],
    "B-PER": [0, 0, 0, 0, 0, 1, 0, 0, 0],
    "I-PER": [0, 0, 0, 0, 0, 0, 1, 0, 0],
    "B-LOC": [0, 0, 0, 0, 0, 0, 0, 1, 0],
    "I-LOC": [0, 0, 0, 0, 0, 0, 0, 0, 1]
}

id_to_label = {
    0: "O",
    1: "B-ORG",
    2: "I-ORG",
    3: "B-MISC",
    4: "I-MISC",
    5: "B-PER",
    6: "I-PER",
    7: "B-LOC",
    8: "I-LOC",
}

def word_sents_to_lword_id_AND_casing_id_sents(word_sents):
    vocab = load_vocab()

    lword_id_sents = []
    casing_id_sents = []

    for word_sent in word_sents:
        lword_id_sent = []
        casing_id_sent = []

        for word in word_sent:
            lword = word.lower()
            lword_id = vocab[lword] if lword in vocab else 1
            lword_id_sent.append(lword_id)

            casing_id = casing_to_id[word_to_casing(word)]
            casing_id_sent.append(casing_id)

        lword_id_sents.append(lword_id_sent)
        casing_id_sents.append(casing_id_sent)

    # Pad sentences
    max_sent_len = len(max(word_sents, key=len))
    lword_id_sents = pad_sequences(lword_id_sents, maxlen=max_sent_len)
    casing_id_sents = pad_sequences(casing_id_sents, maxlen=max_sent_len)

    return lword_id_sents, casing_id_sents

def label_sents_to_label_hot_sents(label_sents):
    label_hot_sents = []

    for label_sent in label_sents:
        label_hot_sent = []

        for label in label_sent:
            label_hot = label_to_hot[label]
            label_hot_sent.append(label_hot)

        label_hot_sents.append(label_hot_sent)

    # Pad sentences
    max_sent_len = len(max(label_sents, key=len))
    label_hot_sents = pad_sequences(label_hot_sents, maxlen=max_sent_len)

    return label_hot_sents

def preprocess_conll_dataset(dataset_name):
    word_sents, label_sents = conll_dataset_to_word_AND_label_sents(dataset_name)
    lword_id_sents, casing_id_sents = word_sents_to_lword_id_AND_casing_id_sents(word_sents)
    label_hot_sents = label_sents_to_label_hot_sents(label_sents)
    return lword_id_sents, casing_id_sents, label_hot_sents

def preprocess_text(text):
    word_sents, pos_sents = text_to_word_AND_pos_sents(text)
    lword_id_sents, casing_id_sents = word_sents_to_lword_id_AND_casing_id_sents(word_sents)
    return lword_id_sents, casing_id_sents, pos_sents