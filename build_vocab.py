import argparse

from utils import conll_dataset_to_word_AND_label_sents, save_vocab

# Arguments

parser = argparse.ArgumentParser()
parser.add_argument("--keep", type=float, default=0.9,
                    help="The percentage of words to keep. Words are ordered by decreasing frequency.")
args = parser.parse_args()

# Load the training dataset

word_sents, label_sents = conll_dataset_to_word_AND_label_sents("train")

# Count the number of occurences of each lowercased word

nb_occurs = {}
for word_sent in word_sents:
    for word in word_sent:
        lword = word.lower()
        if lword not in nb_occurs:
            nb_occurs[lword] = 0
        nb_occurs[lword] += 1

# Keep only the most frequent words
# This is done to improve generalization on never-seen-before words.

sorted_nb_occurs = sorted(nb_occurs.items(), key=lambda kv: kv[1], reverse=True)
sorted_nb_occurs = sorted_nb_occurs[:int(args.keep * len(nb_occurs))]

# Build and save vocabulary
# Rk : Id 0 is reserved for padding and id 1 for never-seen-before words.

vocab = {}
for i, (lword, nb_occurs) in enumerate(sorted_nb_occurs):
    vocab[lword] = i+2
save_vocab(vocab)