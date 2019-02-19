import argparse
import os
from keras.models import load_model, Sequential
from keras.layers import Dense, LSTM, Embedding, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from utils import load_vocab, get_model_path, F1Metrics
from preprocess import preprocess_conll_dataset, casing_to_id, label_to_hot

# Arguments

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    help="Model name")
parser.add_argument("--lr", type=float, default=0.001,
                    help="Learning rate")
parser.add_argument("--epochs", type=int, default=1,
                    help="Number of epochs")
parser.add_argument("--bs", type=int, default=128,
                    help="Batch size")
args = parser.parse_args()

# Load training and validation data

train_lword_id_sents, train_casing_id_sents, train_label_hot_sents = preprocess_conll_dataset("train")
valid_lword_id_sents, valid_casing_id_sents, valid_label_hot_sents = preprocess_conll_dataset("valid")

# Define model architecture parameters

lword_embedding_input_dim = len(load_vocab()) + 2
lword_embedding_output_dim = 64
casing_embedding_input_dim = len(casing_to_id) + 1
casing_embedding_output_dim = 8

lstm_output_dim = 16

nb_labels = len(label_to_hot)

# Define model

model_path = get_model_path(args.model)

if os.path.exists(model_path):
    model = load_model(model_path)
else:
    model = Sequential()
    model.add(Embedding(input_dim=casing_embedding_input_dim,
                        output_dim=casing_embedding_output_dim,
                        mask_zero=True))
    model.add(Bidirectional(LSTM(lstm_output_dim, return_sequences=True)))
    model.add(Dense(nb_labels, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=args.lr), metrics=["acc"])

model.summary()

# Train model

f1_callback = F1Metrics()
model_callback = ModelCheckpoint(model_path, save_best_only=True)

model.fit(train_casing_id_sents, train_label_hot_sents,
          epochs=args.epochs, batch_size=args.bs, verbose=1,
          validation_data=(valid_casing_id_sents, valid_label_hot_sents),
          callbacks=[model_callback, f1_callback])