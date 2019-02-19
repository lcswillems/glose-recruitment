import json

vocab_path = "storage/vocab.json"

def load_vocab():
    with open(vocab_path) as f:
        return json.load(f)

def save_vocab(vocab):
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)