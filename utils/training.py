import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score

class F1Metrics(Callback):
    def on_epoch_end(self, batch, logs={}):
        val_target = self.validation_data[1].argmax(axis=2).flatten()
        val_predict = self.model.predict(self.validation_data[0]).argmax(axis=2).flatten()
        _val_f1 = f1_score(val_target, val_predict, average="weighted")
        print("— val f1: %f" % _val_f1)