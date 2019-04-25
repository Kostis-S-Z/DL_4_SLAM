import numpy as np

class SimpleLstm:

    def __init__(self, **kwargs):
        """
        Initialize Neural Network with data and parameters
        """
        var_defaults = {
            "lr": 0.01,  # learning rate
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

    def train(self, feature_dicts, labels):
        """
        train the RNN
        """
        for i in range(len(feature_dicts)):
            print("WORD ", i)
            print(feature_dicts[i])
            print(labels[i])

        feature_matr = self.oh_enc(feature_dicts)
        self.train_lstm(feature_matr, labels)
        print("Done training (if there was actually training code yet")

    def oh_enc(self, feature_dicts):
        one_hot_vec = np.ones((3,3))
        return one_hot_vec
        #return one_hot_vec

    def train_lstm(self, feature_matr, labels):
        pass