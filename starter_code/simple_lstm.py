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

    def train(self, feature_dicts, labels, names):
        """
        train the RNN
        """
        for i in range(30):
            print("--------------- WORD ", i, "--------------------")
            print("features: ")
            print(training_data[i * 10].to_features())
            print("label: ")
            print(training_labels[training_data[i * 10].instance_id])
            print("name: ")
            print(training_data[i * 10].instance_id)
            print("labels 1 to 40: ")
            print(training_labels[training_data[j].instance_id])

        feature_matr = self.oh_enc(feature_dicts)
        self._train_lstm(feature_matr, labels)
        print("Done training (if there was actually training code yet")

    def oh_enc(self, feature_dicts):
        one_hot_vec = np.ones((3,3))
        return one_hot_vec
        #return one_hot_vec

    def train_lstm(self, feature_matr, labels):
        pass