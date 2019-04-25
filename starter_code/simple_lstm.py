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

    def train_lstm(self, feature_matr, labels):
        pass
        pass
        """
        embed_dim = 128
        lstm_out = 200
        batch_size = 32

        model = Sequential()
        model.add(Embedding(50, embed_dim,input_length = X.shape[1], dropout = 0.2))
        model.add(LSTM(lstm_out, dropout_U = 0.2, dropout_W = 0.2))
        model.add(Dense(2,activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
        """