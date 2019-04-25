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


    def train(self, oh_features, labels):
        """
        train the RNN
        """
        for i in range(len(labels)):
            print("WORD ", i)
            print("one hot features", oh_features[i])
            print("label: ", labels[i])

        self.train_lstm(oh_features, labels)
        print("Done straining (if there was actually training code yet")

    def one_hot_encode(self, training_data, feature_index_dict, n_features):
        """
        !!!WE ARE MISSING OUT A LOT OF TRAINING INSTANCES BECAUSE THEY ARE NOT INCLUDED IN training_data!!!
        print("amount of training instances:", len(training_data))
        """

        one_hot_vec = np.zeros((len(training_data),n_features))

        # for all training examples compute one hot encoding
        for i, training_example in enumerate(training_data):
            for train_feature in training_example.keys():
                feature_attribute, feature_value = train_feature.split(":")
                index_attribute = feature_index_dict[feature_attribute][0]
                index_value = feature_index_dict[feature_attribute][1][feature_value]
                index = index_attribute + index_value
                one_hot_vec[i, index] = 1

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