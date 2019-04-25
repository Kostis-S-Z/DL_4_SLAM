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

    def oh_enc(self, feature_dicts, count_dict, start_index_in_dict):
        print(feature_dicts)
        print("count_dict", count_dict)
        print("start_index_in_dict", start_index_in_dict)
        one_hot_vec = np.zeros((len(feature_dicts),sum([pair[0] for pair in count_dict.values()])))

        for i, feature_dict in enumerate(feature_dicts):
            for feature_and_value in feature_dict.keys():
                feature, feature_value = feature_and_value.split(":")
                print("feature and value", feature, feature_value)
                index = start_index_in_dict[feature] + count_dict[feature][1][feature_value]
                print("index of our 1!!! ", index)
                one_hot_vec[i, index] = 1
                print(one_hot_vec)


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