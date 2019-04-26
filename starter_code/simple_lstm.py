import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM


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


    def train(self, features_one_hot, labels):
        """
        train the RNN
        """
        labels = np.array(labels)
        self.train_lstm(features_one_hot, labels)

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

    def train_lstm(self, X_train, y_train):
        """
        shape X_train = (n, one_hot_m)
        shape y_train = (n, )
        """
        timesteps = 1
        lstm_layer_size = 10
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        model = Sequential()
        model.add(LSTM(lstm_layer_size, return_sequences=False, input_shape = (timesteps, X_train.shape[2])))
        model.add(Dense(1, activation='sigmoid', input_shape = (0, lstm_layer_size)))
        # binary because we're doing binary classification (correct / incorrect
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_train, y_train), epochs=3, batch_size=4)
        Y_pred = model.predict(X_train)
        print("prediction: ", Y_pred)
        print(model.summary())
