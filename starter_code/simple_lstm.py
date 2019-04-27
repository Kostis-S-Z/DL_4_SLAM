import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, TimeDistributed


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
                feature_attribute, feature_value = train_feature.split(":", 1)
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
        timesteps = 32
        lstm_layer_size = 100

        X_train =X_train, y_train = data_in_time(X_train, y_train, timesteps)
        #X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))

        model = Sequential()
        # ensure that the LSTM cell returns all of the outputs from the unrolled LSTM cell through time. If this
        # argument is left out, the LSTM cell will simply provide the output of the LSTM cell from the last time step.
        # return_sequences = True. We want this because we want to output correct/incorrect for each word, not just
        # at the end of the exercise one correct/incorrect
        model.add(LSTM(lstm_layer_size, return_sequences=False, input_shape = (timesteps, X_train.shape[2])))
        #model.add(LSTM(lstm_layer_size, return_sequences=True, input_shape=(timesteps, lstm_layer_size)))
        # This function adds an independent layer for each time step in the recurrent model. So, for instance, if we
        # have 10 time steps in a model, a TimeDistributed layer operating on a Dense layer would produce 10 independent
        # Dense layers, one for each time step.
        #, input_shape = (0, lstm_layer_size)
        #model.add(TimeDistributed(Dense(X_train.shape[2], activation='sigmoid')))
        model.add(Dense(1, activation='sigmoid', input_shape = (0, lstm_layer_size)))
        # binary because we're doing binary classification (correct / incorrect
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_train, y_train), epochs=10, batch_size=100)
        Y_pred = model.predict(X_train)
        #print("prediction: ", Y_pred)
        print(model.summary())


def data_in_time(data_X, data_y, time):
    list = []
    for i in range(len(data_X) - time + 1):
        list.append(data_X[i:i + time])

    data_X = np.reshape( np.concatenate(list), (data_X.shape[0] - time + 1, time, data_X.shape[1]))

    data_y = data_y[time - 1:len(data_y)]
    return data_X, data_y