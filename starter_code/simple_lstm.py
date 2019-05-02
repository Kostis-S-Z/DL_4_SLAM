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

        self.model = Sequential()
        self.timesteps = 22


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


    def data_in_time(self, data_X, data_y=None):
        list = []
        for i in range(len(data_X) - self.timesteps + 1):
            list.append(data_X[i:i + self.timesteps])

        data_X = np.reshape(np.concatenate(list), (data_X.shape[0] - self.timesteps + 1, self.timesteps, data_X.shape[1]))
        if data_y is not None:
            data_y = data_y[self.timesteps - 1:len(data_y)]
            return data_X, data_y
        else:
            return data_X


    def train(self, X_train, y_train, verbose=2):
        """
        shape X_train = (n, one_hot_m)
        shape y_train = (n, )
        """
        y_train = np.array(y_train)
        lstm_layer_size = 100
        X_train, y_train = self.data_in_time(X_train, y_train)
        # return_sequences = True. ensure that the LSTM cell returns all of the outputs from the unrolled
        # LSTM cell through time. If this argument is left out, the LSTM cell will simply provide the
        # output of the LSTM cell from the last time step.
        #  We want this because we want to output correct/incorrect for each word, not just
        # at the end of the exercise one correct/incorrect
        self.model.add(LSTM(lstm_layer_size, return_sequences=False, input_shape = (self.timesteps, X_train.shape[2])))
        # a second LSTM layer
        #self.model.add(LSTM(lstm_layer_size, return_sequences=True, input_shape=(self.timesteps, lstm_layer_size)))
        self.model.add(Dense(1, activation='sigmoid', input_shape = (0, lstm_layer_size)))
        # binary because we're doing binary classification (correct / incorrect
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(X_train, y_train, validation_data=(X_train, y_train), epochs=10, batch_size=100, verbose=verbose)
        if verbose > 0:
            print(self.model.summary())


    def predict(self, X, ids):
        """
        make predictions for one-hot encoded feature vector X using the model.
        Ofcourse, it is useful if the model is trained before making predictions.
        """
        print("shape X: ", X.shape)
        X = self.data_in_time(X)
        Y =self.model.predict(X)
        print("shape Y: ", Y.shape)
        print("length id_list: ", len(ids))
        pred_dict = {}
        for i in range(len(ids)-self.timesteps):
            pred_dict[ids[i+self.timesteps]] = float(Y[i])
        return pred_dict
