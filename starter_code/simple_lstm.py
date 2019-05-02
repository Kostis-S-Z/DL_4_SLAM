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

    def data_in_time(self, data_x, data_y=None):
        data_list = []
        for i in range(len(data_x) - self.timesteps + 1):
            data_list.append(data_x[i:i + self.timesteps])

        data_x = np.reshape(np.concatenate(data_list), (data_x.shape[0] - self.timesteps + 1, self.timesteps, data_x.shape[1]))
        if data_y is not None:
            data_y = data_y[self.timesteps - 1:len(data_y)]
            return data_x, data_y
        else:
            return data_x

    def train(self, x_train, y_train, verbose=2):
        """
        shape X_train = (n, one_hot_m)
        shape y_train = (n, )
        """
        y_train = np.array(y_train)
        lstm_layer_size = 100
        x_train, y_train = self.data_in_time(x_train, y_train)
        # return_sequences = True. ensure that the LSTM cell returns all of the outputs from the unrolled
        # LSTM cell through time. If this argument is left out, the LSTM cell will simply provide the
        # output of the LSTM cell from the last time step.
        #  We want this because we want to output correct/incorrect for each word, not just
        # at the end of the exercise one correct/incorrect
        self.model.add(LSTM(lstm_layer_size, return_sequences=False, input_shape = (self.timesteps, x_train.shape[2])))
        # a second LSTM layer
        # self.model.add(LSTM(lstm_layer_size, return_sequences=True, input_shape=(self.timesteps, lstm_layer_size)))
        self.model.add(Dense(1, activation='sigmoid', input_shape = (0, lstm_layer_size)))
        # binary because we're doing binary classification (correct / incorrect
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(x_train, y_train, validation_data=(x_train, y_train), epochs=1, batch_size=100, verbose=verbose)
        if verbose > 0:
            print(self.model.summary())

    def predict(self, data_x, ids):
        """
        make predictions for one-hot encoded feature vector X using the model.
        Ofcourse, it is useful if the model is trained before making predictions.
        """
        print("shape X: ", data_x.shape)
        data_x = self.data_in_time(data_x)
        data_y = self.model.predict(data_x)
        print("shape Y: ", data_y.shape)
        print("length id_list: ", len(ids))
        pred_dict = {}
        for i in range(len(ids)-self.timesteps):
            pred_dict[ids[i+self.timesteps]] = float(data_y[i])
        return pred_dict
