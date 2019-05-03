import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, LSTM, TimeDistributed


class SimpleLstm:

    def __init__(self, **kwargs):
        """
        Initialize Neural Network with data and parameters
        """
        var_defaults = {
            "lr": 0.01,  # learning rate
            "time_steps": 5  # how many time steps to look back to
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.model = None

    def data_in_time(self, data_x, data_y=None):
        # TODO: Comment these method nalytically
        data_list = []
        for i in range(len(data_x) - self.time_steps + 1):
            data_list.append(data_x[i:i + self.time_steps])

        data_x = np.reshape(np.concatenate(data_list), (data_x.shape[0] - self.time_steps + 1, self.time_steps, data_x.shape[1]))
        if data_y is not None:
            data_y = data_y[self.time_steps - 1:len(data_y)]
            return data_x, data_y
        else:
            return data_x

    def init_model(self, inp_shape, verbose):
        lstm_layer_size = 100
        # return_sequences = True. ensure that the LSTM cell returns all of the outputs from the unrolled
        # LSTM cell through time. If this argument is left out, the LSTM cell will simply provide the
        # output of the LSTM cell from the last time step.
        #  We want this because we want to output correct/incorrect for each word, not just
        # at the end of the exercise one correct/incorrect
        model = Sequential()
        model.add(LSTM(lstm_layer_size, return_sequences=False, input_shape=(self.time_steps, inp_shape)))

        # a second LSTM layer
        # self.model.add(LSTM(lstm_layer_size, return_sequences=True, input_shape=(self.timesteps, lstm_layer_size)))
        model.add(Dense(1, activation='sigmoid', input_shape = (0, lstm_layer_size)))
        # binary because we're doing binary classification (correct / incorrect

        if verbose > 0:
            print(model.summary())

        return model

    def train(self, x_train, y_train, epochs=2, batch_size=100, verbose=2, model=None):
        """
        shape X_train = (n, one_hot_m)
        shape y_train = (n, )
        """
        y_train = np.array(y_train)
        x_train, y_train = self.data_in_time(x_train, y_train)
        # TODO: change these
        x_val = x_train
        y_val = y_train
        if model is None:
            model = self.init_model(x_train.shape[2], verbose)
        else:
            print("Loading pre-existing model...")
            model = self.load_model()

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Fit the training data to the model
        model.fit(x_train, y_train,
                  validation_data=(x_val, y_val),
                  shuffle=False,
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=verbose)

        # save the model to a class variable for further use afterwards (only reading from this variable, no changing)
        self.model = model

    def predict(self, test_data, ids):
        """
        make predictions for one-hot encoded feature vector X using the model.
        Ofcourse, it is useful if the model is trained before making predictions.
        """
        # Format the data in our way
        test_data = self.data_in_time(test_data)
        # Make predictions
        pred_labels = self.model.predict(test_data)

        # TODO: Comment the next 3 lines analytically
        pred_dict = {}
        for i in range(len(ids)-self.time_steps):
            pred_dict[ids[i + self.time_steps]] = float(pred_labels[i])

        return pred_dict

    def save_model(self):
        """
        Save current model with weights
        """
        self.model.save("model.h5")

    def load_model(self):
        """
        Load a model
        """
        return load_model("model.h5")
