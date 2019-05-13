from lstm import MODEL_ID, FEATURES_TO_USE, THRESHOLD_OF_OCC, model_params, main, net_architecture, class_weight, use_pre_processed_data, use_pre_trained_model
import os
'''
Default Params:

FEATURES_TO_USE = ['user', 'countries', 'client', 'session', 'format', 'token', 'time', 'days']
THRESHOLD_OF_OCC = 0

net_architecture = {
    0: 128,
    1: 1
}

class_weight = {
    0: 1.,
    1: 50.
}

model_params = {
    "batch_size": 64,  # number of samples in a batch
    "lr": 0.01,  # learning rate
    "epochs": 20,  # number of epochs
    "time_steps": 100,  # how many time steps to look back to
    'activation': 'sigmoid',
    'optimizer': 'adam'
}

'''

def experiment_0():
    use_pre_processed_data = False
    use_pre_processed_data = False
    model_params['epochs'] = 1

    main()


def save_model(self, model_id):
    """
    Save current model with weights
    """
    if not os.path.exists("models/"):
        os.makedirs("models/")

    self.model.save("models/parameters_" + model_id)


# spexify which experiment you want to run
if __name__ == '__main__':
    experiment_0()

