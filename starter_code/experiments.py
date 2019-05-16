#from lstm import build_dataset, train_path, test_path, run_lstm, write_predictions, evaluate, pred_path, key_path, use_pre_processed_data


from lstm import set_params, save_constant_parameters, save_changing_param_and_results, run_experiment
import os
import datetime

# import all parameters that we might change in our experiments
#from lstm import FEATURES_TO_USE, THRESHOLD_OF_OCC, net_architecture, class_weights, model_params
'''
Default Params:

FEATURES_TO_USE = ['user', 'countries', 'client', 'session', 'format', 'token', 'time', 'days']
THRESHOLD_OF_OCC = 0

net_architecture = {.
    0: 128,
    1: 1
}

class_weights = {
    0: 15,
    1: 85.
}

model_params = {
    "batch_size": 64,  # number of samples in a batch
    "epochs": 20,  # number of epochs
    "time_steps": 100,  # how many time steps to look back to
    'activation': 'sigmoid',
    'optimizer': 'adam'
}
'''

def class_weights():
    '''
    function that runs an example experiment
    writes the used parameters and the results to the file "experiments/experiment_..."
    '''

    # set the name of the experiment
    now = datetime.datetime.now()
    experiment_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute)
    experiment_name = 'class_weights_one_user_' + str(experiment_id)

    # define if you want to use preprocessed data from file
    use_prep_data = False
    if use_prep_data:
        set_params(preproc_data_id='15_5_16.30.58')

    # define the changing parameter and its value
    changing_param_name = 'class_weights'
    changing_param_value = [{0:15, 1:85}, {0:5, 1:100}, {0:4, 1:100}, {0:3, 1:100}, {0:2, 1:100}, {0:1, 1:100}]

    # set constant parameters
    set_params(epochs=10)
    #
    #
    #...

    # save constant parameters to a new "experiment_.." file
    save_constant_parameters(experiment_name, changing_param_name)

    # run experiment for every parameter value
    for value in changing_param_value:
        # update the parameter value
        set_params(class_weights_1=value)

        # update the model_id for this new model
        now = datetime.datetime.now()
        new_model_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute) + "." + str(now.second)
        set_params(model_id = new_model_id)

        # evaluate the new model
        results = run_experiment()

        # save results to the experiment file
        save_changing_param_and_results(experiment_name, new_model_id, changing_param_name, value, results)

        if value == changing_param_value[0]:
            set_params(preproc_data_id=new_model_id)



# specify which experiment you want to run
if __name__ == '__main__':
    class_weights()

