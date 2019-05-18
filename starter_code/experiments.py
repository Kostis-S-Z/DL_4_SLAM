#from lstm import build_dataset, train_path, test_path, run_lstm, write_predictions, evaluate, pred_path, key_path, use_pre_processed_data


from lstm import set_params, save_constant_parameters, run_experiment
import os
import datetime

from multiprocessing import Process
import psutil

import shutil

# import all parameters that we might change in our experiments
#from lstm import FEATURES_TO_USE, THRESHOLD_OF_OCC, net_architecture, class_weights, model_params
'''
Default Params:

FEATURES_TO_USE = ['user', 'countries', 'client', 'session', 'format', 'token', 'time', 'days']
THRESHOLD_OF_OCC = 0

# Define the number of nodes in each layer, the last one is the output
net_architecture = {
    0: 128,
    1: 1
}
class_weights = {
    0: 15,
    1: 85
}

model_params = {
    "batch_size": 64,  # number of samples in a batch
    "epochs": 20,  # number of epochs
    "lr": 0.001,
    "time_steps": 60,  # how many time steps to look back to
    'activation': 'sigmoid',
    'dropout': 0.4,
    'recurrent_dropout': 0.1
}

USE_WORD_EMB = 0
'''

def one_experiment():
    '''
    function that runs an example experiment
    writes the used parameters and the results to the file "experiments/experiment_..."
    '''

    # set the name of the experiment
    now = datetime.datetime.now()
    experiment_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute)
    experiment_name = 'overfit_' + str(experiment_id)

    # define if you want to use preprocessed data from file
    use_prep_data = False
    if use_prep_data:
        set_params(preproc_data_id='16_5_10.16.47')

    # define the changing parameter and its value
    changing_param_name = 'class_weights'
    changing_param_value = [{0:15, 1:85}]#, {0:15, 1:85}]#, {0:4, 1:100}, {0:3, 1:100}, {0:2, 1:100}, {0:1, 1:100}] #[{0:1, 1:1}, {0:15, 1:85}]#

    # set constant parameters
    set_params(use_word_emb=0)
    set_params(epochs=10)
    #
    #
    #...

    # save constant parameters to a new "experiment_.." filgithx+P@2ub
    save_constant_parameters(experiment_name, changing_param_name)

    # run experiment for every parameter value
    for value in changing_param_value:
        process = psutil.Process(os.getpid())
        print("-----MEMORY before starting experiment ------", int(process.memory_info().rss/(8*10**(3))), "KB")

        # update the parameter value
        set_params(class_weights_1=value)

        # update the model_id for this new model
        now = datetime.datetime.now()
        new_model_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute) + "." + str(now.second)
        set_params(model_id = new_model_id)

        # evaluate the new model and save the results in the experiment file
        oneExperiment = Process(target=run_experiment, args=(experiment_name, new_model_id, changing_param_name, value,))
        oneExperiment.start()
        oneExperiment.join()

def class_weights_binary():
    '''
    function that runs an example experiment
    writes the used parameters and the results to the file "experiments/experiment_..."
    '''

    # set the name of the experiment
    now = datetime.datetime.now()
    experiment_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute)
    experiment_name = 'class_weights_binary_' + str(experiment_id)

    # define if you want to use preprocessed data from file
    use_prep_data = False
    if use_prep_data:
        set_params(preproc_data_id='16_5_10.16.47')

    # define the changing parameter and its value
    changing_param_name = 'class_weights'
    changing_param_value = [{0:1, 1:2}, {0:15, 1:85}]#, {0:4, 1:100}, {0:3, 1:100}, {0:2, 1:100}, {0:1, 1:100}] #[{0:1, 1:1}, {0:15, 1:85}]#

    # set constant parameters
    set_params(use_word_emb=0)
    #
    #
    #...

    # save constant parameters to a new "experiment_.." file
    save_constant_parameters(experiment_name, changing_param_name)

    # run experiment for every parameter value
    for value in changing_param_value:
        process = psutil.Process(os.getpid())
        print("-----MEMORY before starting experiment ------", int(process.memory_info().rss/(8*10**(3))), "KB")

        # update the parameter value
        set_params(class_weights_1=value)

        # update the model_id for this new model
        now = datetime.datetime.now()
        new_model_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute) + "." + str(now.second)
        set_params(model_id = new_model_id)

        # evaluate the new model and save the results in the experiment file
        oneExperiment = Process(target=run_experiment, args=(experiment_name, new_model_id, changing_param_name, value,))
        oneExperiment.start()
        oneExperiment.join()

        set_params(preproc_data_id=new_model_id)

def class_weights_embedding():
    '''
    function that runs an example experiment
    writes the used parameters and the results to the file "experiments/experiment_..."
    '''

    # set the name of the experiment
    now = datetime.datetime.now()
    experiment_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute)
    experiment_name = 'class_weights_embedding_' + str(experiment_id)

    # define if you want to use preprocessed data from file
    use_prep_data = False
    if use_prep_data:
        set_params(preproc_data_id='16_5_10.16.47')
    else:
        set_params(use_preproc_data=False)

    # define the changing parameter and its value
    changing_param_name = 'class_weights'
    changing_param_value = [{0:1, 1:2}, {0:15, 1:85}]#, {0:4, 1:100}, {0:3, 1:100}, {0:2, 1:100}, {0:1, 1:100}] #[{0:1, 1:1}, {0:15, 1:85}]#

    # set constant parameters
    set_params(use_word_emb=1)
    #
    #
    #...

    # save constant parameters to a new "experiment_.." file
    save_constant_parameters(experiment_name, changing_param_name)

    # run experiment for every parameter value
    for value in changing_param_value:
        process = psutil.Process(os.getpid())
        print("-----MEMORY before starting experiment ------", int(process.memory_info().rss/(8*10**(3))), "KB")

        # update the parameter value
        set_params(class_weights_1=value)

        # update the model_id for this new model
        now = datetime.datetime.now()
        new_model_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute) + "." + str(now.second)
        set_params(model_id = new_model_id)

        # evaluate the new model and save the results in the experiment file
        oneExperiment = Process(target=run_experiment, args=(experiment_name, new_model_id, changing_param_name, value,))
        oneExperiment.start()
        oneExperiment.join()

        set_params(preproc_data_id=new_model_id)

def lr_experiment():
    '''
    function that runs an example experiment
    writes the used parameters and the results to the file "experiments/experiment_..."
    '''
    print("LR_EXPERIMENT\n")

    # set the name of the experiment
    now = datetime.datetime.now()
    experiment_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute)
    experiment_name = 'lr_' + str(experiment_id)

    # define if you want to use preprocessed data from file
    use_prep_data = False
    if use_prep_data:
        set_params(preproc_data_id='16_5_10.16.47')
    else:
        set_params(use_preproc_data=False)

    # define the changing parameter and its value
    changing_param_name = 'lr'
    changing_param_value = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03,  0.1]

    # set constant parameters
    set_params(use_word_emb=1)
    set_params(epochs=20)
    #
    #
    #...

    # save constant parameters to a new "experiment_.." file
    save_constant_parameters(experiment_name, changing_param_name)


    # run experiment for every parameter value
    for value in changing_param_value:
        # update the changing parameter value
        set_params(lr = value)

        # update the model_id for this new model
        now = datetime.datetime.now()
        new_model_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute) + "." + str(now.second)
        set_params(model_id = new_model_id)

        # evaluate the new model and save the results in the experiment file
        oneExperiment = Process(target=run_experiment, args=(experiment_name, new_model_id, changing_param_name, value,))
        oneExperiment.start()
        oneExperiment.join()

        if value == changing_param_value[0]:
            set_params(preproc_data_id=new_model_id)

def timesteps_experiment():
    '''
    function that runs an example experiment
    writes the used parameters and the results to the file "experiments/experiment_..."
    '''
    print("TIMESTEPS EXPERIMENT")

    # set the name of the experiment
    now = datetime.datetime.now()
    experiment_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute)
    experiment_name = 'timestep_' + str(experiment_id)

    # define if you want to use preprocessed data from file
    use_prep_data = False
    if use_prep_data:
        set_params(preproc_data_id='16_5_10.16.47')
    else:
        set_params(use_preproc_data=False)

    # define the changing parameter and its value
    changing_param_name = 'time_steps'
    changing_param_value = [1,2,4,8,16,32,64,128,256]#, {0:4, 1:100}, {0:3, 1:100}, {0:2, 1:100}, {0:1, 1:100}] #[{0:1, 1:1}, {0:15, 1:85}]#

    # set constant parameters
    set_params(epochs=20)
    set_params(dropout=0.3)
    set_params(use_word_emb = 1)
    #
    #
    #...

    # save constant parameters to a new "experiment_.." file
    save_constant_parameters(experiment_name, changing_param_name)


    # run experiment for every parameter value
    for value in changing_param_value:
        process = psutil.Process(os.getpid())
        print("-----MEMORY before starting experiment ------", int(process.memory_info().rss/(8*10**(3))), "KB")

        # update the parameter value
        set_params(use_word_emb = value)

        # update the model_id for this new model
        now = datetime.datetime.now()
        new_model_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute) + "." + str(now.second)
        set_params(model_id = new_model_id)

        # evaluate the new model and save the results in the experiment file
        oneExperiment = Process(target=run_experiment, args=(experiment_name, new_model_id, changing_param_name, value,))
        oneExperiment.start()
        oneExperiment.join()

        if value == changing_param_value[0]:
            set_params(preproc_data_id=new_model_id)

def emb_experiment():
    '''
    function that runs an example experiment
    writes the used parameters and the results to the file "experiments/experiment_..."
    '''
    print("EMBEDDINGS EXPERIMENT")

    # set the name of the experiment
    now = datetime.datetime.now()
    experiment_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute)
    experiment_name = 'emb_' + str(experiment_id)

    # define if you want to use preprocessed data from file
    use_prep_data = False
    if use_prep_data:
        set_params(preproc_data_id='16_5_10.16.47')
    else:
        set_params(use_preproc_data=False)

    # define the changing parameter and its value
    changing_param_name = 'use_word_emb'
    changing_param_value = [0,1]#, {0:4, 1:100}, {0:3, 1:100}, {0:2, 1:100}, {0:1, 1:100}] #[{0:1, 1:1}, {0:15, 1:85}]#

    # set constant parameters
    set_params(epochs=20)
    set_params(dropout=0.3)
    #
    #
    #...

    # save constant parameters to a new "experiment_.." file
    save_constant_parameters(experiment_name, changing_param_name)


    # run experiment for every parameter value
    for value in changing_param_value:
        process = psutil.Process(os.getpid())
        print("-----MEMORY before starting experiment ------", int(process.memory_info().rss/(8*10**(3))), "KB")

        # update the parameter value
        set_params(use_word_emb = value)

        # update the model_id for this new model
        now = datetime.datetime.now()
        new_model_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute) + "." + str(now.second)
        set_params(model_id = new_model_id)

        # evaluate the new model and save the results in the experiment file
        oneExperiment = Process(target=run_experiment, args=(experiment_name, new_model_id, changing_param_name, value,))
        oneExperiment.start()
        oneExperiment.join()

        if value == changing_param_value[0]:
            set_params(preproc_data_id=new_model_id)

def reg_experiment():
    '''
    function that runs an example experiment
    writes the used parameters and the results to the file "experiments/experiment_..."
    '''
    print("REG_EXPERIMENT")

    # set the name of the experiment
    now = datetime.datetime.now()
    experiment_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute)
    experiment_name = 'regularization_' + str(experiment_id)

    # define if you want to use preprocessed data from file
    use_prep_data = False
    if use_prep_data:
        set_params(preproc_data_id='16_5_10.16.47')
    else:
        set_params(use_preproc_data=False)

    # define the changing parameter and its value
    changing_param_name = 'dropout'
    changing_param_value = [0.0,0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]#, {0:4, 1:100}, {0:3, 1:100}, {0:2, 1:100}, {0:1, 1:100}] #[{0:1, 1:1}, {0:15, 1:85}]#

    # set constant parameters
    set_params(use_word_emb=1)
    set_params(epochs=1)
    #
    #
    #...

    # save constant parameters to a new "experiment_.." file
    save_constant_parameters(experiment_name, changing_param_name)


    # run experiment for every parameter value
    for value in changing_param_value:
        process = psutil.Process(os.getpid())
        print("-----MEMORY before starting experiment ------", int(process.memory_info().rss/(8*10**(3))), "KB")

        # update the parameter value
        set_params(dropout = value)

        # update the model_id for this new model
        now = datetime.datetime.now()
        new_model_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute) + "." + str(now.second)
        set_params(model_id = new_model_id)

        # evaluate the new model and save the results in the experiment file
        oneExperiment = Process(target=run_experiment, args=(experiment_name, new_model_id, changing_param_name, value,))
        oneExperiment.start()
        oneExperiment.join()

        if value == changing_param_value[0]:
            set_params(preproc_data_id=new_model_id)

# specify which experiment you want to run
if __name__ == '__main__':
    #one_experiment()
    #class_weights_binary()
    #shutil.rmtree("proc_data/")
    #class_weights_embedding()
    #reg_experiment()
    for i in range(5):
        timesteps_experiment()
