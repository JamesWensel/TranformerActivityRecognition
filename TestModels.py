#! /usr/bin/python3

import os
import time 
import json
import psutil
import numpy as np

# Personal File Imports 
import run_model
import output
import build_models

MODELS = {'LSTM': build_models.LSTM_Model, 'Transformer': build_models.Transformer_Model}
GET_MEM = psutil.Process(os.getpid()).memory_info

with open('tests.json') as json_file: 
    data = json.load(json_file) 

config = json.loads(data) 

def run_test(features, labels, model, model_name, OutputData, model_test, val):
    """
    Runs a single test, internally tracks and outputs time and memory usage
    Deletes model, predioctions, and saved data upon completion of tests
    
    Arguments
    ---------
        model: tensorflow.keras.Model
            model to run test on
        features: numpy.ndarray
            features to train model on
        labels: numpy.ndarray
            aray of labels associated with features at the same indices
        OutputData: Output Object
            used to track timing and memory usage
        model_test: string
            name of the current test being run
    
    Returns
    -------
        time: string
            total test runtime formatted as a string
        memory: string 
            peak memory usage formatted as a string
        eval_histroy: list
            contains results of evaluations performed after model training
    """
    
    plot_model_str = f"Model_Tests/{model_name}/{model_test}/Models_Graphs/{val}_{model_test}.png"
    (model, predictions, OutputData), history, eval_history = run_model.FitModel(features, labels, model, plot_model_str, OutputData, batch_size=config['Setup']['Batch'], epochs=config['Setup']['Epochs'])
    end_time, mem = OutputData.add_data(time=time.time(), mem=GET_MEM().rss, final=True, filename=f"{val}_{model_test}.csv", directory=f'Model_Tests/{model_name}/{model_test}')
    
    if config['Setup']['Save'] == True: 
        model.save(f'Models/{model_test}')
        np.save(f'Predictions/{model_test}', predictions)
        
    del predictions

    return end_time, mem, history, eval_history


TotalData = output.Output(config['Setup']['Header'], log_level=3)

data = ['Start', 'None', 'Data', time.time(), GET_MEM().rss, 0, 0, config['Setup']['Batch'], config['Setup']['Epochs']]
TotalData.add_data(time.time(), GET_MEM().rss, dict(zip(config['Setup']['Header'],data)))

TotalData.add_data(time.time(), GET_MEM().rss, identifier='Load')

features = np.load(os.path.join(config['Setup']['Dir'], config['Setup']['Features']))
labels = np.load(os.path.join(config['Setup']['Dir'], config['Setup']['Labels']))
(_,sequence_length, image_size) = features.shape

print(f"Features: {features.shape}")

TotalData.add_data(time.time(), GET_MEM().rss, identifier='Load')

best_loss = 100
best_val = -1

for model_type in config['Tests']: 
    for test in config['Tests'][model_type]: 
        for val in config['Tests'][model_type][test]: 
            os.system('clear') 
            print(f"Testing {model_type} {test}.....")
            
            TotalData.add_data(time.time(), GET_MEM().rss, identifier=f'{model_type} {test} = {val}')
            
            defaults = config['Defaults'][model_type]
            store_val = defaults[test]
            defaults[test] = val
            args = list(defaults.values())
            args = [sequence_length, image_size, config['Setup']['Categories']] + args
            
            OutputData = output.Output(log_level=3)
            OutputData.add_data(time.time(), GET_MEM().rss)
            
            OutputData.add_data(time.time(), GET_MEM().rss, identifier="Build Model")
            model = MODELS[model_type](*args) 
            OutputData.add_data(time.time(), GET_MEM().rss, identifier="Build Model")
            
            end_time, mem, history, evaluate = run_test(features, labels, model, model_type, OutputData, test, val)
            defaults[test] = store_val
           
            data = [test, model_type, val, end_time, mem, evaluate[0], evaluate[1], config['Setup']['Batch'], config['Setup']['Epochs']]
            TotalData.add_data(time.time(), GET_MEM().rss, dict(zip(config['Setup']['Header'], data)), identifier=f'{model_type} {test} = {val}')
            
            output.outputTestResults(history, model_type, test, val)
            
            if evaluate[0] < best_loss: 
                best_loss = evaluate[0] 
                best_val = val
            
            del OutputData
            del model
        
        config['Defaults'][model_type][test] = best_val if best_val != -1 else config['Defaults'][model_type][test]
        
        best_loss = 100
        best_val = -1

data = ['Finish', 'None', 'Data', time.time(), GET_MEM().rss, 0, 0, config['Setup']['Batch'], config['Setup']['Epochs']]
TotalData.add_data(time.time(), GET_MEM().rss, dict(zip(config['Setup']['Header'], data)), final=True, filename="Tests.csv", directory='Model_Tests')


best_vals = {}

Feat = config['Setup']['Features'].split('_')[0]
filename = os.path.join(config['Setup']['Dir'], f'{Feat}_BestValues.json')
json_string = json.dumps(config['Defaults'])

with open(filename, 'w') as outfile: 
    json.dump(json_string, outfile) 