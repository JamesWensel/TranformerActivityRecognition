#! /usr/bin/python3

import json

## Add arg parsing to change setup from command line (no need to enter this file to change tests run

config = {}
config['Setup'] = {'Header':  ['Test', 'Model Name', 'Variable', 'Time', 'Memory Usage', 'Loss', 'Accuracy', 'Batch', 'Epochs'],
                   'Dir': 'Total Dataset', 
                   'Features': 'ViT_features_extracted.npy',
                   'Labels': 'labels.npy',
                   'Categories': 101,
                   'Epochs': 50,
                   'Batch': 4,
                   'Save': False}

config['Tests'] = {}
config['Tests']['LSTM'] = {'Layers': [1,2,4,6], 
                           'Units': [32, 64, 128, 256, 512]}
config['Tests']['Transformer'] = {'Layers': [1,2,4,6], 
                                  'Input': [128, 256, 512, 1024],
                                  'Internal': [256, 512, 1024],
                                  'Attention': [1,2,4,6,8,16]}

config['Defaults'] = {}
config['Defaults']['LSTM'] = {'Layers': 1, 'Units': 128}
config['Defaults']['Transformer'] = {'Layers': 6, 'Input': 256, 'Internal': 256, 'Attention': 2}

json_string = json.dumps(config)

with open('tests.json', 'w') as outfile: 
    json.dump(json_string, outfile)