import numpy as np 

online_data_dir = 'data and results/experiment data/online data'
offline_data_dir = 'data and results/experiment data/offline data'

csv_label_names = {
                    12 : [
                        ('PP-12P-RX1.csv', 0),
                        ('PP-12P-RX2.csv', 1)
                        ],

                    11 : [ 
                        ('PP-11P-RX1.csv', 0),
                        ('PP-11P-RX2.csv', 1)
                        ],

                    10 : [
                        ('PP-10P-RX1.csv', 0),
                        ('PP-10P-RX2.csv', 1)
                        ],

                    9 : [
                        ('PP-9P-RX1.csv', 0), 
                        ('PP-9P-RX2.csv', 1)
                        ],

                    8 : [
                        ('PP-8P-RX1.csv', 0),  
                        ('PP-8P-RX2.csv', 1)
                        ], 

                    7 : [
                        ('PP-7P-RX1.csv', 0),
                        ('PP-7P-RX2.csv', 1)
                        ],

                    6 : [ 
                        ('PP-6P-RX1.csv', 0),
                        ('PP-6P-RX2.csv', 1)
                        ],

                    5 : [
                        ('PP-5P-RX1.csv', 0),
                        ('PP-5P-RX2.csv', 1)
                        ],

                    4 : [
                        ('PP-4P-RX1.csv', 0), 
                        ('PP-4P-RX2.csv', 1)
                        ],

                    3 : [
                        ('PP-3P-RX1.csv', 0),  
                        ('PP-3P-RX2.csv', 1)
                        ],

                    2 : [
                        ('PP-2P-RX1.csv', 0),  
                        ('PP-2P-RX2.csv', 1)
                        ],

                    1 : [
                        ('PP-1P-RX1.csv', 0),
                        ('PP-1P-RX2.csv', 1)
                        ],

                    0 : [
                        ('PP-0P-RX1.csv', 0),
                        ('PP-0P-RX2.csv', 1)
                        ]
                    }

amplitude_csv_path = 'data and results/csv files/motion_amplitude.csv'
phase_csv_path = 'data and results/csv files/motion_phase.csv'

seed = 42
split = 0.8
np.random.seed(seed)
val_split = 0.1
test_split = 0.1
num_epoches = 10
batch_size = 128
learning_rate = 0.0001
n_components = 30
callback_acc = 0.96
title='Confusion matrix for {} model'
confusion_matrix_path = 'data and results/visualizations/Confusion matrix.png' 
cum_error_percentage_path = 'cm/CDF error for {} & {}person.png' 

# DNN parameters
dense1 = 1024
dense2 = 512
dense3 = 256
dense4 = 64
keep_prob = 0.6
activation = 'relu'
inf_data_path = 'data and results/weights/dnn weights/inf_data.csv'
dnn_model_weights = 'data and results/weights/dnn weights/motion_sensing_dnn.h5'
dnn_converter = "data and results/weights/dnn weights/dnn.tflite"
dnn_label_encoder = "data and results/weights/dnn weights/dnn_label_encoder.pkl"
dnn_phase_standard_scaler = "data and results/weights/dnn weights/dnn_phase_standard_scaler.pkl"
dnn_model_converter = "data and results/weights/dnn weights/dnn_model_converter.tflite"
dnn_amplitude_standard_scaler = "data and results/weights/dnn weights/dnn_amplitude_standard_scaler.pkl"

# Conv1D parameters
filter1 = 512
filter2 = 256
filter3 = 128
filter4 = 64
densef = 64
keep_prob_c = 0.5
kernal_size = 3
pool_size = 2
conv1D_model_weights = 'data and results/weights/conv1d weights/motion_sensing_conv1d.h5'
conv1D_converter = "data and results/weights/conv1d weights/conv1D.tflite"
conv1d_label_encoder = "data and results/weights/conv1d weights/conv1d_label_encoder.pkl"
conv1d_phase_standard_scaler = "data and results/weights/conv1d weights/conv1d_phase_standard_scaler.pkl"
conv1d_model_converter = "data and results/weights/conv1d weights/conv1d_model_converter.tflite"
conv1d_amplitude_standard_scaler = "data and results/weights/conv1d weights/conv1d_amplitude_standard_scaler.pkl"

#deployment
host = '10.10.10.100'
port = 8050