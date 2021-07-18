import os
import joblib 
import warnings
import itertools
import numpy as np
import pandas as pd 
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle, class_weight
from pandas.core.common import SettingWithCopyWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from variables import *

np.random.seed(seed)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

def collect_dataframes(data_dir, csv_labels, final_csv_path, csv_type):
    first_df = True
    for csv_dir in os.listdir(data_dir):
        for label, csv_files in csv_labels.items():
            for csv_file, tx_idx in csv_files:
                csv_file_path = os.path.join(data_dir, csv_dir, csv_type, csv_file)
                if os.path.exists(csv_file_path):
                    df = pd.read_csv(csv_file_path, index_col=False)
                    del df['real_timestamp']
                    df['TX'] = int(tx_idx)
                    csi_values = df.copy()
                    csi_values['label'] = int(label)
                    csi_values = csi_values[['label'] + df.columns.values.tolist()]

                    if first_df:
                        final_df = csi_values
                        first_df = False
                    else:
                        final_df = pd.concat([final_df, csi_values], ignore_index=True)

        else:
            assert not first_df, "None of the CSV files are existing"

    return final_df

def process_dnn_data(data_dir):
    df_amplitude = collect_dataframes(data_dir, csv_label_names, amplitude_csv_path , 'amp_emva')
    df_phase = collect_dataframes(data_dir, csv_label_names, phase_csv_path, 'phase_san_emva')

    label_amplitude = df_amplitude[df_amplitude.columns.values[0]].values
    label_phase = df_phase[df_phase.columns.values[0]].values

    assert np.array_equal(label_amplitude, label_phase), "Phase and Amplitude not matched with each other"
    del df_phase['TX']

    amplitude_values = df_amplitude[df_amplitude.columns.values[1:]].values
    phase_values = df_phase[df_phase.columns.values[1:]].values

    return amplitude_values, phase_values, label_amplitude

def impute_dnn_data(phase_values, amplitude_values, labels):
    csi_values = np.concatenate([amplitude_values, phase_values], axis=1)

    labels = labels[~np.isnan(csi_values).any(axis=1)]
    csi_values = csi_values[~np.isnan(csi_values).any(axis=1)]

    labels, csi_values = shuffle(labels, csi_values)
    return csi_values, labels

def load_dnn_data():
    amplitude_values, phase_values, labels = process_dnn_data(offline_data_dir)
    online_amplitude_values, online_phase_values, online_labels = process_dnn_data(online_data_dir)

    amplitude_scaler = StandardScaler()
    amplitude_scaler.fit(amplitude_values)
    amplitude_values = amplitude_scaler.transform(amplitude_values)   
    online_amplitude_values = amplitude_scaler.transform(online_amplitude_values)   

    phase_scaler = StandardScaler()
    phase_scaler.fit(phase_values)
    phase_values = phase_scaler.transform(phase_values)
    online_phase_values = phase_scaler.transform(online_phase_values) 

    X, Y = impute_dnn_data(amplitude_values, phase_values, labels)
    # Xinf, Yinf = impute_dnn_data(online_amplitude_values, online_phase_values, online_labels)

    # rand_sample_indices = np.random.choice(len(Yinf), int(0.4 * len(Yinf)), replace=False)
    # Xadd = Xinf[rand_sample_indices]
    # Yadd = Yinf[rand_sample_indices]

    # X = np.concatenate([X, Xadd])
    # Y = np.concatenate([Y, Yadd])
    
    X, Y = shuffle(X, Y)

    X, Xinf, Y, Yinf = train_test_split(
                                        X, Y, 
                                        test_size=0.2, 
                                        random_state=seed
                                            )
    Yinf = Yinf.reshape(-1,1)
    inf_data = np.concatenate((Xinf,Yinf), axis=1)
    inf_data_df = pd.DataFrame(data=inf_data, columns=list(range(Xinf.shape[1]))+['label'])
    inf_data_df = inf_data_df.sort_values(by=['label'])
    inf_data_df.to_csv(inf_data_path, index=False)
    # Xtot = X
    # Ytot = Y
    # rand_sample_indices = np.random.choice(len(Ytot), int(split * len(Ytot)), replace=False)

    # X = Xtot[rand_sample_indices]
    # Y = Ytot[rand_sample_indices]

    # mask = np.ones(len(Ytot), dtype=bool)
    # mask[rand_sample_indices] = 0
    # test_indices = np.nonzero(mask)

    # Xinf = Xtot[test_indices]
    # Yinf = Ytot[test_indices]

    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    Yinf = encoder.transform(Yinf)

    class_weights = class_weight.compute_class_weight('balanced',
                                                    np.unique(Y),
                                                    Y)
    class_weights = {i : class_weights[i] for i in range(len(set(Y)))}

    return X, Xinf, Y, Yinf, class_weights

def process_conv1d_data(data_dir):
    df_amplitude = collect_dataframes(data_dir, csv_label_names, amplitude_csv_path , 'amp_emva')
    df_phase = collect_dataframes(data_dir, csv_label_names, phase_csv_path, 'phase_san_emva')

    label_amplitude = df_amplitude[df_amplitude.columns.values[0]].values
    label_phase = df_phase[df_phase.columns.values[0]].values

    assert np.array_equal(label_amplitude, label_phase), "Phase and Amplitude not matched with each other"

    amplitude_values = df_amplitude[df_amplitude.columns.values[1:]].values
    phase_values = df_phase[df_phase.columns.values[1:]].values

    return amplitude_values, phase_values, label_amplitude

def impute_conv1d_data(phase_values, amplitude_values, labels):
    amplitude_values = np.expand_dims(amplitude_values, axis=2)
    phase_values = np.expand_dims(phase_values, axis=2)
    csi_values = np.concatenate([phase_values,amplitude_values], axis=2)

    labels = labels[~np.isnan(csi_values).any(axis=(1,2))]
    csi_values = csi_values[~np.isnan(csi_values).any(axis=(1,2))]

    labels, csi_values = shuffle(labels, csi_values)
    return csi_values, labels

def load_conv1d_data():
    amplitude_values, phase_values, labels = process_conv1d_data(offline_data_dir)
    online_amplitude_values, online_phase_values, online_labels = process_conv1d_data(online_data_dir)

    amplitude_scaler = MinMaxScaler()
    amplitude_scaler.fit(amplitude_values)
    amplitude_values = amplitude_scaler.transform(amplitude_values)   
    online_amplitude_values = amplitude_scaler.transform(online_amplitude_values)   

    phase_scaler = StandardScaler()
    phase_scaler.fit(phase_values)
    phase_values = phase_scaler.transform(phase_values)
    online_phase_values = phase_scaler.transform(online_phase_values) 

    X, Y = impute_conv1d_data(amplitude_values, phase_values, labels)
    # Xinf, Yinf = impute_conv1d_data(online_amplitude_values, online_phase_values, online_labels)

    # rand_sample_indices = np.random.choice(len(Yinf), int(0.3 * len(Yinf)), replace=False)
    # Xadd = Xinf[rand_sample_indices]
    # Yadd = Yinf[rand_sample_indices]

    # X = np.concatenate([X, Xadd])
    # Y = np.concatenate([Y, Yadd])
    
    # X, Y = shuffle(X, Y)

    # Xtot = X
    # Ytot = Y
    # rand_sample_indices = np.random.choice(len(Ytot), int(split * len(Ytot)), replace=False)

    # X = Xtot[rand_sample_indices]
    # Y = Ytot[rand_sample_indices]

    # mask = np.ones(len(Ytot), dtype=bool)
    # mask[rand_sample_indices] = 0
    # test_indices = np.nonzero(mask)

    # Xinf = Xtot[test_indices]
    # Yinf = Ytot[test_indices]


    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)
    Yinf = encoder.transform(Yinf)

    class_weights = class_weight.compute_class_weight('balanced',
                                                    np.unique(Y),
                                                    Y)
    class_weights = {i : class_weights[i] for i in range(len(set(Y)))}

    return X, Xinf, Y, Yinf, class_weights

def predictions(X, Y, inference):
    n_correct = 0
    n_total = len(Y)
    for x, y in zip(X, Y):
        y_pred = inference.Inference(x)
        p = y_pred[0].argmax()
        if y == p:
            n_correct += 1
    return n_correct/n_total