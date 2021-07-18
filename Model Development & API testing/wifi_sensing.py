import os
import sys
import pathlib
import logging
import warnings
import numpy as np
import tensorflow as tf
from collections import Counter, OrderedDict
from sklearn.exceptions import DataConversionWarning

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from tensorflow.keras.metrics import Precision
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical as OneHotEncoder
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Conv1D, GRU, MaxPooling1D, Flatten

logging.getLogger('tensorflow').disabled = True

from variables import*
from util import*

np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

try:
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

except:
  print("GPU not Found !!!")

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > callback_acc):
            print("\nReached {}% train accuracy.So stop training!".format(callback_acc))
            self.model.stop_training = True

class MotionSentimentDetection(object):
    def __init__(self, model_type):
        if model_type.lower() == 'dnn':
            self.model_weights = dnn_model_weights
            self.classifier = self.dnn_classifier
            self.model_converter = dnn_model_converter
            if not os.path.exists(self.model_converter):
                X, Xinf, Y, Yinf, class_weights = load_dnn_data()
        
        elif model_type.lower() == 'conv1d':
            self.model_weights = conv1D_model_weights
            self.classifier = self.conv1d_classifier
            self.model_converter = conv1d_model_converter
            if not os.path.exists(self.model_converter):
                X, Xinf, Y, Yinf, class_weights = load_conv1d_data()
        
        else:
            sys.exit("Enter Valid Model Architecture !!!")
            
        if not os.path.exists(self.model_converter):
            self.X = X
            self.Y = Y
            self.Y = OneHotEncoder(
                                self.Y, 
                                num_classes=len(set(self.Y))
                                    )
            self.Xinf = Xinf
            self.Yinf = Yinf
            self.Yinf = OneHotEncoder(
                                self.Yinf, 
                                num_classes=len(set(self.Yinf))
                                    )
            self.model_type = model_type
            self.class_weights=class_weights

    def dnn_classifier(self):
        n_features = self.X.shape[1]
        num_classes = self.Y.shape[1]

        inputs = Input(shape=(n_features,))
        
        x = Dense(dense1, activation=activation)(inputs)
        x = Dense(dense1)(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Dropout(keep_prob)(x)

        x = Dense(dense2, activation=activation)(x)
        x = Dense(dense2)(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Dropout(keep_prob)(x)

        x = Dense(dense3, activation=activation)(x)
        x = Dense(dense3)(x)
        x = BatchNormalization()(x)
        x = relu(x)
        x = Dropout(keep_prob)(x)

        x = Dense(dense4, activation=activation)(x)
        x = Dropout(keep_prob)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        self.model = Model(inputs, outputs)

        self.model.summary()

    def conv1d_classifier(self):
        n_features = self.X.shape[2]
        n_timesteps = self.X.shape[1]
        num_classes = self.Y.shape[1]

        inputs = Input(shape=(n_timesteps, n_features))
        x = Conv1D(filter1, kernal_size)(inputs)
        x = MaxPooling1D(pool_size=pool_size)(x)
        x = BatchNormalization()(x)
        x = relu(x)

        x = Conv1D(filter1, kernal_size)(x)
        x = MaxPooling1D(pool_size=pool_size)(x)
        x = BatchNormalization()(x)
        x = relu(x)

        x = Conv1D(filter2, kernal_size)(x)
        x = MaxPooling1D(pool_size=pool_size)(x)
        x = BatchNormalization()(x)
        x = relu(x)

        x = Conv1D(filter3, kernal_size,)(x)
        x = MaxPooling1D(pool_size=pool_size)(x)
        x = BatchNormalization()(x)
        x = relu(x)

        x = Flatten()(x)
        x = Dense(densef, activation=activation)(x)
        x = Dropout(keep_prob_c)(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        self.model = Model(inputs, outputs)
        self.model.summary()

    def train(self):
        callbacks = myCallback()
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate),
            metrics=['accuracy', Precision()],
        )

        num_epoches = 40 if self.model_type.lower() == 'dnn' else 15
        self.history = self.model.fit(
                            self.X,
                            self.Y,
                            batch_size=batch_size,
                            epochs=num_epoches,
                            validation_split=val_split,
                            class_weight=self.class_weights,
                            callbacks= [callbacks]
                            )

    def load_model(self):
        loaded_model = load_model(self.model_weights)
        loaded_model.compile(
                        loss='categorical_crossentropy',
                        optimizer=Adam(learning_rate),
                        metrics=['accuracy', Precision()],
                        )
        self.model = loaded_model

    def save_model(self):
        self.model.save(self.model_weights)

    def evaluation(self):
        self.model.evaluate(self.Xinf, self.Yinf)

    def plot_confusion_matrix(self, X, Y, cmap=None, normalize=True):
        
        P = np.argmax(self.model.predict(X), axis=-1)
        Y = np.argmax(Y, axis=-1)
        cm = confusion_matrix(Y, P)

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(12, 12))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title.format(self.model_type))
        plt.colorbar()

        class_names = list(set(Y))

        if class_names is not None:
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=0)
            plt.yticks(tick_marks, class_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")


        plt.tight_layout()
        plt.ylabel('True labels')
        plt.xlabel('Predicted labels')
        plt.savefig(confusion_matrix_path)

    def error_analysis(self, X, Y):
        P = np.argmax(self.model.predict(X), axis=-1)
        Y = np.argmax(Y, axis=-1)
        error = P - Y 
        abs_error = np.abs(error)

        error_dict = dict(Counter(abs_error))
        error_percentage = {error : count * 100 / len(abs_error) for error, count in error_dict.items()}
        error_percentage = OrderedDict(sorted(error_percentage.items(), 
                                       key=lambda kv: kv[1], reverse=True))
        
        error_percentage = dict(error_percentage)
        print("\nError Analysis of Results")
        print('----------------------------------------------------\n')
        for e in range(len(error_percentage.keys())):
            percentage = round(error_percentage[e], 3)
            if e == 0:
                print('Correct Predictions : {}%\n'.format(percentage))

            else: 
                if len(str(e)) == 1:
                    print('For Error = {}       : {}%\n'.format(e, percentage))
                else:
                    print('For Error = {}      : {}%\n'.format(e, percentage))
        print('----------------------------------------------------\n')

    def comparison_plot(self, X, Y):
        P = np.argmax(self.model.predict(X), axis=-1)
        Y = np.argmax(Y, axis=-1)
        plt.scatter(P, Y)
        plt.plot(np.arange(min(Y), max(Y)+1), np.arange(min(Y), max(Y)+1), c='r')
        plt.ylabel('True labels')
        plt.xlabel('Predicted labels')
        plt.savefig('comparison_plot.png')
        plt.show()

    def predictions(self, X):
        P = self.model.predict(X)
        Y = P.argmax(axis=-1)
        return Y

    def TFconverter(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.target_spec.supported_ops = [
                                tf.lite.OpsSet.TFLITE_BUILTINS,   # Handling unsupported tensorflow Ops 
                                tf.lite.OpsSet.SELECT_TF_OPS 
                                ]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]      # Set optimization default and it configure between latency, accuracy and model size
        tflite_model = converter.convert()

        model_converter_file = pathlib.Path(self.model_converter) 
        model_converter_file.write_bytes(tflite_model) # save the tflite model in byte format

    def TFinterpreter(self):
        self.interpreter = tf.lite.Interpreter(model_path=self.model_converter) # Load tflite model
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details() # Get input details of the model
        self.output_details = self.interpreter.get_output_details() # Get output details of the model

    def Inference(self, features):
        features = features.astype(np.float32)
        input_shape = self.input_details[0]['shape']
        assert np.array_equal(input_shape, features.shape), "Input tensor hasn't correct dimension"

        self.interpreter.set_tensor(self.input_details[0]['index'], features)

        self.interpreter.invoke() # set the inference

        output_data = self.interpreter.get_tensor(self.output_details[0]['index']) # Get predictions
        return output_data

    def run(self):
        if not os.path.exists(self.model_converter):
            if os.path.exists(self.model_weights):
                print("Loading the model !!!")
                self.load_model()
            else:
                print("Training the model !!!")
                self.classifier()
                self.train()
                self.save_model()
            self.evaluation()
            self.TFconverter()
        self.TFinterpreter()

        
        # self.plot_confusion_matrix(self.Xinf, self.Yinf)
        # self.error_analysis(self.Xinf, self.Yinf)