from pathlib import PurePath
import os
import glob
import numpy as np
from pandas import DataFrame, Series
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model
from keras.callbacks import Callback
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras import backend as K
from arff2pandas import a2p


class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """

    def __init__(self, monitor='acc', baseline=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True


class CreateDf():
    def __init__(self):
        self.df = DataFrame(dtype=object)

    def create_df_from_folders(self, folders, anotations):
        if len(anotations) == len(folders):
            last_index_name = ''
            for anot in anotations:
                last_index_name = last_index_name + str(anot) + ','
            last_index_name = 'label@{{{0}}}'.format(last_index_name[:-1])
            print(last_index_name)
            i = 0
            for folder in folders:
                j = 0
                for item in glob.iglob(str(folder) + '/*.npy'):
                    array_data = np.load(item)
                    indexes = ['f' + str(x) + '@NUMERIC' for x in range(len(array_data))]
                    # print(indexes)
                    ser = Series(data=array_data, index=indexes)
                    ser = ser.append(Series(data=anotations[i], index=[last_index_name]))
                    # print(ser)
                    self.df = self.df.append(ser, ignore_index=True)
                    j += 1
                    if j >= 100:
                        break
                i += 1
            # print(indexes.append(last_index_name))
            self.df = self.df.reindex(labels=indexes.append(last_index_name), axis='columns')
            self.df = self.df.astype({last_index_name: 'str'})
            print(type(self.df.loc[199][-1]))
            print(self.df)
            print('Df creation done')
        else:
            print('Anotations list len should be the same as folders list len')

    def save_to_arff2(self, filename):
        with open(str(filename), 'w') as f:
            a2p.dump(self.df, f)

    def load_from_arff2(self, filename):
        with open(str(filename), 'r') as f:
            self.df = a2p.load(f)

    def get_df(self):
        return self.df

    def loaf_df(self, df):
        self.df = df


class SynthANN():
    def __init__(self, square_db_folder, triangle_db_folder):
        series_list = []
        anotation_list = []
        for item in glob.iglob(str(square_db_folder) + '/*.npy'):
            series_list.append(np.load(item))
            anotation_list.append(0)
        for item in glob.iglob(str(triangle_db_folder) + '/*.npy'):
            series_list.append(np.load(item))
            anotation_list.append(1)
        self.df = DataFrame({'x': series_list, 'y': anotation_list})
        self.train_df = None
        self.test_df = None
        del series_list, anotation_list
        self.ndf_train_x = None
        self.ndf_train_y = None
        self.sdf_train_x = None
        self.sdf_train_y = None
        self.ndf_test_x = None
        self.ndf_test_y = None
        self.sdf_test_x = None
        self.sdf_test_y = None
        self.xscale_object = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
        self.xscale_object = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

    def split_data(self, size, shuffle_bool):
        self.train_df, self.test_df = train_test_split(self.df, test_size=size, shuffle=shuffle_bool)

    def normalize_df(self):
        self.ndf_train_x = self.train_df.copy(deep=True)
        self.ndf_train_x[self.ndf_train_x.columns] = (
            self.xscale_object.fit_transform(self.ndf_train_x[self.ndf_train_x.columns])
        )
        self.test_df_xs = self.test_df.copy(deep=True)
        self.test_df_xs[self.test_df_xs.columns] = (
            self.xscale_object.transform(self.test_df_xs[self.test_df_xs.columns])
        )
        print('Data normalized')

    def train_model(train_df, epochs_to_train):
        # overfitCallback = EarlyStopping(monitor='acc', min_delta=0.00001, mode='min', baseline=0.99, patience=3)
        overfitCallback = TerminateOnBaseline(monitor='acc', baseline=0.99)
        train_samle_length = len(train_df.iloc[0][0])
        # print(train_df.iloc[0][0])
        model_fft_input = Input(shape=(train_samle_length,))
        model_fft_dense_1 = Dense(72, activation='relu')(model_fft_input)
        model_fft_dense_2 = Dense(36, activation='relu')(model_fft_dense_1)
        predict_out = Dense(1, activation='sigmoid')(model_fft_dense_2)
        model_fft = Model(inputs=model_fft_input, outputs=predict_out)
        model_fft.compile(optimizer='rmsprop',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        history = model_fft.fit(np.asarray(train_df['x'].tolist()), np.asarray(train_df['y'].tolist()), epochs=epochs_to_train, callbacks=[overfitCallback])
        # print(history.history)
        return history, model_fft


    def train_1model(x_train, y_train, epochs):
        # FFT conv layers
        train_samle_length = len(x_train[0])
        print(train_samle_length)
        print(x_train[0].shape)
        if K.image_data_format() == 'channels_first':
            input_shape = (3, 393, 393)
        else:
            input_shape = (393, 393, 3)

        model_input = Input(shape=input_shape)
        # model_input = Input(shape=(x_train[0].shape, 1))
        model_conv_11 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(model_input)
        model_pool_11 = MaxPooling2D(pool_size=(2, 2))(model_conv_11)
        model_conv_12 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(model_pool_11)
        model_pool_12 = MaxPooling2D(pool_size=(2, 2))(model_conv_12)
        model_flatten_12 = Flatten()(model_pool_12)
        # Output sofmax layer
        predictions = Dense((1), activation='sigmoid')(model_flatten_12)
        model_conv = Model(inputs=[model_input], output=predictions)
        model_conv.compile(optimizer='rmsprop',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        model_conv.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=1)
        return model_conv


def train_2model(x_train, y_train, epochs):
    # FFT conv layers
    train_samle_length = len(x_train[0])
    print(train_samle_length)
    print(x_train[0].shape)
    if K.image_data_format() == 'channels_first':
        input_shape = (3, 393, 393)
    else:
        input_shape = (393, 393, 3)

    model_input = Input(shape=input_shape)
    # model_input = Input(shape=(x_train[0].shape, 1))
    model_conv_11 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(model_input)
    model_pool_11 = MaxPooling2D(pool_size=(2, 2))(model_conv_11)
    model_conv_12 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(model_pool_11)
    model_pool_12 = MaxPooling2D(pool_size=(2, 2))(model_conv_12)
    model_conv_13 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(model_pool_12)
    model_pool_13 = MaxPooling2D(pool_size=(2, 2))(model_conv_13)
    model_flatten_14 = Flatten()(model_pool_13)
    model_dense_15 = Dense((64), activation='relu')(model_flatten_14)
    model_dropout_16 = Dropout(0.2)(model_dense_15)
    # Output sofmax layer
    predictions = Dense((1), activation='sigmoid')(model_dropout_16)
    model_conv = Model(inputs=[model_input], output=predictions)
    model_conv.compile(optimizer='rmsprop',
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
    model_conv.fit(x_train, y_train, epochs=epochs, batch_size=20, verbose=1)
    return model_conv


def test_model(model, test_df):
    x = np.asarray(test_df['x'].tolist())
    y = np.asarray(test_df['y'].tolist())
    scores = model.evaluate(x, y, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

def save_model(model, test_df):
    x = np.asarray(test_df['x'].tolist())
    y = np.asarray(test_df['y'].tolist())
    scores = model.evaluate(x, y, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

def load_model(model, test_df):
    x = np.asarray(test_df['x'].tolist())
    y = np.asarray(test_df['y'].tolist())
    scores = model.evaluate(x, y, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


if __name__ == "__main__":
    square_series_folder = PurePath(os.getcwd(), 'square_db')
    tringle_series_folder = PurePath(os.getcwd(), 'triangle_db')
    df_arff2_filename = PurePath(os.getcwd(), 'squre_triangle').with_suffix('.arff')
    df = CreateDf()
    df.create_df_from_folders([square_series_folder, tringle_series_folder], ['sq', 't'])
    df.save_to_arff2(df_arff2_filename)
    df.load_from_arff2(df_arff2_filename)
    print(df.get_df())
    # SynthANN_o = SynthANN(square_series_folder, tringle_series_folder)
    # SynthANN_o.normalize_df()
    # train, test = train_test_split(df, test_size=0.2, shuffle=True)
    # history, model = train_model(train, epochs_to_train=100)
    # test_model(model, test)