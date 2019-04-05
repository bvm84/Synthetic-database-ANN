from pathlib import PurePath
import os
import glob
import numpy as np
from pandas import DataFrame, Series
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input
from keras.models import Model
from keras.callbacks import Callback
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
        self.df = None

    def create_df_from_folders(self, folders, anotations):
        if len(anotations) == len(folders):
            i = 0
            arrays_list = []
            anotation_list = []
            for folder in folders:
                for item in glob.iglob(str(folder) + '/*.npy'):
                    arrays_list.append(np.load(item))
                    anotation_list.append(anotations[i])
                i += 1
            columns_names = ['f' + str(x) + '@NUMERIC' for x in range(len(arrays_list[-1]))]
            self.df = DataFrame(data=arrays_list, columns=columns_names)
            last_column_name = ''
            for anot in anotations:
                last_column_name = last_column_name + str(anot) + ','
            last_column_name = 'label@{{{0}}}'.format(last_column_name[:-1])
            # print(last_column_name)
            self.df.insert(loc=len(self.df.columns), column=last_column_name, value=anotation_list)
            # print(self.df)
            # self.df = self.df.reindex(labels=indexes.append(last_index_name), axis='columns')
            # self.df = self.df.astype({last_index_name: 'str'})
            # print(type(self.df.loc[199][-1]))
            # print(self.df)
            # print('Df creation done')
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
    def __init__(self, df):
        self.df = df
        self.train_df = None
        self.test_df = None
        self.ndf_train_x = None
        self.ndf_train_y = None
        self.ndf_test_x = None
        self.ndf_test_y = None
        self.sdf_train_x = None
        self.sdf_train_y = None
        self.sdf_test_x = None
        self.sdf_test_y = None
        self.xnscale_object = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
        self.ynscale_object = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
        self.xsscale_object = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)

    def truncate_data(self, rows_to_store):
        indexes_to_drop = [x for x in range(rows_to_store, len(self.df) - rows_to_store)]
        self.df.drop(labels=indexes_to_drop, axis='index', inplace=True)
        # print(self.df)

    def split_data(self, size=0.2, shuffle_bool=False):
        self.train_df, self.test_df = train_test_split(self.df, test_size=size, shuffle=shuffle_bool)

    def normalize_df(self):
        self.ndf_train_x = self.train_df.copy(deep=True).drop(self.train_df.columns[-1], axis='columns')
        self.ndf_train_x[self.ndf_train_x.columns] = (
            self.xnscale_object.fit_transform(self.ndf_train_x[self.ndf_train_x.columns])
        )
        self.ndf_test_x = self.test_df.copy(deep=True).drop(self.test_df.columns[-1], axis='columns')
        self.ndf_test_x[self.ndf_test_x.columns] = (
            self.xnscale_object.transform(self.ndf_test_x[self.ndf_test_x.columns])
        )
        arr = self.train_df.copy(deep=True).iloc[:, -1].values.reshape(-1, 1)[:, 0]
        print(arr)
        narr = self.ynscale_object.fit_transform(arr)
        self.ndf_train_y = Series(narr)
        print(self.ndf_train_y)
        '''
        self.ndf_train_y = Series(self.ynscale_object.fit_transform(
            self.train_df.copy(deep=True).iloc[:, -1].values.reshape(-1, 1)))
        #print(len(self.ndf_train_y))
        self.ndf_test_y = self.ynscale_object.transform(
            self.test_df.copy(deep=True).iloc[:, -1].values.reshape(-1, 1))
        '''
    @staticmethod
    def train_model(train_x, train_y, epochs_to_train):
        overfitCallback = TerminateOnBaseline(monitor='acc', baseline=1)
        train_samle_length = len(train_x[0])
        model_input = Input(shape=(train_samle_length,))
        model_dense_1 = Dense(96, activation='relu')(model_input)
        # model_dropout_1 = Dropout(rate=0.5)(model_dense_1)
        model_dense_2 = Dense(48, activation='relu')(model_dense_1)
        predict_out = Dense((1), activation='hard_sigmoid')(model_dense_2)
        model = Model(inputs=model_input, outputs=predict_out)
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(train_x, train_y, epochs=epochs_to_train, callbacks=[overfitCallback])
        return history, model

    @staticmethod
    def predict_model(model, array):
        predicted = model.predict(np.expand_dims(array, axis=0))
        return np.round(np.squeeze(predicted))

    def get_model(self):
        x = self.ndf_train_x.values.astype(dtype='float64')
        y = self.ndf_train_y.values.astype(dtype='float64')
        # print(len(x))
        # print(len(y))
        _, model = self.train_model(x, y, 10)
        return model

    def test_model(self, model):
        # print(normalized_test_df)
        test_y = self.ndf_test_x.values
        test_x = self.ndf_test_y.values
        scores = model.evaluate(x=test_x, y=test_y, verbose=1)
        return scores

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

    def test_model_loop(self, model):
        predicted_list = []
        for i in range(self.test_df_xs.shape[0]):
            # print(self.test_df_xs.loc[i].values)
            result = self.predict_model(model, self.test_df_xs.loc[i].values)
            predicted_list.append(result)
        d = {'predicted': predicted_list, 'anoted': self.test_df_ys.values}
        result_df = DataFrame(data=d)
        result_df.to_excel("output.xlsx")
        print(result_df)


if __name__ == "__main__":
    square_series_folder = PurePath(os.getcwd(), 'square_db')
    tringle_series_folder = PurePath(os.getcwd(), 'triangle_db')
    df_arff2_filename = PurePath(os.getcwd(), 'squre_triangle').with_suffix('.arff')
    dfo = CreateDf()
    # df.create_df_from_folders([square_series_folder, tringle_series_folder], ['0', '1'])
    # df.save_to_arff2(df_arff2_filename)
    dfo.load_from_arff2(df_arff2_filename)
    dfd = dfo.get_df()
    anno = SynthANN(dfd)
    anno.truncate_data(rows_to_store=200)
    # print(anno.df)
    anno.split_data()
    anno.normalize_df()
    # print(anno.ndf_train_x)
    # print(anno.ndf_train_y)
    model = anno.get_model()
    scores = anno.test_model(model)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
