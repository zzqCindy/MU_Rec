'''
#Trains a Bidirectional LSTM on the IMDB sentiment classification task.
Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.
'''

from __future__ import print_function
import numpy as np, tensorflow as tf
import random, os, time, json, pandas as pd
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import pickle
from keras.models import load_model


max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 200
batch_size = 32

def load_data_csv(datapath):
    files = os.listdir(datapath)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for filename in files:
        docu_csv = pd.read_csv(datapath+filename, nrows=1000, encoding = 'unicode_escape', header=None)
        abstract = list(docu_csv.iloc[:,1])
        label = int(docu_csv.iloc[0,0])
        random.shuffle(abstract)
        x_train.extend(abstract[:800])
        x_test.extend(abstract[800:])
        y_train += [label]*800
        y_test += [label]*200
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(x_train+x_test)
    with open('tmp/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer,handle)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (x_train,y_train,x_test,y_test)

def load_data_npy(datapath):
    files = os.listdir(datapath)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for file in files:
        array = np.load(datapath+file)
        random.shuffle(array)
        x_train.extend(array[:800])
        x_test.extend(array[800:])
        label = int(file[0:2])
        y_train += [label]*800
        y_test += [label]*200
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return (x_train,y_train,x_test,y_test)

def word2vec_BLSTM():
    print('Loading data...')
    (x_train, y_train, x_test, y_test) = load_data_npy('dataset/npy/')

    # print('Pad sequences (samples x time)')
    # x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    # x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(y_train.shape)

    model = Sequential()
    # model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(13, activation='softmax'))

    # try using different optimizers and different optimizer configs
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    print('Train...')
    early_stopping = EarlyStopping(monitor='val_acc',patience = 5, mode='max')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=200,
              validation_data=[x_test, y_test], callbacks=[early_stopping])
    model.save('tmp/word2vec_model%s.h5' %str(time.time()))

def embedding_BLSTM():
    print('Loading data...')
    (x_train, y_train, x_test, y_test) = load_data_csv('dataset/data/')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print(y_train.shape)

    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=maxlen))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(13, activation='softmax'))

    # try using different optimizers and different optimizer configs
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

    print('Train...')
    early_stopping = EarlyStopping(monitor='val_acc',patience = 5, mode='max')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=200,
              validation_data=[x_test, y_test], callbacks=[early_stopping])
    model.save('tmp/token_model%s.h5' %str(time.time()))


def embedding_predict(x):
    # docu_csv = pd.read_csv('dataset/data/Chemicals.csv', nrows=10, encoding = 'unicode_escape', header=None)
    # abstract_list = list(docu_csv.iloc[:,1])
    # with open('model/tokenizer.pickle', 'rb') as handle:
    #     tokenizer = pickle.load(handle)
    # x = tokenizer.texts_to_sequences(abstract_list)
    # x = sequence.pad_sequences(x, maxlen=maxlen)
    model = load_model('model/token_model.h5')
    print(model.predict(x))

# embedding_predict()
