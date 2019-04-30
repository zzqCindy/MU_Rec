import numpy as np
import pandas as pd
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing import text
from sklearn.utils import shuffle
import nltk
from nltk.corpus import stopwords
from gensim.models import KeyedVectors

names = ["class", "title", "content"]

lemma = nltk.wordnet.WordNetLemmatizer()
sno = nltk.stem.SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

def preprocessing(filename):
    docu_csv = pd.read_csv(filename, encoding = 'unicode_escape', usecols=[0,2])
    docu_csv.dropna(inplace=True)
    label = list(docu_csv.iloc[:,0])
    raw_abstract = list(docu_csv.iloc[:,1])
    abstract = []
    for document in raw_abstract:
        words = text.text_to_word_sequence(document)
        doc = []
        for word in words:
            if word not in stop_words:
                word = lemma.lemmatize(word)
                word = sno.stem(word)
                doc.append(word)
        tmp_abstract = ' '.join(doc)
        abstract.append(tmp_abstract)
    data_label = {'Label': label,'Abstract': abstract}
    df = pd.DataFrame(data_label, columns = ['Label','Abstract'])
    df.to_csv('../dataset/abstract.csv')

def save_to_npy(label,data):
    model = KeyedVectors.load_word2vec_format('../model/PubMed-and-PMC-w2v.bin', binary=True)
    vector = []
    for i in range(len(data)):
        if i != 0 and label[i] != label[i-1]:
            np.save('../dataset/vector_%d.npy' %label[i-1], np.array(vector))
            vector = []
        x_test = np.zeros((350, 200))
        index = 0
        abstract = data[i].split(' ')
        for word in abstract:
            if len(word) < 2:
                continue
            if index == 350:
                break
            if word in model.wv.vocab:
                x_test[index] = model.wv[word]
            else:
                x_test[index] = np.array([0]*200)
            index += 1
        vector.append(x_test)
    np.save('../dataset/vector_%d.npy' %label[len(label)-1], np.array(vector))


# preprocessing('../dataset/roughdata.csv')
docu_csv = pd.read_csv('../dataset/abstract.csv', encoding = 'unicode_escape')
docu_csv.dropna(inplace=True)
save_to_npy(list(docu_csv.iloc[:,1]),list(docu_csv.iloc[:,2]))

def to_one_hot(y, n_class):
    return np.eye(n_class)[y.astype(int)]

def load_data(file_name, sample_ratio=1, n_class=15, names=names, one_hot=True):
    '''load data from .csv file'''
    csv_file = pd.read_csv(file_name, names=names)
    shuffle_csv = csv_file.sample(frac=sample_ratio)
    x = pd.Series(shuffle_csv["content"])
    y = pd.Series(shuffle_csv["class"])
    if one_hot:
        y = to_one_hot(y, n_class)
    return x, y

def data_preprocessing(train, test, max_len, max_words=50000):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train)
    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)
    train_words = tokenizer.sequences_to_texts(train_idx)
    test_words = tokenizer.sequences_to_texts(test_idx)
    train_padded = pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded, test_padded, max_words + 2, train_words, test_words, tokenizer

def data_preprocessing_with_dict(train, test, max_len):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token='<UNK>')
    tokenizer.fit_on_texts(train)
    train_idx = tokenizer.texts_to_sequences(train)
    test_idx = tokenizer.texts_to_sequences(test)
    train_padded = pad_sequences(train_idx, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    # vocab size = len(word_docs) + 2  (<UNK>, <PAD>)
    return train_padded, test_padded, tokenizer.word_docs, tokenizer.word_index, len(tokenizer.word_docs) + 2

def data_preprocessing_release(test, tokenizer,max_len =32,max_words= 50000):
    test_idx = tokenizer.texts_to_sequences(test)
    test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
    return test_padded, max_words+2

def split_dataset(x_test, y_test, dev_ratio):
    """split test dataset to test and dev set with ratio """
    test_size = len(x_test)
    print(test_size)
    dev_size = (int)(test_size * dev_ratio)
    print(dev_size)
    x_dev = x_test[:dev_size]
    x_test = x_test[dev_size:]
    y_dev = y_test[:dev_size]
    y_test = y_test[dev_size:]
    return x_test, x_dev, y_test, y_dev, dev_size, test_size - dev_size

def fill_feed_dict(data_X, data_Y, batch_size):
    """Generator to yield batches"""
    # Shuffle data first.
    shuffled_X, shuffled_Y = shuffle(data_X, data_Y)
    # print("before shuffle: ", data_Y[:10])
    # print(data_X.shape[0])
    # perm = np.random.permutation(data_X.shape[0])
    # data_X = data_X[perm]
    # shuffled_Y = data_Y[perm]
    # print("after shuffle: ", shuffled_Y[:10])
    for idx in range(data_X.shape[0] // batch_size):
        x_batch = shuffled_X[batch_size * idx: batch_size * (idx + 1)]
        y_batch = shuffled_Y[batch_size * idx: batch_size * (idx + 1)]
        yield x_batch, y_batch


