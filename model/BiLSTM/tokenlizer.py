import tensorflow as tf
from model.BiLSTM.helperfunction.prepare_data import load_data
from model.BiLSTM.helperfunction.prepare_data import data_preprocessing_v2
import pickle
from keras_preprocessing.sequence import pad_sequences

max_len = 32
x_train, y_train = load_data("./dataset/dbpedia_data/dbpedia_csv/train.csv", sample_ratio=1e-2, one_hot=False)
x_test0, y_test = load_data("./dataset/dbpedia_data/dbpedia_csv/test.csv", one_hot=False)
x_train, x_test, vocab_size, train_words, test_words, tokenizer = data_preprocessing_v2(x_train, x_test0, max_len=32, max_words=50000)
with open('./model/saved/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./model/saved/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

test_idx = tokenizer.texts_to_sequences(x_test0)
test_padded = pad_sequences(test_idx, maxlen=max_len, padding='post', truncating='post')
print(test_padded)
print(x_test)

  #  result = sess.run(prediction, feed_dict={X: x0})
   # print(result.argmax(axis=1))
