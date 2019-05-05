import numpy as np
import pandas as pd
from keras.preprocessing import text
import nltk, os, re
from nltk.corpus import stopwords
from gensim.models import KeyedVectors

names = ["class", "title", "content"]

lemma = nltk.wordnet.WordNetLemmatizer()
sno = nltk.stem.SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

def preprocessing(filename):
    docu_csv = pd.read_csv(filename, encoding = 'unicode_escape', usecols=[0,2], header=None)
    docu_csv.dropna(inplace=True)
    label = list(docu_csv.iloc[:,0])
    raw_abstract = list(docu_csv.iloc[:,1])
    abstract = []
    for document in raw_abstract:
        document = re.sub('[^A-Za-z]+', " ", document)
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
    df.to_csv(filename.replace('raw','data'),index=False,header=False)

def save_to_npy(filename):
    docu_csv = pd.read_csv(filename, encoding = 'unicode_escape', header=None)
    docu_csv.dropna(inplace=True)
    data = list(docu_csv.iloc[:,1])
    vector = []
    count = 0
    for docu in data:
        if count >= 1000:
            break
        count += 1
        x_test = np.zeros((250, 200))
        index = 0
        abstract = docu.split(' ')
        for word in abstract:
            if len(word) < 2:
                continue
            if index == 250:
                break
            if word in model.wv.vocab:
                x_test[index] = model.wv[word]
            else:
                x_test[index] = np.array([0]*200)
            index += 1
        vector.append(x_test)
    np.save(filename.replace('csv','npy').replace('data/','npy/'), np.array(vector))

dir = '../dataset/data/'
files = os.listdir(dir)
model = KeyedVectors.load_word2vec_format('../model/PubMed-and-PMC-w2v.bin', binary=True)
count = 0
for file in files:
    save_to_npy(dir+file)
