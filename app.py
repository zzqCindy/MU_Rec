from flask import Flask, request, abort
import json, warnings, pickle, re, nltk, numpy as np
from nltk.corpus import stopwords
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import WordPunctTokenizer
from database.abstract_topic import Abstract
from keras.models import load_model
from keras.preprocessing import text
from flask import Response
from keras.preprocessing import sequence

app = Flask(__name__)

stopword = []
lemma = nltk.wordnet.WordNetLemmatizer()
sno = nltk.stem.SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

with open("model/LDA/stopwords.txt", "r") as doc:
    contents = doc.readlines()
    for name in contents:
        name = name.strip('\n')
        stopword.append(name)

with open("model/LDA/dict.json", "r") as f:
    dic = json.load(f)

dictionary = Dictionary.load_from_text('model/LDA/dictionary')
lda = LdaModel.load('model/LDA/lda10.model')
# BLSTM = load_model('model/BiLSTM/model/token_model.h5')
with open('model/BiLSTM/model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def lda_preprocessing(lda_data):
    text = WordPunctTokenizer().tokenize(lda_data)     # Extract tokens
    words = [dic.get(word, word) for word in text if (len(word) > 1 and word not in stopword)]
    new_corpus = dictionary.doc2bow(words)
    vector = lda[new_corpus]
    top_list = {}
    for s in vector:
        top_list[s[0]] = s[1]
    return top_list


def raw_preprocessing(raw_abstract):
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
    return abstract


@app.route('/recommend_list/', methods=['GET','POST'])
def recommend_list():
    data_list = json.loads(request.data.decode("utf-8"))
    label = [0]*13
    abstract = raw_preprocessing([data['abstractContent'] for data in data_list if len(data['abstractContent']) > 10])
    token = tokenizer.texts_to_sequences(abstract)
    train = sequence.pad_sequences(token, maxlen=200)
    model = load_model('model/BiLSTM/model/token_model.h5')
    pred = model.predict(train)
    for idx in np.argmax(pred,axis=1):
        label[idx] += 1
    pre = label.index(max(label))
    abs_data = Abstract(lda_preprocessing(' '.join(abstract)),pre)
    data = abs_data.recom()
    # 返回Jason格式的响应
    return Response(json.dumps(data),  mimetype='application/json')


if __name__ == "__main__":
    app.run(port = 8881)
