import pymongo, json, warnings
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import WordPunctTokenizer

client = pymongo.MongoClient('127.0.0.1', 27017)
db = client.test
coll_publi = db.publication
coll_topic = db.abstract_topic
publi = coll_publi.find()

stopwords = []

with open("../model/ZZ/stopwords.txt", "r") as doc:
    contents = doc.readlines()
    for name in contents:
        name = name.strip('\n')
        stopwords.append(name)

with open("../model/ZZ/dict.json", "r") as f:
    dic = json.load(f)

dictionary = Dictionary.load_from_text('../model/ZZ/dictionary')
lda = LdaModel.load('../model/ZZ/lda10.model')

for dict in publi:
    document = dict['abstractContent']
    text = WordPunctTokenizer().tokenize(document)     # Extract tokens
    words = [dic.get(word, word) for word in text if (len(word) > 1 and word not in stopwords)]
    new_corpus = dictionary.doc2bow(words)
    vector = lda[new_corpus]
    top_list = {}
    for s in vector:
        top_list[s[0]] = s[1]
    abstract = {}
    abstract['content'] = document
    abstract['_id'] = dict['_id']
    for i in range(0,10):
        abstract['topic%d'%i] = float(top_list.get(i,0))
    coll_topic.insert_one(abstract)
