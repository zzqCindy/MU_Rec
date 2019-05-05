import pymongo, json, warnings, re
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import WordPunctTokenizer

stopwords = []

with open("../model/LDA/stopwords.txt", "r") as doc:
    contents = doc.readlines()
    for name in contents:
        name = name.strip('\n')
        stopwords.append(name)

with open("../model/LDA/dict.json", "r") as f:
    dic = json.load(f)

dictionary = Dictionary.load_from_text('../model/LDA/dictionary')
lda = LdaModel.load('../model/LDA/lda10.model')

client = pymongo.MongoClient('127.0.0.1', 27017)
db = client.test
coll_topic = db.abstract_topic
db_list = ['Analytical', 'Anatomy','Anthropology','Chemicals','Disciplines', 'Diseases','Health',
        'Humanities','Information','Organisms', 'Phenomena', 'Psychiatry', 'Technology']
for i in range(len(db_list)):
    print(db_list[i])
    coll = db['%s_publication'%db_list[i]]
    publi = list(coll.find({'publishDate':re.compile("2019-04-.*")}).limit(1000))

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
        abstract['_id'] = dict['_id']
        abstract['label'] = i
        for j in range(0,10):
            abstract['topic%d'%j] = float(top_list.get(j,0))
        coll_topic.insert_one(abstract)
