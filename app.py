from flask import Flask, render_template, request, jsonify, abort
import json, warnings
from bson import json_util
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import WordPunctTokenizer
from database.abstract_topic import Abstract

app = Flask(__name__)

stopwords = []

with open("./model/ZZ/stopwords.txt", "r") as doc:
    contents = doc.readlines()
    for name in contents:
        name = name.strip('\n')
        stopwords.append(name)

with open("./model/ZZ/dict.json", "r") as f:
    dic = json.load(f)

dictionary = Dictionary.load_from_text('./model/ZZ/dictionary')
lda = LdaModel.load('./model/ZZ/lda10.model')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        document = request.form['content']
        text = WordPunctTokenizer().tokenize(document)     # Extract tokens
        words = [dic.get(word, word) for word in text if (len(word) > 1 and word not in stopwords)]
        new_corpus = dictionary.doc2bow(words)
        vector = lda[new_corpus]
        return render_template('index.html', RESULT = "Result: " + str(vector))
    return render_template('index.html')

@app.route('/add_task/', methods=['POST'])
def add_task():
    if not request.json or 'content' not in request.json:
        abort(400)
    document = request.json['content']
    text = WordPunctTokenizer().tokenize(document)     # Extract tokens
    words = [dic.get(word, word) for word in text if (len(word) > 1 and word not in stopwords)]
    new_corpus = dictionary.doc2bow(words)
    vector = lda[new_corpus]
    top_list = [s[1] for s in vector]
    abstract = Abstract(document,top_list)
    abstract.save()
    return jsonify({'result': 'success'})


@app.route('/get_task/', methods=['GET'])
def get_task():
    tasks = Abstract.query_abstract()
    return json.dumps(list(tasks), default=json_util.default)



if __name__ == "__main__":
    app.run(port = 8881)
