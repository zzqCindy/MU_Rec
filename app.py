from flask import Flask, render_template, request, jsonify, abort
import json, warnings, os
warnings.filterwarnings('ignore')  # To ignore all warnings that arise here to enhance clarity
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import WordPunctTokenizer

app = Flask(__name__)

tasks = []
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
    if len(tasks) > 0:
        id = tasks[-1]['id']+1
    else:
        id = 1
    document = request.json['content']
    text = WordPunctTokenizer().tokenize(document)     # Extract tokens
    words = [dic.get(word, word) for word in text if (len(word) > 1 and word not in stopwords)]
    new_corpus = dictionary.doc2bow(words)
    vector = lda[new_corpus]
    task = {
        'id': id,
        'content': document,
        'value': str(vector)
    }
    tasks.append(task)
    return jsonify({'result': 'success'})


@app.route('/get_task/', methods=['GET'])
def get_task():
    if not request.args or 'id' not in request.args:
        # 没有指定id则返回全部
        return jsonify(tasks)
    else:
        task_id = request.args['id']
        task = filter(lambda t: t['id'] == int(task_id), tasks)
        return jsonify(task) if task else jsonify({'result': 'not found'})



if __name__ == "__main__":
    app.run(port = 8881)
