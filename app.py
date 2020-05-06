from flask import Flask, render_template, request
from model import *

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        text = request.form['rawtext']
        sents = text.split()
        tags = ner_tagger.predict(text)

    return render_template("index.html", sents=sents, tags=tags[0])


if __name__ == '__main__':
    ner_tagger = NERTagger()
    ner_tagger.load()

    app.run(debug=True)