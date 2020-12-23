from flask import Flask, render_template, request

from models import QuestionAnsweringModel
import argparse

parser = argparse.ArgumentParser(description="Question Answering Model!")
parser.add_argument('--ir-model-path', type=str, default='./models/sbert-pretrained')
parser.add_argument('--embeddings-path', type=str, default='./data/IR_data/dantri_vector.pt')
parser.add_argument('--phoBert-path', type=str, default='./models/phobert-pretrained')
parser.add_argument('--finetuned-PhoBert-path', type=str, default='./models/phobert-finetuned/model.pt')

parser.add_argument('--corpus-path', type=str, default='./data/IR_data/full_dantri.json')
parser.add_argument('--faiss-path', type=str, default=None)

args = parser.parse_args()

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(model.get_top_answers(userText, 1)[0]['answer'])


if __name__ == "__main__": 
    model = QuestionAnsweringModel(args)
    app.run(host='0.0.0.0', port=8000, debug=False) 