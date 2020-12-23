from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from models import QuestionAnsweringModel
import time

import uvicorn

import argparse

parser = argparse.ArgumentParser(description="Question Answering Model!")
parser.add_argument('--ir-model-path', type=str, default='./models/sbert-pretrained')
parser.add_argument('--embeddings-path', type=str, default='./data/IR_data/tensor_data.pt')
parser.add_argument('--phoBert-path', type=str, default='./models/phobert-pretrained')
parser.add_argument('--finetuned-PhoBert-path', type=str, default='./models/phobert-finetuned/model.pt')

parser.add_argument('--corpus-path', type=str, default='./data/IR_data/dantri_corpus.json')
parser.add_argument('--faiss-path', type=str, default='./models/faiss-model/dantri_vectors.index')

args = parser.parse_args()

model = QuestionAnsweringModel(args)

class Input(BaseModel):
    question: str
    topk: Optional[int] = 3

app = FastAPI()

@app.post("/")
async def create_item(input: Input):
    s = time.time()
    responses = model.get_top_answers(input.question, input.topk)
    total_time = time.time() - s
    return {'responses': responses,
            'total_time': str(total_time)}

if __name__ == "__main__":
    uvicorn.run('api:app', host='0.0.0.0', port=8000, debug=True)
    