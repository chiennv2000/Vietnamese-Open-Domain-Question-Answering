from typing import Tuple, List
import re
import json

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from tqdm import tqdm

from nltk import sent_tokenize
import nltk

from models import FaissQuery

def convert_data_to_tensor(data: List[Tuple],
                           label: List[int],
                           MAX_LEN,
                           tokenizer):
    
    X = tokenizer.batch_encode_plus(data, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
    y = torch.tensor(label, dtype=torch.long)
    
    data_tensor = TensorDataset(X['input_ids'], X['attention_mask'], y)
    
    return data_tensor

def create_data_loader(data_tensor,
                       batch_size,
                       shuffle=True):
    
    data_loader = DataLoader(data_tensor, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def format_question(text):
    while not text[-1].isalnum():
        text = text[:-1]
    text += "?"
    return text

def format_text(text):
    while not text[-1].isalnum():
        text = text[:-1]
    text += "."
    return text

def docs2sents(documents: List[str]) -> List[str]:
    sentences = []
    for doc in documents:
        doc = re.sub("\n", " ", doc.strip())
        doc = re.sub(" +", " ", doc.strip())
        for sentence in sent_tokenize(doc):
            if len(re.findall('\w+', sentence)) >= 10:
                sentences.append(sentence)
    return sentences

def build_index(X, index_name):
    faiss_model = FaissQuery()
    faiss_model.build(X)
    faiss_model.save('./models/faiss-model/' + index_name + '.index')

def merge_data(file_head, monthes: List[str], fileout_path):
    full_data = []
    for m in tqdm(monthes):
        with open('./data/IR_data/' + file_head + m + '.json', 'r', encoding='utf-8') as fin:
            data = json.loads(fin.read())['response']['docs']
        for d in data:
            full_data.extend(docs2sents([d.get('message', '')]))

    full_data = list(set(full_data))
    print(len(full_data))
    with open(fileout_path, 'w', encoding='utf-8') as fout:
        fout.write(json.dumps(full_data, ensure_ascii=False, indent=4))



# with open('./data/IR_data/dantri_vector.npy', 'rb') as f:
#     X = np.load(f)
# X = torch.tensor(X)
# torch.save(X, './data/IR_data/dantri_vector.pt')
