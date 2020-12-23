import json
import re 
from tqdm import tqdm
from nltk import sent_tokenize

from gensim import utils

import string

def preprocess_text(text):
    text = re.sub("\n", " ", text.strip())
    text = re.sub("'+", '"', text.strip())
    text = re.sub('"+', '"', text.strip())
    text = re.sub("[#$%!@^_?*=;]", " ", text.strip())
    text = re.sub("  +", " ", text.strip())
    return text

def get_sentences_from_datri_news(file_path='data/IR_data/raw_dantri.json'):
    with open('data/raw_dantri.json', mode='r', encoding='utf-8') as fin:
        data = json.loads(fin.read())

    corpus = []
    i = 0
    for doc in tqdm(data['response']['docs']):
        try:
            #message = preprocess(doc['message'])
            message = doc['message']
            message = message.replace('\n', '.')
            for sentence in sent_tokenize(message):
                if len(sentence.split()) >= 20:
                    corpus.append(sentence)
        except:
            print("\nError: ", i)
            i += 1
                
    print(len(corpus))
    corpus = list(set(corpus))
    print(len(corpus))

    with open('./data/IR_data/dantri.json', mode='w', encoding='utf-8') as fout:
        fout.write(json.dumps(corpus, ensure_ascii=False, indent=4))
    
def get_sentences_from_wiki(file_path=None):
    corpus = []
    with utils.open(file_path, 'rb') as f:
        for i, line in tqdm(enumerate(f)):
            article = json.loads(line)
            text = ". ".join(article['section_texts'])
            text = preprocess_text(text)
            
            for sentence in sent_tokenize(text):
                if len(sentence.split()) >= 15:
                    corpus.append(sentence)
                    
    print(len(corpus))
    corpus = list(set(corpus))
    print(len(corpus))
       
    with open('./data/IR_data/wiki_data_corpus.json', mode='w', encoding='utf-8') as fout:
        fout.write(json.dumps(corpus, ensure_ascii=False, indent=4))
    
    print("Successfully load %d sentence from %d articles from wiki" %(len(corpus), i))
    return 0

def count_sample():
    dataset = json.loads(open('./data/QnA_data/zalo_train.json', 'r', encoding='utf-8').read())
    count = [0]*15
    MAX = 0
    index = -1
    data = []
    
    for i, record in enumerate(dataset):
        #count[len(sent_tokenize(record['text']))] += 1
        if MAX < len(sent_tokenize(record['text'])):
            MAX = len(sent_tokenize(record['text']))
            index = i
    # for record in dataset:
    #     if "vì sao" in record['question'].lower() or "nguyên nhân" in record['question'].lower():
    #         count += 1
    #         data.append(record)
    # with open('cause-result.json', 'w', encoding='utf-8') as fout:
    #     fout.write(json.dumps(data, ensure_ascii=False, indent=4))
    print(MAX)
    print(dataset[index]['question'])

count_sample()


        
    

