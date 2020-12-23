import os
from datasets import load_dataset, DatasetDict
from utils import format_question, format_text, convert_data_to_tensor

from tqdm import tqdm

def download_dataset(dataset_name):
    saved_path = './datasets/' + dataset_name
    if os.path.exists(saved_path):
        raise NameError(f'dataset {dataset_name} existed !')
    
    data = load_dataset(dataset_name, 'qnli')
    data.save_to_disk(saved_path)
    print(f"Successfully to save {dataset_name} to {saved_path}")
    return None

def load_dataset_from_disk(dataset_name):
    data_path = './datasets/' + dataset_name
    if not os.path.exists(data_path):
        raise NameError(f'dataset {dataset_name} does not exist !')
    data = DatasetDict.load_from_disk(data_path)
    return data

def load_qnli_data(type_data='train'):
    data = [] 
    labels = []
    qnli_data = load_dataset_from_disk('glue')[type_data]
    print("Loading qnli data...")
    for d in tqdm(qnli_data):
        question = format_question(d['question'])
        answer = d['sentence']
        
        data.append((question, answer))
        labels.append(int(d['label']))
    
    return data, labels

def load_wiki_qa_data(type_data='train'):
    data = [] 
    labels = []
    wiki_qa_data = load_dataset_from_disk('wiki_qa')[type_data]
    print("Loading wiki qa data...")
    for d in tqdm(wiki_qa_data):
        question = format_question(d['question'])
        answer = d['answer']
        
        data.append((question, answer))
        labels.append(int(d['label']))
    return data, labels

def load_qa_zre_data(type_data='train'):
    data = [] 
    labels = []
    qa_zre_data = load_dataset_from_disk('qa_zre')[type_data]
    print("Loading qa_zre data...")
    for d in tqdm(qa_zre_data):
        question = format_question(d['question'].replace('XXX', d['subject']))
        answer = d['context']
        
        data.append((question, answer))
        if len(d['answers']) > 0:
            labels.append(1)
        else:
            labels.append(0)

    return data, labels

def load_squad_v2_data(type_data='train'):
    data = [] 
    labels = []
    
    num = 0

    squad_v2_data = load_dataset_from_disk('squad_v2')[type_data]
    print(f"Loading squad_v2 data {type_data} ...")
    for d in tqdm(squad_v2_data):
        question = format_question(d['question'])
        d['context'] = format_text(d['context'])
        if len(d['answers']['answer_start']) > 0:
            start_index = max(d['answers']['answer_start'])
            try:
                end_index = d['context'][start_index:].index(".")
                
                answer = d['context'][: start_index + end_index + 1]
                data.append((question, answer))
                labels.append(1)
                
                """
                if len(d['context'][start_index + end_index + 1: ].strip()) > 0 and num < 30000:
                    false_answer = d['context'][start_index + end_index + 1: ].strip()
                    data.append((question, false_answer))
                    labels.append(0)
                    num += 1
                """

            except ValueError:
                data.append((question, d['context']))
                labels.append(1)
                
        else:
            data.append((question, d['context']))
            labels.append(0)

    return data, labels
    
    
    