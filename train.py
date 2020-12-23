from typing import Tuple, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from transformers import AutoModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import data_processor
from utils import convert_data_to_tensor, create_data_loader
from models import PhoBertModel



tokenizer_path = 'models/phobert-pretrained'
roberta_path = 'models/phobert-pretrained'

BATCH_SIZE = 32
MAX_LEN = 300
EPOCHS = 8
device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device('cpu')


tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

def convert_data_to_tensor(data: List[Tuple],
                           label: List[int]):
    
    X = tokenizer.batch_encode_plus(data, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
    y = torch.tensor(label, dtype=torch.long)
    
    data_tensor = TensorDataset(X['input_ids'], X['attention_mask'], y)
    
    return data_tensor

def create_data_loader(data_tensor, shuffle=True):
    
    data_loader = DataLoader(data_tensor, batch_size=BATCH_SIZE, shuffle=shuffle)
    return data_loader

def initialize_model(train_tensor, lr=3e-5, num_warmup_steps=200):
    model = PhoBertModel(roberta_path)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
            {'params': [param for name, param in param_optimizer if not any(nd in name for nd in no_decay)],
                'weight_decay_rate': 0.01},
                {'params': [param for name, param in param_optimizer if any(nd in name for nd in no_decay)],
            'weight_decay_rate': 0.0}
    ]
    #model.to(device)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    n_steps = int(len(train_tensor)/BATCH_SIZE) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_training_steps=n_steps, num_warmup_steps=num_warmup_steps)
    loss_fn = nn.CrossEntropyLoss()
    return model, optimizer, scheduler, loss_fn


def step(model, optimizer, scheduler, loss_fn, batch):
    model.train()
    input_ids, attention_mask, label = tuple(t.to(device) for t in batch)
    optimizer.zero_grad()

    y_pred = model.forward(input_ids, attention_mask)
    loss = loss_fn(y_pred, label)
    loss.backward()

    optimizer.step()
    scheduler.step()

    return loss.item()

def validate(model, loss_fn, test_loader):
    print("Evaluating....")
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        accuracy = 0
        all_y_true = []
        all_y_pred = []
        for i, batch in enumerate(test_loader):
            input_ids, attention_mask, y_true = tuple(t.to(device) for t in batch)
            output = model.forward(input_ids, attention_mask)
            loss = loss_fn(output, y_true)

            total_loss += loss.item()
            y_pred = output.argmax(1)

            all_y_true.extend(list(y_true.to('cpu').numpy()))
            all_y_pred.extend(list(y_pred.to('cpu').numpy()))

        val_loss = total_loss/len(test_loader)
        accuracy = accuracy_score(all_y_true, all_y_pred)
        f1 = f1_score(all_y_true, all_y_pred)

    return val_loss, accuracy, f1 

def train(model, optimizer, scheduler, loss_fn, train_tensor, test_tensor):
    max_f1 = 0.8
    test_loader = create_data_loader(test_tensor, shuffle=False)
    for epoch in range(EPOCHS):
        total_loss = 0.0
        train_loader = create_data_loader(train_tensor, shuffle=True)
        for i, batch in enumerate(train_loader):
            loss = step(model, optimizer, scheduler, loss_fn, batch)
            total_loss += loss 
            if (i + 1) % 50 == 0:
                print("Epoch: {} - iter: {}/{} - train_loss: {}".format(epoch, i + 1, len(train_loader), total_loss/(i + 1)))

        val_loss, accuracy, f1 = validate(model, loss_fn, test_loader)
        print("Val_loss: {} - Accuracy: {} - F1-score: {}".format(val_loss, accuracy, f1))

        if f1 > max_f1:
            max_f1 = f1
        torch.save(model.state_dict(), f'./models/finetuned/model_epoch_{epoch}.pt')
        with open('./models/finetuned/epoch_{}.txt'.format(epoch), mode='w', encoding='utf-8') as f:
            f.write("Val_loss: {} - Accuracy: {} - F1-score: {} - Max F1: {}".format(val_loss, accuracy, f1, max_f1))

        print(f"Saved english model at epoch {epoch}.")

if __name__ == '__main__':
    train_data = []
    train_labels = []
    
    datasets, labels = data_processor.load_data("./data/QnA_data/zalo_train.json")
    train_data, test_data, train_labels, test_labels = train_test_split(datasets, labels, test_size=0.2, random_state=42)
    
    
    print("Total training examples : ", len(train_data))
    
    
    print("Converting to tensor...")
    train_tensor = convert_data_to_tensor(train_data, train_labels)
    test_tensor = convert_data_to_tensor(test_data, test_labels)
    
    print("Loading PhoBert....")
    model, optimizer, scheduler, loss_fn = initialize_model(train_tensor, lr=2e-5, num_warmup_steps=50)
    model.to(device)
    print("Starting training...")
    train(model, optimizer, scheduler, loss_fn, train_tensor=train_tensor, test_tensor=test_tensor)
    
    
    
    



    
    
    
    
    