import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util

#import faiss

class FaissQuery():
    def __init__(self, index=None):
        self.index = index
    
    def build(self, X=None):
        d = X.shape[1]
        self.index = faiss.IndexFlatIP(d) 
        self.index.train(X)
        self.index.add(X)
    
    @staticmethod
    def load(path):
        return FaissQuery(faiss.read_index(path))
    
    def save(self, path):
        faiss.write_index(self.index, path)
    
    def query(self, query_vector=None, n_query=5):
        distances, indices = self.index.search(query_vector, n_query)
        return indices[0]
        
    

class RetrievalModel():
    def __init__(self, model_path, data=None, faiss_path=None, embeddings_path=None):
        self.device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
        self.model = SentenceTransformer(model_path, device=self.device)
        self.data = data
        self.sentence_embeddings = None
        self.faiss_model = None
        
        if faiss_path != None:
            self.faiss_model = FaissQuery.load(faiss_path)
        if embeddings_path != None:
            self.sentence_embeddings = self.load_sentence_embeddings(embeddings_path).cpu().numpy()
        
    def load_sentence_embeddings(self, embeddings_path):
        return torch.load(embeddings_path, map_location=self.device)
    
    def get_relevant_sentences(self, question, k=32):
        responses = []
        scores, indices = self.query_by_sbert(question, k)
        for i in indices:
            responses.append(self.data[i])
        return scores, responses
    
    def query_by_faiss(self, question, k):
        question_embedding = self.model.encode([question])
        indices = self.faiss_model.query(question_embedding, n_query=k)
        return indices
    
    def query_by_sbert(self, question, k):
        #question_embedding = self.model.encode(question, convert_to_tensor=True)
        question_embedding = self.model.encode(question)
        cos_scores = util.pytorch_cos_sim(question_embedding, self.sentence_embeddings)[0]
            
        top_results = torch.topk(cos_scores, k=k)
        scores, indices = list(top_results[0].cpu().numpy()), list(top_results[1].cpu().numpy())
        return scores, indices
    
    def build_tensor_from_data(self):
        return self.model.encode(self.data)
        

class PhoBertModel(nn.Module):
    def __init__(self, phoBert_path):
        super(PhoBertModel, self).__init__()
        self.phoBert = AutoModel.from_pretrained(phoBert_path)
        self.linear_1 = nn.Linear(768, 100)
        self.linear_2 = nn.Linear(100, 2)
        
    def forward(self, input_ids, attention_mask):
        output = self.phoBert(input_ids, attention_mask)[1]
        x = self.linear_1(output)
        x = self.linear_2(x)
        return x

class QnA(object):
    def __init__(self, phoBert_path, finetuned_PhoBert_path):
        self.device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
        print("Loading PhoBert ....")
        self.qna_model = PhoBertModel(phoBert_path)
        self.qna_model.load_state_dict(torch.load(finetuned_PhoBert_path, map_location=self.device))
        self.qna_model.to(self.device)
        
        print("Loading BPE ...")
        self.bpe = AutoTokenizer.from_pretrained(phoBert_path)
        
        self.MAX_LEN = 256 
    
    def find_answer(self, question, sentences, topk=3):
        questions = [question] * len(sentences)
        input_datas = list(zip(questions, sentences))
        
        scores = torch.tensor([], device=self.device)
        dataloader = torch.utils.data.DataLoader(input_datas, batch_size=4, shuffle=False)
        for input_data in dataloader:
            batch_input = self.bpe.batch_encode_plus(list(zip(input_data[0], input_data[1])), padding=True, truncation=True, max_length=self.MAX_LEN, return_tensors='pt')
            pred = self.qna_model.forward(batch_input['input_ids'].to(self.device), batch_input['attention_mask'].to(self.device))
            results = torch.nn.functional.softmax(pred, dim=1)
            
            score, pred_labels = results.max(1)
            for i, label in enumerate(pred_labels):
                if label == 0:
                    score[i] = 1 - score[i]
            
            scores = torch.cat((scores, score), dim=0)

        scores = scores.tolist()
        responses = list(sorted(zip(sentences, scores), key=lambda x:x[1], reverse=True))
        if topk > len(responses):
            topk = len(responses)
            
        top_responses = []
        for i in range(topk):
            top_responses.append({'answer': responses[i][0],
                                'score': str(responses[i][1])})

        return top_responses

class QuestionAnsweringModel():
    def __init__(self, args):
        self.device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
        
        print("Loading corpus ...")
        with open(args.corpus_path, 'r', encoding='utf-8') as fin:
            self.data = json.loads(fin.read())
            
        print("Loading PhoBert ....")
        self.qna_model = PhoBertModel(args.phoBert_path)
        self.qna_model.load_state_dict(torch.load(args.finetuned_PhoBert_path, map_location=self.device))
        self.qna_model.to(self.device)
        self.qna_model.eval()
        
        print("Loading SentenceBert ....")
        self.ir_model = RetrievalModel(args.ir_model_path, self.data, args.faiss_path, args.embeddings_path)
        
        print("Loading BPE ...")
        self.bpe = AutoTokenizer.from_pretrained(args.phoBert_path)
            
        self.MAX_LEN = 256 
        print("Successfully loading.")
        
        
    def get_top_answers(self, question, topk=10):
        relevant_scores, answers = self.ir_model.get_relevant_sentences(question, k=100)
        questions = [question] * len(answers)
        input_datas = list(zip(questions, answers))
        
        scores = torch.tensor([], device=self.device)
        dataloader = torch.utils.data.DataLoader(input_datas, batch_size=4, shuffle=False)
        with torch.no_grad():
            for input_data in dataloader:
                batch_input = self.bpe.batch_encode_plus(list(zip(input_data[0], input_data[1])), padding=True, truncation=True, max_length=self.MAX_LEN, return_tensors='pt')
                pred = self.qna_model.forward(batch_input['input_ids'].to(self.device), batch_input['attention_mask'].to(self.device))
                results = torch.nn.functional.softmax(pred, dim=1)
                
                score, pred_labels = results.max(1)
                for i, label in enumerate(pred_labels):
                    if label == 0:
                        score[i] = 1 - score[i]
                
                scores = torch.cat((scores, score), dim=0)
            
        if any([check for check in ['vì sao', 'tại sao', 'nguyên nhân', 'lý do', 'hệ quả', 'hậu quả', 'dẫn đến', 'gây ra', 'kết quả', 'mang lại', 'sẽ xảy ra'] if check in question.lower()]):
            scores = scores * torch.tensor(list(relevant_scores), device=self.device)
            
        scores = scores.tolist()
        responses = list(sorted(zip(answers, scores), key=lambda x:x[1], reverse=True))
        if topk > len(responses):
            topk = len(responses)
            
        top_responses = []
        for i in range(topk):
            top_responses.append({'answer': responses[i][0],
                                'score': str(responses[i][1])})

        return top_responses

        
        
    
    
    
    
    
    
    
        
        