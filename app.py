from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
import torch
from transformers import BertTokenizer
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models

class STSBertModel(nn.Module):

    def __init__(self):

        super(STSBertModel, self).__init__()

        word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=128)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.sts_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    def forward(self, input_data):

        output = self.sts_model(input_data)

        return output


trained_model = STSBertModel()
trained_model.load_state_dict(torch.load("./saved_models/BERT_model",map_location=torch.device('cpu')))
tokenizer = BertTokenizer.from_pretrained("./saved_models/bert-tokenizer")

def get_STS_Score(texts):
    trained_model.to('cpu')
    trained_model.eval()

    input = tokenizer(texts, padding='max_length', max_length = 128, truncation=True, return_tensors="pt")
    input['input_ids'] = input['input_ids']
    input['attention_mask'] = input['attention_mask']
    del input['token_type_ids']

    test_output = trained_model(input)['sentence_embedding']
    sim = torch.nn.functional.cosine_similarity(test_output[0], test_output[1], dim=0).item()

    return sim


text:str = "Text Similarity Score"

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.post("/predict")
async def predict_route(text1, text2):
    try:
        # input_text = dict(input_text)
        # print(input_text)
        # text1 = input_text['text1']
        # text2 = input_text['text2']
        sim = get_STS_Score([text1, text2])
        print("Similarity Score :- ",sim)
        return sim
    except Exception as e:
        raise e
    
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)