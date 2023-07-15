import streamlit as st
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
trained_model.load_state_dict(torch.load("./saved_models/BERT_model"))
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

def main():
    st.title("Text Similarity Score")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> Text Similarity Score ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)

    
    Text1 = st.text_input("Text1","Type Here")
    Text2 = st.text_input("Text2","Type Here")

    
    safe_html ="""  
    <div style="background-color:#80ff80; padding:10px >
    <h2 style="color:white;text-align:center;"></h2>
    </div>
    """
    if st.button("Predict the age"):
         output = get_STS_Score([Text1, Text2])
         st.success('Simlarity Score is :-  {}'.format(output))


if __name__=='__main__':
    main()