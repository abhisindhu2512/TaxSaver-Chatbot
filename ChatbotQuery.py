import pandas as pd
from typing import List
import numpy as np
from numpy.linalg import norm
from prompt import *
import json
import re

class chatbot:

    def __init__(self,llm_service,chat_model,embedding_model,document):

        self.llm_service = llm_service
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        document = document.replace(".pdf","")
        self.database = pd.read_csv(f"{document}.csv")

    def rephrase_question(self,user_question,chat_history):

        system_prompt = f'''
        Actual Task Input:
        system_prompt:{rephrase_template}
        chat_history:{chat_history}

        Actual Task Output:'''

        rephrase_question = self.llm_service.get_chat_completion(system_message = system_prompt,user_message = user_question,model= self.chat_model)

        return rephrase_question
    
    def string_to_list(self,text:str)-> List:
        text = re.sub(r"[\[\]]", "", text)
        text = text.split(',')
        text = [float(i) for i in text]
        return text
    
    def cosine_similarity(self,a:List,b:List):
        cosine = np.dot(a,b)/(norm(a)*norm(b))
        return cosine
    
    def retrieve_context(self,rephrase_question):
       
        self.database['ContentVector'] = self.database['ContentVector'].astype('str')
        self.database['ContentVector'] = self.database['ContentVector'].apply(lambda x:self.string_to_list(x))
        vectorize_query = self.llm_service.get_embedding(text = rephrase_question,model=self.embedding_model)
        self.database['Score'] = self.database['ContentVector'].apply(lambda x:self.cosine_similarity(vectorize_query,x))
        top_contexts =  self.database.sort_values('Score',ascending=False).head(3).reset_index(drop=True)
        context = ""
        context = ""
        for i, row in top_contexts.iterrows():
            context += f"{row['PageContent']}\n************************\n"

        return context
    
    def query_answering(self,query):

        try:
            with open('chat_history.json', 'r') as file:
                chat_history  = json.load(file)

        except:
            chat_history = []

        rephrased_question = self.rephrase_question(user_question = query,chat_history = chat_history)
        context = self.retrieve_context(rephrase_question = rephrased_question)

        system_prompt = f'''
        Actual Task Input:
        system_prompt:{query_template}
        context:{context}

        Actual Task Output:'''

        answer = self.llm_service.get_chat_completion(system_message=system_prompt,user_message = query,model=self.chat_model)

        return answer
