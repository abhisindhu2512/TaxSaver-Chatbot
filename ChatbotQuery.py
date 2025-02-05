import pandas as pd
from typing import List
import numpy as np
from numpy.linalg import norm
from prompt import *
import json
import re

class chatbot:

    """
    A chatbot class that interacts with an LLM service for question-answering. 
    It retrieves relevant context from an embedded document database and generates responses.
    
    Attributes:
        llm_service: An instance of the LLM service for chat completion and embedding retrieval.
        chat_model: The name of the chat model used for generating responses.
        embedding_model: The name of the embedding model used for vector representation.
        database: A Pandas DataFrame containing the document's text and corresponding embeddings.
    """

    def __init__(self,llm_service,chat_model,embedding_model,document)->None:

        """
        Initializes the chatbot with the given models and loads the document database.

        Args:
            llm_service: The LLM service used for chat completions and embedding model.
            chat_model: The name of the chat model.
            embedding_model: The name of the embedding model.
            document: The filename of the document (without .pdf extension).
        """

        self.llm_service = llm_service
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        document = document.replace(".pdf","")
        self.database = pd.read_csv(f"{document}.csv")

    def rephrase_question(self,user_question:str,chat_history:List)->str:
        
        """
        Rephrases the user's question using the LLM service.

        Args:
            user_question: The original question from the user.
            chat_history: A list containing previous chat interactions.

        Returns:
            A rephrased version of the user's question.
        """

        system_prompt = f'''
        Actual Task Input:
        system_prompt:{rephrase_template}
        chat_history:{chat_history}

        Actual Task Output:'''

        rephrase_question = self.llm_service.get_chat_completion(system_message = system_prompt,user_message = user_question,model= self.chat_model)

        return rephrase_question
    
    def string_to_list(self,text:str)-> List:
        
        """
        Converts a string representation of a list into an actual list of floats.

        Args:
            text: A string containing a list of numbers.

        Returns:
            A list of floating-point numbers.
        """

        text = re.sub(r"[\[\]]", "", text)
        text = text.split(',')
        text = [float(i) for i in text]
        return text
    
    def cosine_similarity(self,a:List,b:List)->float:

        """
        Computes the cosine similarity between two vectors.

        Args:
            a: The first vector.
            b: The second vector.

        Returns:
            The cosine similarity score between the two vectors.
        """

        cosine = np.dot(a,b)/(norm(a)*norm(b))
        return cosine
    
    def retrieve_context(self,rephrase_question:str)->str:

        """
        Retrieves the most relevant context from the database based on the rephrased question.

        Args:
            rephrase_question: The rephrased user question.

        Returns:
            A string containing the top matching context from the document.
        """
       
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
    
    def query_answering(self,query:str)->str:

        """
        Processes a user query by rephrasing it, retrieving relevant context, and generating an answer.

        Args:
            query: The user's original question.

        Returns:
            The generated answer based on the retrieved context.
        """

        try:
            with open('chat_history.json', 'r') as file:
                chat_history  = json.load(file)

        except FileNotFoundError:
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
