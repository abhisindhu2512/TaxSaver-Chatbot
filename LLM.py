from openai import OpenAI
from typing import List

class LLMService:

    def __init__(self,api_key:str)->None:

        self.client = OpenAI(api_key = api_key)


    def get_chat_completion(self,system_message:str,user_message:str,model:str)->str:

        completion = self.client.chat.completions.create(
        model= model,
        messages=[{"role": "system", "content": system_message},
        {"role": "user","content": user_message}]
        )
    
        result = completion.choices[0].message.content

        return result
    
    
    def get_embedding(self,text:str,model:str)->List[float]:

        response = self.client.embeddings.create(input = [text], model= model,encoding_format="float")
        embedding = response.data[0].embedding

        return embedding
    

 


