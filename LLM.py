from openai import OpenAI
from typing import List

class LLMService:

    """
    A service class for interacting with OpenAI's API.
    
    This class provides methods to generate chat completions and retrieve text embeddings 
    using OpenAI's models.
    
    Attributes:
        client (OpenAI): An instance of the OpenAI client initialized with an API key.
    """

    def __init__(self,api_key:str)->None:

        """
        Initializes the LLMService with the provided OpenAI API key.

        Args:
            api_key (str): The API key used to authenticate with OpenAI.
        """
        self.client = OpenAI(api_key=api_key)


    def get_chat_completion(self,system_message:str,user_message:str,model:str)->str:

        """
        Generates a chat response based on a system message and user input.

        Args:
            system_message (str): The system's instruction or context for the chat.
            user_message (str): The user's message or query.
            model (str): The name of the OpenAI chat model to use.

        Returns:
            str: The generated response from the model.
        """

        completion = self.client.chat.completions.create(
        model= model,
        messages=[{"role": "system", "content": system_message},
        {"role": "user","content": user_message}]
        )
    
        result = completion.choices[0].message.content

        return result
    
    
    def get_embedding(self,text:str,model:str)->List[float]:

        """
        Retrieves the embedding vector for a given text using OpenAI's embedding model.

        Args:
            text (str): The input text to be embedded.
            model (str): The name of the embedding model to use.

        Returns:
            List[float]: A list of floating-point numbers representing the text embedding.
        """

        response = self.client.embeddings.create(input = [text], model= model,encoding_format="float")
        embedding = response.data[0].embedding

        return embedding
    

 


