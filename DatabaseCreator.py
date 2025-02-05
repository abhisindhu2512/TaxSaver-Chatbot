from pypdf import PdfReader
from typing import List
import pandas as pd
import re


class Create_DataBase:


    """
    A class to process and convert a PDF document into a structured database.

    This class extracts text from a PDF, normalizes it, generates embeddings, and 
    stores the processed data in a CSV file.

    Attributes:
        llm_service: An instance of the LLM service used to generate embeddings.
        embedding_model: The name of the embedding model to be used.
        document: The filename of the PDF document to be processed.
    """

    def __init__(self,llm_service:str,embedding_model:str,document:str)-> None:

        """
        Initializes the Create_DataBase class with the provided LLM service, embedding model, and document.

        Args:
            llm_service (str): The LLM service used for generating text embeddings.
            embedding_model (str): The name of the embedding model.
            document (str): The filename of the PDF document to be processed.
        """
        
        self.llm_service = llm_service
        self.document = document
        self.embedding_model = embedding_model
    

    def parse_pdf(self,verbose=True)-> List:

        """
        Extracts text from a PDF file and maps it to corresponding page numbers.

        Args:
            verbose (bool, optional): If True, prints progress messages. Defaults to True.

        Returns:
            List: A list of tuples where each tuple contains a page number and its extracted text.
        """
        page_map = []
        path = './data/'
        if verbose: print(f"Extracting text using PyPDF")
        reader = PdfReader(path + self.document)
        pages = reader.pages
        for page_num, p in enumerate(pages):
            page_text = p.extract_text()
            page_map.append((page_num,page_text))

        return page_map
    
    def normalize_text(self,text:str, sep_token = " \n ")-> str:

        """
        Cleans and normalizes extracted text by removing unnecessary spaces, line breaks, and formatting inconsistencies.

        Args:
            text (str): The raw extracted text from the PDF.
            sep_token (str, optional): The separator token to handle line breaks. Defaults to " \n ".

        Returns:
            str: The cleaned and normalized text.
        """
        text = re.sub(r'\s+',  ' ', text).strip()
        text = re.sub(r". ,","",text)
        # remove all instances of multiple spaces
        text = text.replace("..",".")
        text = text.replace(". .",".")
        text = text.replace("\n", "")
        text = text.strip()

        return text
    
    def transform_pdf_to_dataframe(self):

        """
        Processes the PDF by extracting text, normalizing it, generating embeddings, and 
        saving the structured data into a CSV file.

        The CSV file includes metadata, page content, and its corresponding embedding vector.
        """
        df = pd.DataFrame()
        for page in self.parse_pdf():
            doc_info = self.document + str(page[0])
            content = self.normalize_text(page[1])
            content_vector = self.llm_service.get_embedding(text = content,model = self.embedding_model)
            temp_df = pd.DataFrame([[doc_info,content,content_vector]],columns=['MetaData','PageContent','ContentVector'])
            df = pd.concat([df,temp_df],axis=0)
        database = self.document.replace(".pdf","")
        df.to_csv(f"{database}.csv",index=False)

if __name__ == "__main__":
    Create_DataBase()

        


