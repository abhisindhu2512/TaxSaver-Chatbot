from pypdf import PdfReader
from typing import List
import pandas as pd
import re


class Create_DataBase:

    def __init__(self,llm_service:str,embedding_model:str,document:str)-> None:
        
        self.llm_service = llm_service
        self.document = document
        self.embedding_model = embedding_model
    

    def parse_pdf(self,verbose=True)-> List:
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
        text = re.sub(r'\s+',  ' ', text).strip()
        text = re.sub(r". ,","",text)
        # remove all instances of multiple spaces
        text = text.replace("..",".")
        text = text.replace(". .",".")
        text = text.replace("\n", "")
        text = text.strip()

        return text
    
    def transform_pdf_to_dataframe(self):
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

        


