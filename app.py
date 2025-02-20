import streamlit as st
## Import Libraries
from ChatbotQuery import chatbot
from LLM import LLMService
import json


## Load config file
with open('config.json') as f:
    config = json.load(f)


openai_key = config['OPENAI']['API_KEY']
chat_model = config['OPENAI']['LLM_MODEL']
embedding_model = config['OPENAI']['EMBEDDING_MODEL']
document = 'Tax Data.pdf'


llm_service = LLMService(api_key=openai_key)

chatbot = chatbot(llm_service=llm_service,chat_model=chat_model,embedding_model=embedding_model,document=document)


def response_generator(text):

    
    response = chatbot.query_answering(query = text)

    return response


st.title("🦜🔗TaxSaver Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []



# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)


# Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})