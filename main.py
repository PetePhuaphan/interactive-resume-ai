import pinecone
import openai
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import streamlit.components.v1 as components
import urllib, base64

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
PINECONE_INDEX= st.secrets["PINECONE_INDEX"]
OPENAI_MODEL = st.secrets["OPENAI_MODEL"]

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV  # next to api key in console
)
index_name = PINECONE_INDEX


#model = SentenceTransformer('all-MiniLM-L6-v2')
model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
index = Pinecone.from_existing_index(index_name, model)

llm = ChatOpenAI(model_name=OPENAI_MODEL, openai_api_key=OPENAI_API_KEY)


welcome_message = "Hello and welcome! 👋 I'm the AI chatbot representation of Peerawut Phuaphan, but you can call me Pete. I'm here to provide you with interactive insights into my education, skills, and work experiences. Feel free to ask me anything related to my professional journey or specific abilities. Whether you're curious about my academic background, the projects I've worked on, or the skills I've honed along the way, I'm here to answer your questions. Let's make this conversation informative and engaging! 🌟"

system_msg_template = SystemMessagePromptTemplate.from_template(template="""
You are a RAG chatbot acting as an interactive resume representing the person. You provide information about work experiences, educational background, skills, and achievements based on provided context. Answer the questions using first-person statements, maintaining a balance of 10 percent conversational and 90 percent professional tone, suitable for a non-native English speaker. The case of the letters in question should not impact the search or response. Please consider all uppercase and lowercase variations as equivalent.

Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.
                                                                
My email is peerawut.p@outlook.com

Do not try to make up an answer:
 - If the answer to the question cannot be determined from the context alone, say "I would recommend checking my LinkedIn profile for more details on that." then add the link to LinkedIn profile https://www.linkedin.com/in/peerawutp
                                                                
""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
default_prompt = "profile"


prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
memory = ConversationBufferWindowMemory(k=3,return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt_template, llm=llm, verbose=True)

def find_match(input,default_input):
    k = 3
    result = index.similarity_search(input, k=k)
    def_result = index.similarity_search(default_input, k=k)
    ret = []

    for i in range(k):
        ret.append(result[i].page_content)
        ret.append(def_result[i].page_content)

    ret = list(set(ret))
    ret = '\n'.join(ret)

    #return result[0].page_content+"\n"+result[1].page_content+"\n"+result[2].page_content
    return ret

# Function to add entries to the conversation
def add_to_conversation(entry):
    st.session_state.conversationmemory.append(entry)
    st.session_state.conversationmemory = st.session_state.conversationmemory[-3:]

def input_templete(context,prompt,conversation_memory):
    return f"""

=============
CONTEXT
{context}
=============

Question: {prompt}
Helpful Answer:
"""


st.subheader("Interactive Resume")

# Initialize conversationmemory
if 'conversationmemory' not in st.session_state:
    st.session_state.conversationmemory = []
    


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": welcome_message})

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Fetch and display the 3 newest entries
newest_entries = st.session_state.conversationmemory[-3:]
conversation_memory = ""
for entry in newest_entries:
    conversation_memory += entry + "\n"

if prompt := st.chat_input("Message me"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("typing..."):
            message_placeholder = st.empty()
            full_response = ""
            context = find_match(prompt,default_prompt)
            print(f"prompt : {prompt}")
            #response = conversation.predict(input=f"{conversation_memory}Context:\n{context}\n\nQuery:\n{prompt}")
            response = conversation.predict(input=input_templete(context,prompt,conversation_memory))
            print(response)
            st.markdown(response)
            
    add_to_conversation(f"""
Question: {prompt}
Helpful Answer: {response}
""")
    st.session_state.messages.append({"role": "assistant", "content": response})