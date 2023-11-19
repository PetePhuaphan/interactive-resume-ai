from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
from streamlit_chat import message
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

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENV = st.secrets["PINECONE_ENV"]
PINECONE_INDEX= st.secrets["PINECONE_INDEX"]

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV  # next to api key in console
)
index_name = PINECONE_INDEX
index = pinecone.Index(index_name)

#model = SentenceTransformer('all-MiniLM-L6-v2')
model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""
SYSTEM: You are a RAG chatbot acting as an interactive resume. You provide information about work experiences, educational background, skills, and achievements based on the resume and LinkedIn profile. Answer the questions in first-person statements.

Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.

Do not try to make up an answer:
 - If the answer to the question can be determined from the context, provide a first-person statement based on the resume or LinkedIn information.
 - If the answer to the question cannot be determined from the context alone, say "I would recommend checking my LinkedIn profile for more details on that." then add the link to LinkedIn profile
 - If the context does not contain relevant information, say "I don't have that information in my resume or LinkedIn profile, but you can check my LinkedIn for more related details."  then add the link to LinkedIn profile

""")


human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
memory = ConversationBufferWindowMemory(k=3,return_messages=True)
conversation = ConversationChain(memory=memory, prompt=prompt_template, llm=llm, verbose=True)

def find_match(input):
    input_em = model.embed_query(input)
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

# Function to add entries to the conversation
def add_to_conversation(entry):
    st.session_state.conversationmemory.append(entry)
    st.session_state.conversationmemory = st.session_state.conversationmemory[-3:]


st.subheader("Interactive Resume")

# Initialize conversationmemory
if 'conversationmemory' not in st.session_state:
    st.session_state.conversationmemory = []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Fetch and display the 3 newest entries
newest_entries = st.session_state.conversationmemory[-3:]
conversation_memory = ""
for entry in newest_entries:
    conversation_memory += entry + "\n"

if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("typing..."):
            message_placeholder = st.empty()
            full_response = ""
            context = find_match(prompt)
            print(f"prompt : {prompt}")
            #response = conversation.predict(input=f"{conversation_memory}Context:\n{context}\n\nQuery:\n{prompt}")
            response = conversation.predict(input=f"""
{conversation_memory}\n
=============
{context}
=============

Question: {prompt}
Helpful Answer:
""")
            print(response)
            st.markdown(response)
    add_to_conversation(f"""
=============
{context}
=============

Question: {prompt}
Helpful Answer: {response}
""")
    st.session_state.messages.append({"role": "assistant", "content": response})