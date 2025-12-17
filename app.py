import os
import streamlit as st
import validators
from dotenv import load_dotenv

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Streamlit setup
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 RAG Chatbot")

# Session state initialization
if "store" not in st.session_state:
    st.session_state.store = {}
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Sidebar for API keys & mode selection
with st.sidebar:
    GROQ_API_KEY = st.text_input("Groq API Key", type="password")
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
    MODE = st.radio("Mode", ["Website RAG", "OpenAI Chat"])

    if st.button("Initialize"):
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        st.success("Initialized")

# Cached embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = get_embeddings()

# ---------------- Website RAG ----------------
if MODE == "Website RAG":
    url = st.text_input("Website URL")
    if st.button("Load Website"):
        if not validators.url(url):
            st.error("Invalid URL")
        else:
            loader = UnstructuredURLLoader(
                urls=[url],
                ssl_verify=False,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = splitter.split_documents(docs)

            st.session_state.vectorstore = Chroma.from_documents(
                splits, embeddings
            )
            st.success("Website loaded")

# ---------------- Chat with RAG ----------------
if MODE == "Website RAG" and st.session_state.vectorstore:
    retriever = st.session_state.vectorstore.as_retriever(k=4)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        groq_api_key=GROQ_API_KEY
    )

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rephrase the question using chat history."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever, qa_chain
    )

    def get_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    chat_chain = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    if user_input := st.chat_input("Ask something"):
        response = chat_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default"}}
        )
        st.chat_message("assistant").write(response["answer"])

# ---------------- OpenAI Chat ----------------
if MODE == "OpenAI Chat":
    if prompt := st.chat_input("Say something"):
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=OPENAI_API_KEY
        )
        response = llm.invoke(prompt)
        st.chat_message("assistant").write(response.content)
