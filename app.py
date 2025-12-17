import os
import logging
import validators
import streamlit as st
from dotenv import load_dotenv
import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
    st.warning(
        "BeautifulSoup not installed. Some web scraping features may not work. "
        "Install with: pip install beautifulsoup4"
    )

# ------------------------------------------------------------------
# ❌ REMOVED nest_asyncio (BREAKS Streamlit + uvloop in Docker/K8s)
# import nest_asyncio
# nest_asyncio.apply()
# ------------------------------------------------------------------

# LangChain imports
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader,
    PyPDFLoader,
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# LLMs
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load env vars
load_dotenv()

# ------------------------------------------------------------------
# Streamlit config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Chatbot App",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Unified RAG Chatbot with LangChain, HuggingFace, OpenAI & Groq")

# ------------------------------------------------------------------
# Session State
# ------------------------------------------------------------------
if "store" not in st.session_state:
    st.session_state.store = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "app_initialized" not in st.session_state:
    st.session_state.app_initialized = False

if "config_valid" not in st.session_state:
    st.session_state.config_valid = False

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def initialize_environment(groq_key, hf_token, openai_key, langchain_key):
    os.environ["GROQ_API_KEY"] = groq_key
    os.environ["HF_TOKEN"] = hf_token
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["LANGCHAIN_API_KEY"] = langchain_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Unified-RAG-Chatbot"

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 API Keys")

    groq_api_key = st.text_input("Groq API Key", type="password")
    hf_token = st.text_input("HuggingFace Token", type="password")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    langchain_api_key = st.text_input("LangChain API Key (Optional)", type="password")

    st.header("⚙️ Mode")

    mode = st.radio(
        "Select Mode",
        [
            "Chat with Websites (RAG)",
            "Chat with PDFs (RAG)",
            "General Chatbot (OpenAI)",
        ],
    )

    if st.button("🚀 Initialize App"):
        if mode in ["Chat with Websites (RAG)", "Chat with PDFs (RAG)"]:
            if not groq_api_key or not hf_token:
                st.error("Groq API key & HuggingFace token are required")
            else:
                initialize_environment(
                    groq_api_key, hf_token, openai_api_key, langchain_api_key
                )
                st.session_state.app_initialized = True
                st.success("App initialized")
                st.rerun()
        else:
            if not openai_api_key:
                st.error("OpenAI API key required")
            else:
                initialize_environment(
                    groq_api_key, hf_token, openai_api_key, langchain_api_key
                )
                st.session_state.app_initialized = True
                st.success("App initialized")
                st.rerun()

# ------------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------------
if not st.session_state.app_initialized:
    st.info("👈 Configure API keys and initialize the app")
    st.stop()

embeddings = get_embeddings()

# ==============================================================
# WEBSITE RAG
# ==============================================================
if mode == "Chat with Websites (RAG)":
    st.subheader("🌐 Website RAG Chat")

    session_id = st.text_input("Session ID", value="web_session")

    website_url = st.text_input("Enter Website URL")

    if st.button("Load Website") and website_url:
        loader = UnstructuredURLLoader(
            urls=[website_url],
            headers={"User-Agent": "Mozilla/5.0"},
            ssl_verify=False,
        )
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        splits = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            splits,
            embeddings,
            collection_name=f"web_{session_id}",
        )

        st.session_state[f"vs_{session_id}"] = vectorstore
        st.success("Website loaded successfully")

    if f"vs_{session_id}" in st.session_state:
        retriever = st.session_state[f"vs_{session_id}"].as_retriever(k=4)

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
        )

        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Rephrase question using chat history"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_prompt
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Answer using provided context.\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_retriever, qa_chain)

        def get_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_chain = RunnableWithMessageHistory(
            rag_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        if question := st.chat_input("Ask about the website"):
            result = conversational_chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}},
            )
            st.chat_message("assistant").write(result["answer"])

# ==============================================================
# OPENAI CHAT
# ==============================================================
elif mode == "General Chatbot (OpenAI)":
    st.subheader("💬 OpenAI Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        st.chat_message(m["role"]).write(m["content"])

    if prompt := st.chat_input("Say something"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
        )

        response = llm.invoke(prompt)
        st.session_state.messages.append(
            {"role": "assistant", "content": response.content}
        )
        st.chat_message("assistant").write(response.content)
