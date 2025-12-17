import os
import logging
import validators
import streamlit as st
import requests
from dotenv import load_dotenv

# =========================
# LangChain (NEW IMPORTS)
# =========================
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# =========================
# BASIC SETUP
# =========================
load_dotenv()
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Unified RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Unified RAG Chatbot (Groq • OpenAI • RAG)")

# =========================
# SESSION STATE
# =========================
if "store" not in st.session_state:
    st.session_state.store = {}

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("🔐 API Keys")

    GROQ_API_KEY = st.text_input("Groq API Key", type="password")
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")

    MODE = st.radio(
        "Select Mode",
        ["Website RAG", "PDF RAG", "OpenAI Chat"]
    )

    if st.button("Initialize"):
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        st.success("Initialized")

# =========================
# EMBEDDINGS
# =========================
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

embeddings = get_embeddings()

# =========================
# WEBSITE RAG
# =========================
if MODE == "Website RAG":
    st.subheader("🌐 Chat with Website")

    url = st.text_input("Enter Website URL")

    if st.button("Load Website"):
        if not validators.url(url):
            st.error("Invalid URL")
        else:
            with st.spinner("Loading website..."):
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
                    splits,
                    embeddings
                )

                st.success("Website content loaded")

# =========================
# PDF RAG
# =========================
if MODE == "PDF RAG":
    st.subheader("📄 Chat with PDFs")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            docs = []
            for file in uploaded_files:
                path = f"/tmp/{file.name}"
                with open(path, "wb") as f:
                    f.write(file.read())
                loader = PyPDFLoader(path)
                docs.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = splitter.split_documents(docs)

            st.session_state.vectorstore = Chroma.from_documents(
                splits,
                embeddings
            )

            st.success("PDFs loaded")

# =========================
# CHAT SECTION (RAG)
# =========================
if MODE in ["Website RAG", "PDF RAG"] and st.session_state.vectorstore:
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
        llm,
        retriever,
        contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer using provided context.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain
    )

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    if user_input := st.chat_input("Ask something"):
        response = conversational_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": "default"}}
        )
        st.chat_message("assistant").write(response["answer"])

# =========================
# OPENAI CHAT
# =========================
if MODE == "OpenAI Chat":
    st.subheader("💬 OpenAI Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Say something"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=OPENAI_API_KEY
        )

        response = llm.invoke(prompt)
        answer = response.content

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.chat_message("assistant").write(answer)
