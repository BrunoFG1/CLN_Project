import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -------------------
# CONFIG
# -------------------

PDF_DIR = "../data/"
DB_DIR = "../db/"

os.makedirs(PDF_DIR, exist_ok=True)

st.set_page_config(
    page_title="RAG App",
    page_icon="🤖",
    layout="wide"
)

# -------------------
# DB + EMBEDDINGS
# -------------------

@st.cache_resource
def load_db():

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        encode_kwargs={"batch_size": 8}
    )

    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    return db, embeddings


db, embeddings = load_db()

# -------------------
# INCREMENTAL INGESTION (MESMO ESTILO TEU)
# -------------------

def ingest_pdf(pdf_path, db):

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # MESMO MÉTODO QUE TU USAS
    texts = [
        page.page_content.strip()
        for page in pages
        if page.page_content.strip()
    ]

    docs = [
        Document(page_content=t)
        for t in texts
    ]

    db.add_documents(docs)

# -------------------
# SIDEBAR
# -------------------

st.sidebar.title("📁 PDFs")

pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

st.sidebar.write("### Documentos atuais:")
for p in pdfs:
    st.sidebar.write("📄", p)

uploaded_file = st.sidebar.file_uploader(
    "Adicionar novo PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    save_path = os.path.join(PDF_DIR, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ingest_pdf(save_path, db)

    st.sidebar.success("PDF adicionado e indexado!")

# -------------------
# LLM + RAG
# -------------------

llm = OllamaLLM(
    model="mistral",
    temperature=0.0
)

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10}
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

prompt = ChatPromptTemplate.from_template("""
Tu és um robô rígido que APENAS responde com base no contexto fornecido.

    INSTRUÇÕES CRUCIAIS:
    1. Responde à pergunta usando ÚNICA e EXCLUSIVAMENTE os pedaços de contexto abaixo.
    2. Se a resposta não estiver explicitamente escrita no contexto, diz OBRIGATORIAMENTE: "Não encontrei essa informação no manual."
    3. NÃO uses o teu conhecimento prévio do mundo para inventar comandos ou explicações que não estejam no texto abaixo.

    Contexto:
    {context}

    Pergunta: {question}

    Resposta:
""")

rag_chain = (
    {"context": retriever | format_docs,
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -------------------
# CHAT UI
# -------------------

st.title("🤖 Chat com RAG (Incremental)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Faz a tua pergunta...")

if question:

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("A procurar nos documentos..."):
            response = rag_chain.invoke(question)

        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )