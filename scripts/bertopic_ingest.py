from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from bertopic import BERTopic
import os
from hdbscan import HDBSCAN
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

path = "../data/"

paragrafos = []

def convert_text(doc):
    blocks = [page.page_content.strip() for page in doc if page.page_content.strip()]
    return blocks

for pdf in os.listdir(path):
    if pdf.endswith(".pdf"):
        doc = PyPDFLoader(path + pdf)
        content = doc.load()
        p = convert_text(content)
        paragrafos.extend(p)
        print(f"li o pdf {pdf} e tem {len(content)}")

model_sentence = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

cluster_model = HDBSCAN(
    min_cluster_size=4, 
    min_samples=2, 
    prediction_data=True
)

model_topic = BERTopic(embedding_model=model_sentence, hdbscan_model=cluster_model, calculate_probabilities=True)

topics, probs = model_topic.fit_transform(paragrafos)

updated_topics = model_topic.reduce_outliers(
    paragrafos, 
    topics, 
    strategy="probabilities", 
    probabilities=probs
)
model_topic.update_topics(paragrafos, topics=updated_topics)

topics = updated_topics

model_topic.save("../modelos/bertopic_model", serialization="safetensors", save_embedding_model=False)

documents_chroma = [Document(page_content=texto) for texto in paragrafos]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(documents_chroma, embeddings, persist_directory="../db")

