from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from bertopic import BERTopic
import os
from hdbscan import HDBSCAN

path = "../data/"

paragrafos = []

def convert_text(doc):
    lista = []
    for i in doc:
        lista.append(i.page_content)

    text = "\n".join(lista)
    paragrafos = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragrafos

for pdf in os.listdir(path):
    doc = PyPDFLoader(path + pdf)
    content = doc.lazy_load()
    p = convert_text(content)
    paragrafos.extend(p)

model_sentence = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")

cluster_model = HDBSCAN(
    min_cluster_size=4, 
    min_samples=2, 
    prediction_data=True
)

model_topic = BERTopic(embedding_model=model_sentence, hdbscan_model=cluster_model, calculate_probabilities=True)

topics, probs = model_topic.fit_transform(paragrafos)

model_topic.save("../modelos/bertopic_model", serialization="safetensors", save_embedding_model=False)




