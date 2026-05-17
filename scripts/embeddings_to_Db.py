import read_and_chucking_pdf
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

chunks = read_and_chucking_pdf.chunks

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

