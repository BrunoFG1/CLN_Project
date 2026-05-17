from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

pdf = PyPDFLoader("../data/Linux-Tutorial.pdf")

content = pdf.lazy_load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function=len,   
    add_start_index=True,  
)

chunks = text_splitter.split_documents(content)


