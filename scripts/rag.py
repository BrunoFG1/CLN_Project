from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


llm = OllamaLLM(model="mistral", temperature=0.0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(persist_directory="../db/", embedding_function=embeddings)
retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

personality = """Tu és um robô rígido que APENAS responde com base no contexto fornecido.

INSTRUÇÕES CRUCIAIS:
1. Responde à pergunta usando ÚNICA e EXCLUSIVAMENTE os pedaços de contexto abaixo.
2. Se a resposta não estiver explicitamente escrita no contexto, diz OBRIGATORIAMENTE: "Não encontrei essa informação no manual."
3. NÃO uses o teu conhecimento prévio do mundo para inventar comandos ou explicações que não estejam no texto abaixo.

Contexto:
{context}

Pergunta: {question}

Resposta:"""

prompt = ChatPromptTemplate.from_template(personality)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
while(True):

    a = input("What do you need my help for?\n")

    if a == "sair":
        break

    response = rag_chain.invoke(a)

    print(response)