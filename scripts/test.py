from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="mistral")

TEXT = "How are you today, can you be my virtual assistant?"
response = llm.invoke(TEXT)

print(response)