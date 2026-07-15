from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import ollama
#from langchain_ollama import OllamaEmbeddings

persist_directory = "db/chroma_db"

#embedding_model = OllamaEmbeddings(model="nomic-embed-text")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model,collection_metadata={"hnsw:space":"cosine"})

query = "where is Nvidia's Headquater loacated? and who is the ceo?"

retriever = db.as_retriever(search_kwargs = {"k":3})

revelant_docs = retriever.invoke(query)
# print(f"User Query:{query}") 

# print("----Context-----")

# for i ,docs in enumerate(revelant_docs,1):
#     print(f"Document{i}:\n{docs.page_content}\n")

client = ollama.Client()
model = "qwen2.5-32k"

prompt = f"""Based on the following documents answer the question{query}

documents:
{chr(10).join([f"-{doc.page_content}"for doc in revelant_docs])}

please provide a clear and helpful answer using only the infromantion from these documents.If you can't find answer just say IDK
"""

response = client.generate(model=model, prompt=prompt)
print("Response from Ollama:")
print(response.response)