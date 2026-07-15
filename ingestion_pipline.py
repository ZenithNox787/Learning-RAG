import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def load_documents(docs_path = "docs"):
    print(f"Loading documents form {docs_path}")

    #checking if the directory exist or not
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The Directory doesn't exist")
    
    #Loading the docs
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader
    )

    documents = loader.load()
    #"documents" will be a list of "Document" object of Langchain , each obj will have metadata and page_content , there are many other types of loaders hence this format makes a standard for all of them 

    if len(documents) == 0 :
        raise FileExistsError(f"File DNE or are empty")
    
    for i, docs  in enumerate(documents):
        print(f"\nDocument:{i+1}")
        print(f" Source:{docs.metadata['source']}")
        print(f" Content Length : {len(docs.page_content)}")
        print(f" Content Preview: {docs.page_content[:100]}")
        print(f" Metadata: {docs.metadata}")
    
    return documents

def split_documents(documents,chunk_size = 1000,chunk_overlap = 0):
    print(f"Splitting the docs into small chunks")

    text_spliter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
    )
    

    chunks = text_spliter.split_documents(documents)
    
    if chunks:

        for i ,chunk in enumerate(chunks[:5]):
            print(f"\n---Chunk {i+1}")
            print(f"Source:{chunk.metadata['source']}")
            print(f"Length:{len(chunk.page_content)} Character")
            print(f"Content \n{chunk.page_content}")
            print("-"*50)
        
        if len(chunks) > 5:
            print(f"\n....and {len(chunks)-5} and more chunks")
    
    return chunks

def create_vector_storage(chunks,presist_directory = "db/chroma_db"):
    print("Creating embeddings and storing in ChromaDB")

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Creating Vector Storage")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=presist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    print(f"Vreated Vector DB")
    print(f"Vector stored  to {presist_directory}")
    return vector_store


def main():
    print("Main Function")
    #Loading Documents
    documents = load_documents(docs_path="docs")
    #Text Chunking
    chunks = split_documents(documents)
    #Vector Storage
    vectorstore = create_vector_storage(chunks)


if __name__ == "__main__":
    main()