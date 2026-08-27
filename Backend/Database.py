from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def create_vectorstore(pdf_path):
    data=PyPDFLoader(pdf_path)

    docs=data.load()


    splitter=RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    chunks=splitter.split_documents(docs)

    embedding_model=HuggingFaceEmbeddings()

    vectore_store=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory='chromadb'
    )
    
    return vectore_store