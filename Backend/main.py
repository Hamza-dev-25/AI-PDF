from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from Backend.Database import create_vectorstore

load_dotenv()

llm=ChatMistralAI(model='mistral-medium-latest')

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an AI PDF Study Assistant.

Answer the user questions using only the provided context from the PDF.

If the answer cannot be found in the provided context, say:
"I couldn't find the answer in the provided PDF."

Do not use your own knowledge or make up information.

Context:
{context}
"""),
   
    ("human",  "{question}" )
])

embedding_model=HuggingFaceEmbeddings()

vectore_store=Chroma(
    embedding_function=embedding_model,
    persist_directory='chromadb'
)

retriever=vectore_store.as_retriever(
    search_type='mmr',
     search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)


print("\n Welcome to the chatbot")
print('Type exit to quit.')

def ask_question(question,retriever):
    result=retriever.invoke(question)
    
    context = "\n\n".join(doc.page_content for doc in result)
    
    final_promt=prompt.invoke({"context":context,"question":question})
    response=llm.invoke(final_promt)
    return response.content,result