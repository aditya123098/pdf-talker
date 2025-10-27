from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


encoder = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'} 
)

def create_vector_store(chunks):
    vector_store=FAISS.from_documents(documents=chunks,embedding=encoder)
    retriever= vector_store.as_retriever(search_type='mmr',search_kwargs={"k": 10})
    return retriever
