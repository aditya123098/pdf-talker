from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_docs(docs):
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200)
    chunks=splitter.split_documents(docs)

    return chunks