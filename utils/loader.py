from langchain_community.document_loaders import PyPDFLoader


def load_docs(pdf):
    loader=PyPDFLoader(pdf)
    docs= loader.load()
    return docs
