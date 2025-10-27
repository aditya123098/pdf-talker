from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda,RunnablePassthrough,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import streamlit as st 
hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

def output(retriever):
    

    #helper function to only join the page content of the retireved docs for formatting
    def format_docs(docs):
        return '\n\n'.join( doc.page_content for doc in docs)
    
    
    #parser
    parser=StrOutputParser()

    #defining prompt
    prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        You are an intelligent assistant that answers user questions using ONLY the information provided in the context below. 

        If the context does not contain enough information to answer the question, respond with:
        "I don’t know based on the given document."

        Guidelines:
        - Always stay truthful to the context. Do not add external knowledge. 
        - If multiple relevant parts exist, combine them into a clear, concise answer.
        - If the question is broad (like 'summarize'), give a structured summary based only on the context.
        - If the question cannot be answered, clearly say you don’t know.
        - Provide step-by-step reasoning when needed.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        )
    
    # defining open source hugging face model for text generation
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.3-70B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=hf_token,
        temperature=0.2)
    model = ChatHuggingFace(llm=llm)

    # parallel chain to get the question and retriever parallely and then pass it to the final chain that produces the outpu

    parallel_chain= RunnableParallel(
        {'context': retriever | RunnableLambda(format_docs), 'question': RunnablePassthrough() }
    )
    
    output_chain= parallel_chain|prompt|model|parser
     


    return output_chain
