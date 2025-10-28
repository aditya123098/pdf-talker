from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda,RunnablePassthrough,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import streamlit as st 
import os

# Prefer environment variable so tokens aren't checked into source control.
# Fallback to Streamlit secrets if env var is not set.
hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    # Clear, actionable message for the developer/operator.
    msg = (
        "Hugging Face API token not found.\n"
        "Generate a new User Access Token at https://huggingface.co/settings/tokens (scope: 'Read' or 'Read + Write' depending on use),\n"
        "then either: \n"
        "  - set it as an environment variable in your shell: `setx HUGGINGFACEHUB_API_TOKEN \"hf_...\"` (restart your terminal/app), or\n"
        "  - add it to `.streamlit/secrets.toml` as: `HUGGINGFACEHUB_API_TOKEN = \"hf_...\"`\n"
        "After that, restart your Streamlit app. If you previously had a token and see a 401 Unauthorized, rotate the token using the link above."
    )
    # show error in Streamlit and raise so failure is obvious during app startup
    try:
        st.error(msg)
    except Exception:
        # If Streamlit isn't available in this context, print the message instead.
        print(msg)
    raise RuntimeError("HUGGINGFACEHUB_API_TOKEN is missing. See application log for instructions.")

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
    # Create the HF endpoint; if the token is expired you'll typically see an HTTP 401
    # when the model is invoked. We surface that as a clearer Streamlit error below.
    try:
        llm = HuggingFaceEndpoint(
            repo_id="meta-llama/Llama-3.3-70B-Instruct",
            task="text-generation",
            huggingfacehub_api_token=hf_token,
            temperature=0.2)
        model = ChatHuggingFace(llm=llm)
    except Exception as e:
        # Provide actionable guidance on 401/unauthorized errors which commonly
        # indicate expired/rotated tokens.
        err_text = str(e)
        guidance = (
            "Error while creating HuggingFace model client. This can happen if the token is invalid or expired.\n"
            "Please regenerate a token at https://huggingface.co/settings/tokens and update the environment or `.streamlit/secrets.toml`.\n"
            "Detailed error: {}".format(err_text)
        )
        try:
            st.error(guidance)
        except Exception:
            print(guidance)
        raise

    # parallel chain to get the question and retriever parallely and then pass it to the final chain that produces the outpu

    parallel_chain= RunnableParallel(
        {'context': retriever | RunnableLambda(format_docs), 'question': RunnablePassthrough() }
    )
    
    output_chain= parallel_chain|prompt|model|parser
     


    return output_chain
