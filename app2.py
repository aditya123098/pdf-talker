import streamlit as st
from utils import loader, chunker, vectorstore, final_chain
import tempfile
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="PDF Intelligence Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .upload-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 2rem;
        margin-bottom: 2rem;
    }
    .response-box {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        border-left: 5px solid #00acc1;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .history-item {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #667eea;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0

# Sidebar
with st.sidebar:
    st.markdown("### 📚 PDF Intelligence Assistant")
    st.markdown("---")
    
    # Statistics
    st.markdown("### 📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", st.session_state.total_queries)
    with col2:
        st.metric("Chat History", len(st.session_state.chat_history))
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    show_history = st.checkbox("Show Chat History", value=True)
    clear_history = st.button("🗑️ Clear History")
    
    if clear_history:
        st.session_state.chat_history = []
        st.session_state.total_queries = 0
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.info("""
    - Upload a PDF document
    - Ask specific questions
    - Review chat history
    - Get instant answers
    """)
    
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("[Documentation](#) | [Support](#) | [About](#)")

# Main content
st.markdown("<h1 style='text-align: center; color: white;'>📚 PDF Intelligence Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white; font-size: 1.2rem;'>Upload your PDF and get instant answers to your questions</p>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Upload section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    st.markdown("### 📄 Upload Your Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to analyze",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        file_details = {
            "Filename": uploaded_file.name,
            "FileSize": f"{uploaded_file.size / 1024:.2f} KB",
            "FileType": uploaded_file.type
        }
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"**📄 Name:** {file_details['Filename']}")
        with col_b:
            st.markdown(f"**💾 Size:** {file_details['FileSize']}")
        with col_c:
            st.markdown(f"**📋 Type:** PDF")
        
        # Process PDF button
        if st.button("🚀 Process PDF", use_container_width=True):
            with st.spinner("🔄 Processing your PDF..."):
                progress_bar = st.progress(0)
                
                # Save temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name
                progress_bar.progress(25)
                
                # Load and process
                docs = loader.load_docs(tmp_file_path)
                progress_bar.progress(50)
                
                chunks = chunker.chunk_docs(docs=docs)
                progress_bar.progress(75)
                
                st.session_state.retriever = vectorstore.create_vector_store(chunks)
                progress_bar.progress(100)
                
                st.session_state.pdf_processed = True
                st.session_state.current_pdf = uploaded_file.name
                
                st.success(f"✅ PDF '{uploaded_file.name}' processed successfully!")
                time.sleep(1)
                st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Status")
    if st.session_state.pdf_processed:
        st.success("✅ PDF Ready")
        st.markdown(f"**Current PDF:**  \n{st.session_state.current_pdf}")
    else:
        st.warning("⏳ No PDF loaded")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Chat interface
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
st.markdown("### 💬 Ask Your Question")

query = st.text_area(
    "Type your question here...",
    height=100,
    placeholder="e.g., What is the main topic of this document?",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    ask_button = st.button("🔍 Get Answer", use_container_width=True)
with col2:
    example_button = st.button("💡 Example", use_container_width=True)

if example_button:
    st.info("📝 Example questions:\n- Summarize the main points\n- What methodology was used?\n- List the key findings")

if ask_button:
    if not st.session_state.pdf_processed:
        st.error("⚠️ Please upload and process a PDF first!")
    elif not query.strip():
        st.warning("⚠️ Please enter a question!")
    else:
        with st.spinner("🤔 Thinking..."):
            try:
                response_chain = final_chain.output(st.session_state.retriever)
                response = response_chain.invoke(query)
                
                # Display response
                st.markdown("<div class='response-box'>", unsafe_allow_html=True)
                st.markdown("### 🎯 Answer")
                st.markdown(response)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Update history
                st.session_state.chat_history.append({
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'question': query,
                    'answer': response
                })
                st.session_state.total_queries += 1
                
                st.success("✅ Answer generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.markdown("</div>", unsafe_allow_html=True)

# Chat history
if show_history and st.session_state.chat_history:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    st.markdown("### 📜 Chat History")
    
    for idx, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
        with st.expander(f"🕐 {chat['timestamp']} - Q: {chat['question'][:50]}..."):
            st.markdown(f"**Question:** {chat['question']}")
            st.markdown(f"**Answer:** {chat['answer']}")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Made with ❤️ using Streamlit & LangChain</p>", unsafe_allow_html=True)