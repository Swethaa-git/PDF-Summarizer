
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sqlite3
import hashlib
from datetime import datetime
import pandas as pd
from io import StringIO

# Page Config
st.set_page_config(page_title="Ask-PDF Pro", page_icon="📄", layout="wide")


# AUTHENTICATION SYSTEM

def init_db():
    conn = sqlite3.connect('auth.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  email TEXT UNIQUE,
                  password TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, email, password):
    conn = sqlite3.connect('auth.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                  (username, email, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect('auth.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0] == hash_password(password):
        return True
    return False

def signup_page():
    st.subheader("Create New Account")
    with st.form("signup_form"):
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.form_submit_button("Sign Up"):
            if password == confirm_password:
                if register_user(username, email, password):
                    st.success("Account created successfully! Please log in.")
                    st.session_state.page = "signin"
                    st.experimental_rerun()
                else:
                    st.error("Username or email already exists")
            else:
                st.error("Passwords do not match")
    
    if st.button("Back to Login"):
        st.session_state.page = "signin"
        st.experimental_rerun()

def login_page():
    st.subheader("Login to Your Account")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.form_submit_button("Login"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
    
    if st.button("Create New Account"):
        st.session_state.page = "signup"
        st.experimental_rerun()

# ENHANCED CUSTOM STYLING
st.markdown("""
    <style>
    :root {
        --primary: #00bcd4;
        --primary-dark: #0097a7;
        --background: #1a1a1a;
        --surface: #2a2a2a;
        --text-primary: #d0d0d0;
        --text-secondary: #a0a0a0;
    }
    
    body, .stApp {
        background-color: var(--background);
        color: var(--text-primary);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary);
        margin-bottom: 0.75rem;
    }
    
    .stButton > button {
        background-color: var(--primary);
        color: #111;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: var(--primary-dark) !important;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .stTextInput > div > input, 
    .stTextArea > div > textarea,
    .stFileUploader > div > div {
        background-color: var(--surface);
        color: var(--text-primary);
        border: 1px solid #444;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    .stTextInput label, 
    .stTextArea label,
    .stFileUploader label {
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    .navbar {
        background-color: #2b313e;
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background-color: var(--surface);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 1.05rem;
        box-shadow: 0 4px 6px rgba(0, 188, 212, 0.1);
        transition: transform 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 188, 212, 0.15);
    }
    
    .summary-box {
        background-color: var(--surface);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid var(--primary);
        max-height: 300px;
        overflow-y: auto;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        max-width: 80%;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .user-message {
        background-color: #2b5278;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .bot-message {
        background-color: var(--surface);
        border-bottom-left-radius: 4px;
    }
    
    .loading-spinner {
        color: var(--primary) !important;
    }
    
    .error-message {
        color: #ff6b6b;
        font-weight: bold;
    }
    
    .structured-content {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        overflow-x: auto;
    }
    
    table.dataframe {
        width: 100%;
        border-collapse: collapse;
        color: var(--text-primary);
    }
    
    table.dataframe th, table.dataframe td {
        border: 1px solid #444;
        padding: 0.5rem;
    }
    
    table.dataframe th {
        background-color: #333;
        font-weight: 600;
    }
    
    table.dataframe tr:nth-child(even) {
        background-color: #2a2a2a;
    }
    
    .stExpander > div {
        background-color: var(--surface) !important;
        border-radius: 8px !important;
        border: 1px solid #444 !important;
    }
    
    .stExpander > div > div {
        padding: 1rem !important;
    }
    
    .account-modal {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0,0,0,0.5);
        z-index: 1000;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .account-content {
        background-color: #2a2a2a;
        padding: 2rem;
        border-radius: 12px;
        width: 400px;
        max-width: 90%;
    }
    
    @media (max-width: 768px) {
        .navbar {
            flex-direction: column;
            padding: 1rem;
        }
        
        .feature-card {
            margin-bottom: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)


# IMPROVED CONTENT PROCESSING

def is_content_page(text):
    """Check if page contains actual content (not administrative info)"""
    # Skip pages with very little text
    if len(text.strip()) < 100:
        return False
        
    # Patterns that indicate administrative/unwanted content
    unwanted_patterns = [
        r'\b(college|university|institute|faculty|department)\b',
        r'\b(lecturer|professor|instructor)\b',
        r'\b(syllabus|course code|reference|text books)\b',
        r'\b(unit \d+|module \d+|chapter \d+)\b',
        r'\b(outcome|objective)\b',
        r'page\s*\d+\s*of\s*\d+',
        r'\b(this page intentionally left blank)\b',
        r'Dept of \w+, Unit -\d+ Page \d+'
    ]
    
    # Count matches of unwanted patterns
    unwanted_count = sum(len(re.findall(pattern, text.lower())) 
                      for pattern in unwanted_patterns)
    
    # Page is considered content if it has minimal unwanted patterns
    return unwanted_count < 2

def clean_text(text):
    """Clean text while preserving structure and removing admin content"""
    # Remove headers/footers with page numbers
    text = re.sub(r'^.*\bpage\s*\d+\b.*$', '', text, flags=re.MULTILINE)
    
    # Remove author/college information patterns
    author_patterns = [
        r'\b(by|author)\s*:\s*.*$',
        r'\b(faculty|department)\s*:\s*.*$',
        r'\b(college|university)\s*:\s*.*$',
        r'\b(lecturer|professor|instructor)\s*:\s*.*$'
    ]
    for pattern in author_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE|re.MULTILINE)
    
    # Normalize whitespace and preserve paragraph breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Single newlines to space
    text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    
    return text.strip()

def generate_concise_summary(text, num_sentences=5):
    """Generate clean summary with only key points"""
    # Remove technical patterns that shouldn't be in summary
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', '', text)
    text = re.sub(r'\\begin{equation}.*?\\end{equation}', '', text, flags=re.DOTALL)
    text = re.sub(r'\(.*?\)', '', text)  # Remove parentheses content
    
    # Split into sentences and filter
    sentences = [
        s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text) 
        if len(s.split()) > 8  # Only longer sentences
        and not any(word in s.lower() for word in ['example', 'figure', 'table', 'reference'])
    ]
    
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    
    # Use TF-IDF to select most important sentences
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000
    )
    X = vectorizer.fit_transform(sentences)
    scores = np.array(X.sum(axis=1)).flatten()
    top_indices = scores.argsort()[-num_sentences:][::-1]
    
    # Join the most important sentences
    summary = ' '.join([sentences[i] for i in sorted(top_indices)])
    
    # Remove any remaining technical references
    summary = re.sub(r'\b(see|refer to|figure|table)\b.*?\.', '', summary, flags=re.IGNORECASE)
    return summary

# PDF PROCESSING

def get_pdf_text_and_map(pdf_docs):
    """Process PDFs with enhanced content filtering"""
    chunks, chunk_map, full_text, pdf_summaries = [], [], "", {}
    all_structured_elements = {}

    for pdf in pdf_docs:
        with st.spinner(f"Processing {pdf.name}..."):
            try:
                reader = PdfReader(pdf)
                pdf_text = ""
                structured_elements = []
                
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and is_content_page(text):
                        text = clean_text(text)
                        pdf_text += text + "\n\n"
                        full_text += text + "\n\n"
                
                if not pdf_text.strip():
                    st.warning(f"No significant content found in {pdf.name}")
                    continue
                
                # Generate clean summary
                summary = generate_concise_summary(pdf_text)
                pdf_summaries[pdf.name] = summary
                
                # Create chunks from cleaned text
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=300,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
                )
                split_chunks = splitter.split_text(pdf_text)
                chunks.extend(split_chunks)
                chunk_map += [(pdf.name, i, chunk) for i, chunk in enumerate(split_chunks)]
                
            except Exception as e:
                st.error(f"Error processing {pdf.name}: {str(e)}")
                continue
    
    return chunks, chunk_map, full_text, pdf_summaries


# EMBEDDINGS & VECTORSTORE

def get_vectorstore(text_chunks):
    """Create vector store from text chunks"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

# IMPROVED QUESTION ANSWERING

def retrieve_best_answer(user_question, vectorstore, full_text=""):
    """Enhanced answer retrieval that provides relatable answers"""
    try:
        # Semantic search with score threshold
        docs = vectorstore.similarity_search_with_score(user_question, k=3)
        if docs and docs[0][1] < 1.0:  # Score threshold
            # Combine the most relevant chunks
            context = "\n\n".join([doc[0].page_content for doc in docs])
            
            # Generate a natural language response
            if len(context) > 1000:
                return f"Based on the document content:\n\n{context[:1000]}...\n\n[Content truncated]"
            return f"Here's what the document says about this:\n\n{context}"
        
        # Fallback to keyword matching with context
        keywords = re.findall(r'\b\w{3,}\b', user_question.lower())
        for word in keywords:
            if word in full_text.lower():
                return f"The document discusses '{word}' in several places. Here's one relevant section:\n\n{get_keyword_context(word, full_text)}"
                
        return "I couldn't find specific information about this in the documents. Try asking in a different way."
    except Exception as e:
        return f"Sorry, I encountered an error processing your question. Please try again."

def get_keyword_context(keyword, text, window=300):
    """Get context around a keyword occurrence"""
    matches = re.finditer(rf'\b{re.escape(keyword)}\b', text, re.IGNORECASE)
    contexts = []
    for match in list(matches)[:2]:  # Get first 2 occurrences
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        context = text[start:end]
        # Highlight the keyword
        context = re.sub(
            rf'\b({re.escape(keyword)})\b', 
            r'<span style="background-color: rgba(0, 188, 212, 0.3); font-weight: bold;">\1</span>', 
            context, 
            flags=re.IGNORECASE
        )
        contexts.append(f"...{context}...")
    return "\n\n".join(contexts)

# USER INTERFACE

def show_navbar():
    """Navigation bar UI"""
    col1, col2, col3 = st.columns([6, 1.5, 1.5])
    with col1:
        st.title("📄 Ask-PDF Pro")
    with col2:
        if st.session_state.get("logged_in", False):
            if st.button("👤 Account"):
                st.session_state.show_account = True
        elif st.session_state.page != "signin":
            if st.button("🔑 Sign In"):
                st.session_state.page = "signin"
                st.experimental_rerun()
    with col3:
        if not st.session_state.get("logged_in", False) and st.session_state.page != "signup":
            if st.button("✍️ Sign Up"):
                st.session_state.page = "signup"
                st.experimental_rerun()
        elif st.session_state.get("logged_in", False):
            if st.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.session_state.page = "home"
                st.session_state.clear()
                st.experimental_rerun()

def show_account_info():
    """Account information popup"""
    if st.session_state.get("show_account", False):
        st.markdown("""
            <div class='account-modal'>
                <div class='account-content'>
        """, unsafe_allow_html=True)
        
        st.markdown("### Account Information")
        st.markdown(f"**Username:** {st.session_state.username}")
        
        if st.button("Close"):
            st.session_state.show_account = False
            st.experimental_rerun()
        
        st.markdown("</div></div>", unsafe_allow_html=True)

def show_homepage():
    """Homepage for visitors"""
    st.markdown("### Your Document Understanding Assistant")
    st.markdown("#### ✨ Key Features")
    
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""<div class='feature-card'>📥<br><strong>PDF Processing</strong><br>Upload and process any PDFs</div>""", 
                   unsafe_allow_html=True)
    with cols[1]:
        st.markdown("""<div class='feature-card'>🧠<br><strong>Key Point Extraction</strong><br>Get clear document summaries</div>""", 
                   unsafe_allow_html=True)
    with cols[2]:
        st.markdown("""<div class='feature-card'>🤖<br><strong>Natural Q&A</strong><br>Ask questions in plain language</div>""", 
                   unsafe_allow_html=True)

def show_logged_in_interface():
    """Main interface for logged-in users"""
    # Initialize session state
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'pdf_summaries' not in st.session_state:
        st.session_state.pdf_summaries = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Welcome message
    st.markdown(f"### Welcome back, {st.session_state.username}!")
    
    # PDF Upload Section
    st.subheader("📥 Upload Your Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files to analyze", 
        type=["pdf"], 
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Analyzing document content..."):
            try:
                chunks, _, full_text, summaries = get_pdf_text_and_map(uploaded_files)
                st.session_state.vectorstore = get_vectorstore(chunks)
                st.session_state.full_text = full_text
                st.session_state.pdf_summaries = summaries
                st.success(f"Processed {len(uploaded_files)} document(s) successfully!")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

    # Display Summaries
    if st.session_state.pdf_summaries:
        st.subheader("📄 Document Overview")
        for filename, summary in st.session_state.pdf_summaries.items():
            with st.expander(f"📘 {filename}", expanded=False):
                st.markdown(f"<div class='summary-box'>{summary}</div>", 
                           unsafe_allow_html=True)

    # Q&A Section
    if st.session_state.vectorstore:
        st.subheader("💬 Ask About Your Documents")
        question = st.text_input(
            "Type your question about the document content:", 
            key="question_input",
            placeholder="e.g. What are the main points about...?"
        )
        
        if question:
            with st.spinner("Finding relevant information..."):
                answer = retrieve_best_answer(
                    question,
                    st.session_state.vectorstore,
                    st.session_state.full_text
                )
                
                # Update chat history
                st.session_state.chat_history.append(("user", question))
                st.session_state.chat_history.append(("bot", answer))
                
                # Display conversation
                st.markdown("### Conversation History")
                for role, content in st.session_state.chat_history[-4:]:  # Show last 2 exchanges
                    if role == "user":
                        st.markdown(
                            f"<div class='chat-message user-message'>"
                            f"<strong>You:</strong> {content}"
                            f"</div>", 
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='chat-message bot-message'>"
                            f"<strong>Ask-PDF Pro:</strong> {content}"
                            f"</div>", 
                            unsafe_allow_html=True
                        )


# MAIN APPLICATION


def main():
    """Main application function"""
    # Load environment variables
    load_dotenv()
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'show_account' not in st.session_state:
        st.session_state.show_account = False
    
    # Show navigation
    show_navbar()
    
    # Show account info if needed
    if st.session_state.get("logged_in", False) and st.session_state.get("show_account", False):
        show_account_info()
    
    # Route to appropriate page
    if st.session_state.logged_in:
        show_logged_in_interface()
    else:
        if st.session_state.page == "signin":
            login_page()
        elif st.session_state.page == "signup":
            signup_page()
        else:
            show_homepage()

if __name__ == "__main__":
    main()