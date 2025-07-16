import os
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
os.environ['FAISS_NO_AVX2'] = '1'  # Disable AVX2 to prevent compatibility issues

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
st.set_page_config(page_title="PDF Expert Pro", page_icon="📄", layout="wide")


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

# 🌐 ENHANCED CUSTOM STYLING
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

# --------------------------
# 📄 IMPROVED CONTENT PROCESSING
# --------------------------

def is_content_page(text):
    """Check if page contains actual content (not administrative info)"""
    if len(text.strip()) < 100:
        return False
        
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
    
    unwanted_count = sum(len(re.findall(pattern, text.lower())) 
                      for pattern in unwanted_patterns)
    
    return unwanted_count < 2

def clean_text(text):
    """Clean text while preserving structure and removing admin content"""
    text = re.sub(r'^.*\bpage\s*\d+\b.*$', '', text, flags=re.MULTILINE)
    
    author_patterns = [
        r'\b(by|author)\s*:\s*.*$',
        r'\b(faculty|department)\s*:\s*.*$',
        r'\b(college|university)\s*:\s*.*$',
        r'\b(lecturer|professor|instructor)\s*:\s*.*$'
    ]
    for pattern in author_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE|re.MULTILINE)
    
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()

def generate_concise_summary(text, num_sentences=5):
    """Generate clean summary with only key points"""
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'\$(.*?)\$', '', text)
    text = re.sub(r'\\begin{equation}.*?\\end{equation}', '', text, flags=re.DOTALL)
    text = re.sub(r'\(.*?\)', '', text)
    
    sentences = [
        s.strip() for s in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text) 
        if len(s.split()) > 8
        and not any(word in s.lower() for word in ['example', 'figure', 'table'])
    ]
    
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000
    )
    X = vectorizer.fit_transform(sentences)
    scores = np.array(X.sum(axis=1)).flatten()
    top_indices = scores.argsort()[-num_sentences:][::-1]
    
    summary = ' '.join([sentences[i] for i in sorted(top_indices)])
    summary = re.sub(r'\b(see|refer to|figure|table)\b.*?\.', '', summary, flags=re.IGNORECASE)
    return summary

# --------------------------
# 🧠 PDF PROCESSING CORE
# --------------------------

def get_pdf_text_and_map(pdf_docs):
    """Process PDFs with enhanced content filtering"""
    chunks, chunk_map, full_text, pdf_summaries = [], [], "", {}
    
    for pdf in pdf_docs:
        with st.spinner(f"Processing {pdf.name}..."):
            try:
                reader = PdfReader(pdf)
                pdf_text = ""
                
                for page in reader.pages:
                    text = page.extract_text()
                    if text and is_content_page(text):
                        text = clean_text(text)
                        pdf_text += text + "\n\n"
                        full_text += text + "\n\n"
                
                if not pdf_text.strip():
                    st.warning(f"No significant content found in {pdf.name}")
                    continue
                
                summary = generate_concise_summary(pdf_text)
                pdf_summaries[pdf.name] = summary
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
                )
                split_chunks = splitter.split_text(pdf_text)
                chunks.extend(split_chunks)
                
            except Exception as e:
                st.error(f"Error processing {pdf.name}: {str(e)}")
                continue
    
    return chunks, full_text, pdf_summaries

def get_vectorstore(text_chunks):
    """Create vector store with error handling"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except Exception as e:
        st.error(f"Vectorstore error: {str(e)}")
        return None

def retrieve_best_answer(user_question, vectorstore, full_text=""):
    """Enhanced answer retrieval with better matching and context understanding"""
    try:
        # Pre-process the user question
        processed_question = preprocess_question(user_question)
        
        # First try semantic search with the vector store
        if vectorstore:
            docs = vectorstore.similarity_search_with_score(processed_question, k=5)
            
            # Filter for good matches (adjust threshold as needed)
            good_matches = [doc for doc in docs if doc[1] < 0.8]
            
            if good_matches:
                context = "\n\n".join([doc[0].page_content for doc in good_matches])
                return format_answer(user_question, context)
        
        # If semantic search fails or no vectorstore, try keyword search
        keywords = extract_relevant_keywords(processed_question)
        keyword_contexts = []
        
        for keyword in keywords:
            context = find_keyword_context(keyword, full_text)
            if context:
                keyword_contexts.append((keyword, context))
        
        if keyword_contexts:
            return format_keyword_answer(user_question, keyword_contexts)
        
        # Final fallback - try direct text search for definition questions
        if is_definition_question(user_question):
            definition = find_direct_definition(user_question, full_text)
            if definition:
                return format_definition_response(user_question, definition)
        
        return suggest_alternative_approaches(user_question)
    
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}. Please try again."

def preprocess_question(question):
    """Clean and normalize the question"""
    question = question.lower().strip()
    # Remove question words and common verbs
    stop_phrases = ['what is', 'what are', 'who is', 'can you', 'please', 
                   'could you', 'would you', 'tell me about', 'explain']
    for phrase in stop_phrases:
        question = question.replace(phrase, '')
    return question.strip()

def extract_relevant_keywords(text):
    """Extract meaningful keywords with better filtering"""
    # Keep only nouns and proper nouns (simple implementation)
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    stopwords = {'the', 'and', 'or', 'but', 'this', 'that', 'these', 'those', 
                'for', 'with', 'from', 'about', 'when', 'where', 'how', 'why'}
    return [word for word in words if word not in stopwords]

def find_keyword_context(keyword, text, window=300):
    """Find context around keyword occurrences"""
    # Try different capitalization forms
    patterns = [
        rf'\b{re.escape(keyword)}\b',
        rf'\b{re.escape(keyword.capitalize())}\b',
        rf'\b{re.escape(keyword.upper())}\b'
    ]
    
    contexts = []
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in list(matches)[:3]:  # Get first 3 matches
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            context = text[start:end]
            # Highlight the keyword
            highlighted = re.sub(
                pattern, 
                r'<span style="background-color: rgba(0,188,212,0.3);font-weight:bold;">\g<0></span>', 
                context, 
                flags=re.IGNORECASE
            )
            contexts.append(highlighted)
    
    return "\n\n[...] ".join(contexts) if contexts else None

def is_definition_question(question):
    """Check if question is asking for a definition"""
    return question.lower().startswith(('what is', 'what are', 'define', 'who is'))

def find_direct_definition(question, text):
    """Try to find a direct definition in the text"""
    target = question.lower().replace('what is', '').replace('what are', '').replace('define', '').strip()
    if not target:
        return None
    
    # Look for definition patterns
    patterns = [
        rf'{re.escape(target)}\s*(is|are|was|were|refers to|means)\s*([^\.]+)\.',
        rf'([^\.]*\b{re.escape(target)}\b[^\.]*(is|are|was|were|refers to|means)[^\.]+)\.',
        rf'\b{re.escape(target.capitalize())}\b[^\.]+\.'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    
    return None

def format_answer(question, context):
    """Format a well-structured answer"""
    if is_definition_question(question):
        definition = find_direct_definition(question, context)
        if definition:
            return format_definition_response(question, definition)
    
    # Truncate if too long
    if len(context) > 1500:
        context = context[:1500] + "\n\n[Content truncated for readability]"
    
    return f"""Here's what I found related to your question:

{context}

Does this answer your question about "{question}"?"""

def format_definition_response(question, definition):
    """Format a definition response"""
    term = question.lower().replace('what is', '').replace('what are', '').replace('define', '').strip()
    return f"""The document defines '{term}' as:

{definition}

Would you like more information about this?"""

def format_keyword_answer(question, keyword_contexts):
    """Format answer when we found keyword matches"""
    response = ["Here's what I found related to your question:"]
    
    for keyword, context in keyword_contexts:
        response.append(f"\nInformation about '{keyword}':\n{context}")
    
    response.append(f"\nDoes this help answer your question about '{question}'?")
    return "\n".join(response)

def suggest_alternative_approaches(question):
    """Suggest alternatives when no answer is found"""
    suggestions = [
        "Try using different keywords from your question",
        "The information might be phrased differently in the document",
        "Check if you've uploaded the correct documents containing this information",
        "Try asking a more specific or differently worded question"
    ]
    
    return f"""I couldn't find a direct answer to "{question}". Here are some suggestions:
    
- """ + "\n- ".join(suggestions)

def show_navbar():
    """Navigation bar UI"""
    col1, col2, col3 = st.columns([6, 1.5, 1.5])
    with col1:
        st.title("📄 PDF Expert Pro")
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
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'pdf_summaries' not in st.session_state:
        st.session_state.pdf_summaries = {}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.markdown(f"### Welcome back, {st.session_state.username}!")
    
    st.subheader("📥 Upload Your Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files (you can select multiple)", 
        type=["pdf"], 
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Analyzing document content..."):
            try:
                chunks, full_text, summaries = get_pdf_text_and_map(uploaded_files)
                st.session_state.vectorstore = get_vectorstore(chunks)
                st.session_state.full_text = full_text
                st.session_state.pdf_summaries = summaries
                st.success(f"Processed {len(uploaded_files)} document(s) successfully!")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")

    if st.session_state.pdf_summaries:
        st.subheader("📄 Document Summaries")
        for filename, summary in st.session_state.pdf_summaries.items():
            with st.expander(f"📘 {filename}", expanded=False):
                st.markdown(f"<div class='summary-box'>{summary}</div>", 
                           unsafe_allow_html=True)

    if st.session_state.vectorstore:
        st.subheader("💬 Ask About Your Documents")
        question = st.text_input(
            "Ask anything about the document content:", 
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
                
                st.session_state.chat_history.append(("user", question))
                st.session_state.chat_history.append(("bot", answer))
                
                st.markdown("### Conversation History")
                for role, content in st.session_state.chat_history[-4:]:
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
                            f"<strong>PDF Expert:</strong> {content}"
                            f"</div>", 
                            unsafe_allow_html=True
                        )


def main():
    load_dotenv()
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'show_account' not in st.session_state:
        st.session_state.show_account = False
    
    show_navbar()
    
    if st.session_state.get("logged_in", False) and st.session_state.get("show_account", False):
        show_account_info()
    
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