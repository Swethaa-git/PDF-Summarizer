import streamlit as st
from htmltemplate import bot_template, user_template

def chat_pdf_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>📄 Chat with your PDF</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)

        question = st.text_input("Ask something about the PDF", placeholder="Type your question here...")
        summarize = st.button("Summarize Document")

        if question:
            response = f"Bot's answer to: {question}"
            st.markdown(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
            st.markdown(bot_template.replace("{{MSG}}", response), unsafe_allow_html=True)

        if summarize:
            summary = "This is a summary of the uploaded PDF."
            st.markdown(bot_template.replace("{{MSG}}", summary), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
