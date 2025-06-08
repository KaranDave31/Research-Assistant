import streamlit as st
import tempfile
import os
from dotenv import load_dotenv
from chat_bot import load_faiss_index, create_faiss_index, build_rag_chain

load_dotenv()

st.set_page_config(page_title="Research Assistant", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE; font-size: 48px;'>
        RAG Based Research Assistant
    </h1>
    <p style='text-align: center; color: gray; font-size: 18px;'>
        Upload a PDF and get contextual responses about it.
    </p>
""", unsafe_allow_html=True)


with st.container():
    st.markdown("### Upload your PDF document")
    uploaded_file = st.file_uploader(
        label="Drag and drop or browse your file",
        type=["pdf"],
        label_visibility="collapsed"
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if uploaded_file is not None and st.session_state.rag_chain is None:
    st.success(f"PDF `{uploaded_file.name}` uploaded successfully!")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    with st.spinner("Loading or creating FAISS index "):
        faiss_db = load_faiss_index()
        if not faiss_db:
            faiss_db = create_faiss_index(tmp_file_path)

    os.remove(tmp_file_path)

    st.session_state.rag_chain = build_rag_chain(faiss_db)

for message in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(f"**You:** {message['user']}")
    with st.chat_message("assistant"):
        st.markdown(f"**Assistant:** {message['bot']}")

if st.session_state.rag_chain:
    user_query = st.chat_input("Ask something about the document ")

    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Thinking..."):
            response = st.session_state.rag_chain.invoke(user_query)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.chat_history.append({"user": user_query, "bot": response})
