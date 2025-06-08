import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings

FAISS_PATH = "faiss_index"

chat_model = ChatMistralAI(model_name="mistral-medium-latest", temperature=0.7)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_faiss_index():
    if os.path.exists(FAISS_PATH):
        return FAISS.load_local(FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return None

def create_faiss_index(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n"
    )
    split_docs = splitter.split_documents(docs)

    faiss_db = FAISS.from_documents(split_docs, embedding_model)
    faiss_db.save_local(FAISS_PATH)
    return faiss_db

def build_rag_chain(docsearch):
    prompt = ChatPromptTemplate.from_template(
        """You are a research assistant. Use the following context to answer the question.

        Context:
        {context}

        Question:
        {query}"""
    )

    rag_chain = (
        {"context": docsearch.as_retriever(), "query": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )
    return rag_chain
