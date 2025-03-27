import os
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Caricamento del modello locale con LlamaCpp
llm = OllamaLLM(model="mistral")

# Directory della knowledge base
knowledge_base_dir = 'knowledge_base'

#Questo da la possibilita di mettere nella Knowledge_base più formati dei file
# Mappatura estensioni file ai relativi loader
LOADER_MAPPING = {
    ".md": UnstructuredMarkdownLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".pdf": UnstructuredPDFLoader
}

def load_documents(directory):
    """Carica tutti i documenti da una cartella, indipendentemente dal formato."""
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        ext = os.path.splitext(filename)[-1].lower()
        
        if ext in LOADER_MAPPING:
            loader_cls = LOADER_MAPPING[ext]
            loader = loader_cls(file_path)
            documents.extend(loader.load())  # Carica il contenuto del file

    return documents

# Caricamento documenti da più formati
documents = load_documents(knowledge_base_dir)

# Suddivisione in chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

# Creazione degli embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)

# Creazione della pipeline RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Interfaccia da terminale
print("\n🔹 RAG con Mistral per il rilevamento di Security Smell 🔹")
print("Scrivi la tua domanda (oppure 'exit' per uscire).")

while True:
    query = input("\nInserisci domanda: ")
    if query.lower() in ["exit", "quit", "esci"]:
        print("Chiusura programma.")
        break

    response = qa_chain.run(query)
    print("\n🤖 Risposta:\n", response)
