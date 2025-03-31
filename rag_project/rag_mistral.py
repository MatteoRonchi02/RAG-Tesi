import os
import json
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document  # Importa la classe Document
import warnings

# Ignora i FutureWarning, problema non bloccante ma da risolvere
warnings.filterwarnings("ignore", category=FutureWarning)


# Recupera il token in modo sicuro dall'ambiente
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError("Il token HuggingFace deve essere impostato nella variabile HUGGINGFACEHUB_API_TOKEN")


# Caricamento del modello da Hugging Face.
llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    task="text-generation",
    temperature=0.3, #creatività del modello, va da 0 a 1
    model_kwargs={"max_length": 512},
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

# Directory della knowledge base
knowledge_base_dir = 'knowledge_base'

#Questo da la possibilita di mettere nella Knowledge_base più formati dei file
# Mappatura estensioni file ai relativi loader
LOADER_MAPPING = {
    ".md": UnstructuredMarkdownLoader,
    ".txt": TextLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".pdf": UnstructuredPDFLoader,
    ".json": None
}

def load_json_documents(file_path):
    """Carica e processa un documento JSON."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Crea un'unica stringa concatenando tutti i campi rilevanti
    content = (
        f"Name Small: {data['name_small']}\n"
        f"Definition: {data['definition']}\n"
        f"How to Find: {' '.join(data['how_find'])}\n"
        f"Example: {data['example']}\n"
    )
    return Document(page_content=content)

def load_documents(directory):
    """Carica tutti i documenti da una cartella, indipendentemente dal formato."""
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        ext = os.path.splitext(filename)[-1].lower()
        
        if ext in LOADER_MAPPING:
            if ext == ".json":
                document_content = load_json_documents(file_path)
                documents.append(document_content)
            else:
                loader_cls = LOADER_MAPPING[ext]
                loader = loader_cls(file_path)
                documents.extend(loader.load())  # Carica il contenuto del file

    return documents

# Caricamento documenti da più formati
documents = load_documents(knowledge_base_dir)

# Suddivisione in chunk
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size= 10000, #lunghezza di caratteri del chunk
    chunk_overlap = 0 # caratteri ripetuti dal chunk precedente
)

docs = text_splitter.split_documents(documents)

# Creazione degli embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)


# Creazione del template del prompt
# È un prompt di prova, andrà adattato alle nostre esigienze 
prompt_template = PromptTemplate(
    input_variables=["query", "context"],
    template="Answer the following question using the context provided, in case the content does not help you answer by writing \"I don't have the necessary information to answer\":\n"
             "Question: {query}"
             "Context: {context}"
)

# Genera il prompt formattato
def generate_prompt(query, relevant_doc):
    # Recupera il contenuto dei documenti rilevanti come contesto, per adesso passo solo il primo chunk più simile
    #context = "\n".join([doc.page_content for doc in relevant_docs])
    context = relevant_doc.page_content if relevant_doc else "No context found."
    return prompt_template.format(query=query, context=context)

# Creazione della pipeline RAG
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Interfaccia da terminale
print("\n RAG with Tiiuae for Security Smell detection ")
print(" Write your question (or ‘exit’ to exit).")
while True:
    query = input("\nInsert question: ")
    if query.lower() in ["exit", "quit", "esci"]:
        print("Exit program.")
        break
    
    # Recupera i documenti più rilevanti con i punteggi di similarità
    relevant_docs = retriever.invoke(query)  

    # Prende solo il primo documento rilevante come context
    if relevant_docs:  # Verifica che ci siano documenti rilevanti
        first_relevant_doc = relevant_docs[0]
    else:
        first_relevant_doc = ""  # Nel caso in cui non ci siano documenti rilevanti
    
    # Mostra il primo chunk più simile
    print("\n First most similar chunk found:")
    if relevant_docs:
        snippet = relevant_docs[0].page_content.strip().replace("\n", " ")
        print(f"--- Chunk 1 ---\n{snippet}\n")
    else:
        print("No relevant documents found.\n")

    # Genera la risposta finale con il QA chain
    prompt = generate_prompt(query, first_relevant_doc)

    # Mostra il prompt in output, parte di codice di prova (poi da cancellare)
    #print("\n🔍 Prompt generato:")
    #print(prompt)

    response = qa_chain.invoke({"query": query, "context": prompt}) 
    print("\n Answer:\n", response['result'])
