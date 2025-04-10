import os
import re
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import warnings
from deep_translator import GoogleTranslator

# Ignora i FutureWarning, problema non bloccante ma da risolvere
warnings.filterwarnings("ignore", category=FutureWarning)


# Recupera il token in modo sicuro dall'ambiente
GEMINI_API_KEY_PRIV = os.environ.get("GEMINI_API_KEY_PRIV")
if not GEMINI_API_KEY_PRIV:
    raise ValueError("The HuggingFace token must be set in the variable GEMINI_API_KEY_PRIV.")


# Caricamento del modello da Hugging Face.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",   
    google_api_key=GEMINI_API_KEY_PRIV,
    temperature=0.1, # È il grado di casualità nella generazione del testo
    max_tokens=2048
)

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

def split_document_by_sections(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    sections = []
    current_section = ""
    # Divide per righe; una linea che inizia con '#' viene considerata l'inizio di una nuova sezione
    for line in text.splitlines():
        if line.strip().startswith("#"):
            if current_section:
                sections.append(current_section)
            current_section = line + "\n"
        else:
            current_section += line + "\n"
    if current_section:
        sections.append(current_section)
    
    documents = []
    for section in sections:
        # Estrae l'intestazione, se presente
        header_match = re.search(r'^#\s*(.+)', section)
        header = header_match.group(1).strip() if header_match else "No Header"
        # Crea un documento con metadati che includono il percorso del file e l'header
        from langchain.docstore.document import Document  # Import locale per Document
        documents.append(Document(page_content=section, metadata={"source": file_path, "header": header}))
    return documents


def load_documents(directory):
    """Carica tutti i documenti da una cartella, indipendentemente dal formato."""
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        ext = os.path.splitext(filename)[-1].lower()
        
        
        if ext in LOADER_MAPPING:
            loader_cls = LOADER_MAPPING[ext]
            # Se il file è di tipo testo o markdown, applichiamo la segmentazione per sezioni
            if ext in [".txt", ".md"]:
                try:
                    docs = split_document_by_sections(file_path)
                except Exception as e:
                    # Se la segmentazione fallisce, usa il loader standard
                    loader = loader_cls(file_path)
                    docs = loader.load()
            else:
                loader = loader_cls(file_path)
                docs = loader.load()
            documents.extend(docs)

    return documents

# Caricamento documenti da più formati
documents = load_documents(knowledge_base_dir) 

# Creazione degli embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_documents(documents, embeddings)


# Creazione del template del prompt
# È un prompt di prova, andrà adattato alle nostre esigienze 
prompt_template = PromptTemplate(
    input_variables=["query", "context"],
    template=
            """Instructions:
            1. The Question may include either source code excerpts or a file path pointing to a microservices application.
            2. Analyze the Question carefully and use any provided file path to retrieve and review the relevant source code.
            3. Combine the information from the Question, the Context, and your internal expertise as a TYPE smell expert in microservices applications to generate a detailed and comprehensive response.
            4. Your answer should include a thorough analysis of potential TYPE smells along with practical insights and recommendations.
            5. If the provided Context and file content do not contain sufficient information to deliver an accurate answer, please respond with: "I don't have the necessary information to answer".
                
            """
            "Question: {query}\n"
            "Context: {context}\n"
            "Answer must have the same language of question.\n"
            "Answer:"
) #TYPE = scrivi il tipo di smell che ti interessa

# Genera il prompt formattato
def generate_prompt(query, relevant_context):
    # Recupera il contenuto dei documenti rilevanti come contesto, per adesso passo solo il primo chunk più simile
    #context = "\n".join([doc.page_content for doc in relevant_docs])
    #context = relevant_doc.page_content if relevant_doc else "No context found."
    #return prompt_template.format(query=query, context=context)
    return prompt_template.format(query=query, context=relevant_context)

def aggregate_full_context(source_file):
    """ Dato il percorso di un file, carica e restituisce il contenuto completo."""
    with open(source_file, "r", encoding="utf-8") as f:
        return f.read()

# Creazione della pipeline RAG
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

# Interfaccia da terminale
print("\n RAG with Gemini for Security Smell detection , Language English")
print(" Write your question (or ‘exit’ to exit).")
while True:
    query = input("\nInsert question: ")
    if query.lower() in ["exit", "quit", "esci"]:
        print("Exit program.")
        break
    
    # Traduzione da italiano a inglese
    #query_translated = GoogleTranslator(source='auto', target='en').translate(query)
    #print(f"\nTranslated query: {query_translated}")
    results = vectorstore.similarity_search_with_score(query, k=5) # kk = umero di chunk da recuperare
    valid_results = []

    if results:
        # Filtra i risultati per mostrare solo quelli con una percentuale di similarità positiva
        for i, (doc, score) in enumerate(results, start=1):
            similarity_pct = max(0, 100 - (score * 100))
            if similarity_pct > 0:
                valid_results.append((doc, similarity_pct))

        if valid_results:
            print("\nChunk found with similarity percentage:")
            for i, (doc, similarity_pct) in enumerate(valid_results, start=1):
                
                snippet = doc.page_content.strip().replace("\n", " ")
                print(f"--- Chunk {i} ---")
                print(f"Similarity: {similarity_pct:.1f}%")
                print(f"Context: {snippet}\n")
            # Usa il primo chunk per identificare il file di origine
            best_doc = results[0][0]
            source_file = best_doc.metadata.get("source")
            print(f"Aggregating full context from file: {source_file}\n")
            full_context = aggregate_full_context(source_file)
        else:
            full_context = None
            print("No relevant documents found.\n")

    # Genera la risposta finale con il QA chain
    prompt = generate_prompt(query, full_context)

    # Mostra il prompt in output, parte di codice di prova (poi da cancellare)
    print("\n Prompt generato:")
    print(prompt)

    response = qa_chain.invoke({"query": query, "context": prompt}) 
    print("\n Answer:\n", response['result'])
