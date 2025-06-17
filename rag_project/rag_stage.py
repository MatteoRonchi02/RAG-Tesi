import os
import warnings
import json
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader,
    TextLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from google.cloud import aiplatform
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from tqdm import tqdm

#Cambiare il tipo di smell da analizzare
TYPE_SMELL = os.getenv("TYPE_SMELL", "Architectural")   

load_dotenv()
aiplatform.init(
    project=os.getenv("GPC_PROJECT_ID"),
    location= "europe-west1"
)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Ignora i FutureWarning, problema non bloccante ma da risolvere
warnings.filterwarnings("ignore", category=FutureWarning)

# Caricamento del modello LLM
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro", 
        temperature=0.1,
        max_tokens=8192 # Token massimi per la *risposta* generata
    )
    print(f"-----------Model LLM '{llm.model}' loaded successfully-----------")
    print("=============================================================================")
except Exception as e:
    raise ValueError(f"Error loading LLM model: {e}. Check the model name and API key.")

# Directory della knowledge base principale
knowledge_base_dir = 'knowledge_base'

# Mappatura estensioni file ai relativi loader
LOADER_MAPPING = {
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".pdf": (UnstructuredPDFLoader, {}),
    ".java": (TextLoader, {"encoding": "utf-8"}),
    ".js": (TextLoader, {"encoding": "utf-8"}),
    ".vue": (TextLoader, {"encoding": "utf-8"}),
    ".html": (TextLoader, {"encoding": "utf-8"}),
    "dockerfile": (TextLoader, {"encoding": "utf-8"})
}

# Carica i dati della KB dal formato json
def load_smell_data(smell_name: str, kb_directory: str) -> dict | None:
    file_name = f"{smell_name.replace(' ', '_').lower()}.json"
    file_path = os.path.join(kb_directory, file_name)
    print(f"\nLoading smell definition from: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print("Smell definition loaded successfully.")
            return data
    except FileNotFoundError:
        print(f"ERROR: Knowledge base file not found for '{smell_name}'. Make sure '{file_path}' exists.")
        return None
    except json.JSONDecodeError:
        print(f"ERROR: The file '{file_path}' is not a valid JSON.")
        return None

#Carica tutti i documenti (interi) da una directory e sottocartelle.
def load_folder_path_documents(directory: str) -> list[Document]:
    all_documents = []
    print(f"Uploading documents from the chosen {directory} path")
    for root, _, files in os.walk(directory):
        for filename in files:
            # Controllo per ignorare le estensioni non mappate
            ext = os.path.splitext(filename)[-1].lower() if os.path.splitext(filename)[-1] else filename.lower()
            if ext in LOADER_MAPPING:
                file_path = os.path.join(root, filename)
                loader_class, loader_kwargs = LOADER_MAPPING[ext]
                print(f"Upload: {file_path} (type: {ext})")
                try:
                    loader = loader_class(file_path, **loader_kwargs)
                    docs = loader.load()
                    if docs:
                        # Aggiungiamo il percorso completo come 'source' per chiarezza
                        for doc in docs:
                            doc.metadata["source"] = file_path 
                        all_documents.extend(docs)
                except Exception as e:
                    print(f"Error while loading {file_path}: {e}")

    print(f"Total documents uploaded by {directory}: {len(all_documents)}\n")
    return all_documents

# Server per fare il chunk del codice in base al suo linguaggio
def get_code_chunks(code_documents: list[Document]) -> list[Document]:
    print("Splitting source code into manageable chunks...")
    all_chunks = []
    # Mappatura delle estensioni al tipo di linguaggio per lo splitter
    language_map = {
        ".java": Language.JAVA,
        ".js": Language.JS,
        ".html": Language.HTML
    }
    
    # Splitter di default per file non mappati (es. Dockerfile, .vue, .txt)
    default_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for doc in tqdm(code_documents, desc="Splitting source code"):
        ext = os.path.splitext(doc.metadata["source"])[-1].lower()
        language = language_map.get(ext)
        
        if language:
            splitter = RecursiveCharacterTextSplitter.from_language(language=language, chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents([doc])
        else:
             # Usa lo splitter di default per Dockerfile, Vue, etc.
            chunks = default_splitter.split_documents([doc])
        all_chunks.extend(chunks)

    print(f"Source code split into {len(all_chunks)} chunks.")
    return all_chunks

print("--------------------Initialization embeddings in progress--------------------")
try:
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("Embeddings model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error when creating embeddings model: {e}")

print("=============================================================================")


# Nuovo prompt, si può migliorare
prompt_template_str = """Instructions:
1. You are an expert {TYPE_SMELL} auditor. Your task is to analyze specific code snippets for a given {TYPE_SMELL} smell.
2. The 'Smell Definition' provides the official description and remediation strategies for the {TYPE_SMELL} vulnerability.
3. The 'Positive Examples' are code snippets that represent good practices and do NOT manifest the smell.
4. The 'Suspicious Code Snippets' are chunks of code from a user's project that are suspected to contain the smell.
5. Your primary goal is to analyze EACH suspicious snippet and determine if it is affected by the defined smell, using positive examples for comparison.
6. Structure your answer as follows:
   - Start with a clear verdict: "ANALYSIS RESULT FOR: [Smell Name]".
   - For each analyzed file path, create a section.
   - Under each file path, list the snippets that ARE VULNERABLE.
   - For each vulnerable snippet, provide:
     a. The line of code or block that contains the smell.
     b. A clear explanation of WHY it is a vulnerability in this context.
   - If a snippet is NOT vulnerable, you don't need to mention it.
   - If, after analyzing all provided snippets, you find NO vulnerabilities, state clearly: "No instances of the '[Smell Name]' smell were found in the provided code snippets."

--- Smell Definition ---
{smell_definition}

--- Positive Examples (without smell) ---
{positive_examples}

--- Suspicious Code Snippets from Provided Folder ---
{additional_folder_context}

Answer (in the same language as the Question):"""

# Creazione efettiva del prompt
prompt_template = PromptTemplate(
    input_variables=["TYPE_SMELL", "smell_definition", "positive_examples", "additional_folder_context"],
    template=prompt_template_str
)

# Funzione per il conteggio dei prompt
def count_tokens_for_llm_input(text_input: str, llm_instance: ChatGoogleGenerativeAI) -> int:
    try:
        return llm_instance.get_num_tokens(text_input)
    except Exception as e:
        print(f"Errore durante il conteggio dei token: {e}")
        return -1

print(f"\n-----------------RAG with Gemini for {TYPE_SMELL} Smell detection----------------")
print("Type the name of the smell you want to do the analysis on (or 'exit' to exit).")

while True:
    user_query = input(f"Name of {TYPE_SMELL} Smell: ")
    if user_query.lower() in ["exit", "quit", "esci", "stop", "x", "q"]:
        print("Exit from the program.")
        break

    # Caricamento dello smell
    smell_data = load_smell_data(user_query, knowledge_base_dir)
    if not smell_data:
        continue # Chiedi un nuovo input se lo smell non esiste

    # Caricamento e chunking del codice passato nel path
    folder_path_input = input("Specify a folder path for the code to be analyzed: ").strip()
    if not (folder_path_input and os.path.isdir(folder_path_input)):
        print("Invalid or empty folder path. Please try again.")
        continue

    source_code_docs = load_folder_path_documents(folder_path_input)
    if not source_code_docs:
        print("No source code documents were found in the specified path.")
        continue
        
    code_chunks = get_code_chunks(source_code_docs)
    if not code_chunks:
        print("Could not split source code into chunks.")
        continue

    # Creazione del vector store temporaneo del codice passato nel path
    print("Creating temporary vector store for the user's code...")

    # Si può aumentare la batch_size se hai molta RAM, per andare più veloce.
    batch_size = 64
    all_embeddings = []
    all_texts = []
    all_metadatas = []

    # Usiamo TQDM (barra di progressione) per iterare sui batch di documenti
    for i in tqdm(range(0, len(code_chunks), batch_size), desc="Generating Embeddings"):
        # Seleziona il batch di documenti corrente
        batch_docs = code_chunks[i:i + batch_size]
        # Estrai il testo da ogni documento nel batch
        batch_texts = [doc.page_content for doc in batch_docs]
        
        # Crea gli embeddings per l'intero batch in una sola chiamata (efficiente)
        batch_embeddings = embeddings_model.embed_documents(batch_texts)
        
        # Aggiungi i risultati alle liste principali
        all_embeddings.extend(batch_embeddings)
        all_texts.extend(batch_texts)
        all_metadatas.extend([doc.metadata for doc in batch_docs])

    # Creiamo l'indice FAISS
    print("Building FAISS index...")
    text_embedding_pairs = list(zip(all_texts, all_embeddings))

    # Usiamo from_embeddings per costruire il vector store da dati pre-calcolati
    code_vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=embeddings_model, 
        metadatas=all_metadatas    
    )
    print("Vector store created successfully.")

    # Cerchiamo le parti di codice sospette
    print("Searching for suspicious code snippets based on examples from the KB...")
    # Estrai tutti gli esempi negativi dalla KB per usarli come query
    search_queries = [ex['negative_example'] for ex in smell_data.get('manifestations', [])]
    if not search_queries:
        print("Warning: No 'negative examples' found in the KB for this smell. Analysis might be inaccurate.")
        continue
    
    # Concatena gli esempi
    search_query_str = "\n".join(search_queries)

    # Esegui la ricerca nel vectorstore del codice utente
    retrieved_code_snippets = code_vectorstore.similarity_search(search_query_str, k=20) # k indica il numero di snippet di codice da recuperare

    if not retrieved_code_snippets:
        print("No code snippets were found to be similar to the examples. The code is likely clean for this smell.")
        continue
    
    print(f"Found {len(retrieved_code_snippets)} potentially suspicious code snippets to analyze.")

    # Prepara i campi per il prompt
    smell_definition = f"Description: {smell_data['brief_description']}"
    positive_examples = "\n\n".join(
        [f"--- Positive Example ({ex['language']}) ---\n{ex['positive_example']}\nExplanation: {ex['explanation']}" for ex in smell_data.get('positive', [])]
        ) if 'positive' in smell_data else "No positive examples available."

    # Contesto dal codice utente: solo gli snippet sospetti
    code_context_for_prompt = "\n\n".join(
        [f"--- Snippet from file: {doc.metadata.get('source', 'Unknown')} ---\n```\n{doc.page_content}\n```" for doc in retrieved_code_snippets]
    )
    
    final_prompt_string = prompt_template.format(
        TYPE_SMELL=TYPE_SMELL,
        smell_definition=smell_definition,
        positive_examples=positive_examples,
        additional_folder_context=code_context_for_prompt
    ).replace("[Smell Name]", user_query) # Sostituisce il placeholder nel template



    # ("\n--- Prompt Finale (prima dell'invio all'LLM) ---")
    # è commentato perchè è lungo, usarlo come debug
    # print(final_prompt_string) 
    print(f"(Prompt length: {len(final_prompt_string)} characters)")


    # Conteggio dei token (opzionale)
    token_count = count_tokens_for_llm_input(final_prompt_string, llm)
    if token_count != -1:
        print(f"Number of tokens estimated for input to the LLM: {token_count}")

    # Invocazione dell'LLM con il prompt composto
    print("\nRequest to the LLM in progress...")
    try:
        response = llm.invoke(final_prompt_string)
        answer = response.content
        print("\n--- LLM's response ---")
        print(answer)
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        print("Make sure that the LLM model is configured correctly and that the API key is valid and enabled.")