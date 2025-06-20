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
TYPE_SMELL = os.getenv("TYPE_SMELL", "Security")   

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
    ".java":          (TextLoader, {"encoding": "utf-8"}),
    #".js":            (TextLoader, {"encoding": "utf-8"}),
    ".vue":           (TextLoader, {"encoding": "utf-8"}),
    ".html":          (TextLoader, {"encoding": "utf-8"}),
    "dockerfile":     (TextLoader, {"encoding": "utf-8"}),
    ".sh":            (TextLoader, {"encoding": "utf-8"}),
    ".groovy":        (TextLoader, {"encoding": "utf-8"}),
    ".json":          (TextLoader, {"encoding": "utf-8"}),
    ".properties":    (TextLoader, {"encoding": "utf-8"}),
    ".yml":           (TextLoader, {"encoding": "utf-8"}),
    ".yaml":          (TextLoader, {"encoding": "utf-8"}),
    ".env":           (TextLoader, {"encoding": "utf-8"}),
    ".xml":           (TextLoader, {"encoding": "utf-8"}),
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
    for root, dirs, files in os.walk(directory):
        if "node_modules" in dirs:
            dirs.remove("node_modules")
        
        for filename in files:
            # Controllo per ignorare le estensioni non mappate
            lower = filename.lower()
            if lower in ("package-lock.json", "yarn.lock"):
                continue

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
        ".html": Language.HTML,
        ".scala": Language.SCALA
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

def extract_services_from_llm_output(answer: str) -> list[str]:
    lines = answer.splitlines()
    services = []
    in_service_section = False

    for line in lines:
        if "Analyzed services with security smell:" in line:
            in_service_section = True
            continue
        if in_service_section:
            if line.strip().startswith("-"):
                service_name = line.strip().lstrip("-").strip()
                services.append(service_name)
            elif line.strip() == "":
                break 
    return services

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
   - Create a list that contains ONLY the name of services that ONLY contain security smell, like this: "Analyzed services with security smell: \n - name of service", if there aren't make and empty list.
   - For each analyzed file path, create a section, divided by a line of #.
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

def analyze_services_individually(smell_data, base_folder_path, user_query):
    """
    Esegue l'analisi RAG iterando su ogni microservizio trovato nel percorso base.
    """
    # 1. Identifica le cartelle dei servizi
    try:
        service_folders = [f for f in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, f))]
        # Filtra cartelle non pertinenti come .git, etc.
        service_folders = [f for f in service_folders if not f.startswith('.') and f not in ['kubernetes', 'knowledge_base']]
        if not service_folders:
            print(f"Nessuna cartella di servizio trovata in '{base_folder_path}'. Assicurati che il path contenga le cartelle dei microservizi.")
            return
        print(f"Found {len(service_folders)} services to analyze: {service_folders}")
    except Exception as e:
        print(f"Errore nella lettura delle cartelle dei servizi da '{base_folder_path}': {e}")
        return

    all_retrieved_snippets_from_all_services = []
    processed_content = set()
    
    # K snippet da recuperare per OGNI servizio.
    k_per_service = 5

    # 2. Itera su ogni servizio
    for service_name in tqdm(service_folders, desc="Analyzing services"):
        service_path = os.path.join(base_folder_path, service_name)
        print(f"\n--- Analyzing service: {service_name} ---")

        # 3. Carica e splitta i documenti SOLO per il servizio corrente
        source_code_docs = load_folder_path_documents(service_path)
        if not source_code_docs:
            print(f"Nessun documento di codice sorgente trovato per il servizio '{service_name}'.")
            continue
            
        code_chunks = get_code_chunks(source_code_docs)
        if not code_chunks:
            print(f"Impossibile dividere il codice sorgente in chunk per il servizio '{service_name}'.")
            continue

        # 4. Crea un vector store TEMPORANEO e ISOLATO per il servizio corrente
        print(f"Creazione del vector store temporaneo per '{service_name}'...")
        # NOTA: from_documents è più diretto se non si ha bisogno di gestire i batch manualmente
        service_vectorstore = FAISS.from_documents(code_chunks, embeddings_model)
        print(f"Vector store per '{service_name}' creato con successo.")


        # 5. Esegui la similarity search su questo vector store isolato
        search_queries = [ex['negative_example'] for ex in smell_data.get('manifestations', [])]
        if not search_queries:
            print("Attenzione: Nessun 'negative_example' trovato nella KB per questo smell. L'analisi potrebbe essere imprecisa.")
            continue
        search_query_str = "\n".join(search_queries)

        print(f"Ricerca di {k_per_service} snippet sospetti per il servizio '{service_name}'...")
        retrieved_for_service = service_vectorstore.similarity_search(
            query=search_query_str,
            k=k_per_service
        )
        
        print(f"Recuperati {len(retrieved_for_service)} snippet per il servizio '{service_name}'.")

        # Aggiungi gli snippet alla lista globale, evitando duplicati di contenuto
        for snippet in retrieved_for_service:
            if snippet.page_content not in processed_content:
                all_retrieved_snippets_from_all_services.append(snippet)
                processed_content.add(snippet.page_content)


    if not all_retrieved_snippets_from_all_services:
        print("\nNessuno snippet di codice simile agli esempi è stato trovato nei servizi. Il codice è probabilmente pulito per questo smell.")
        return

    print(f"\n_Trovati in totale {len(all_retrieved_snippets_from_all_services)} snippet di codice potenzialmente sospetti da analizzare.")

    # Il resto del codice per formattare il prompt e chiamare l'LLM rimane quasi invariato
    smell_definition = f"Description: {smell_data['brief_description']}"
    positive_examples = "\n\n".join(
        [f"--- Positive Example ({ex['language']}) ---\n{ex['positive_example']}\nExplanation: {ex['explanation']}" for ex in smell_data.get('positive', [])]
    ) if 'positive' in smell_data else "No positive examples available."

    code_context_for_prompt = "\n\n".join(
        [f"--- Snippet from file: {doc.metadata.get('source', 'Unknown')} ---\n```\n{doc.page_content}\n```" for doc in all_retrieved_snippets_from_all_services]
    )
    
    final_prompt_string = prompt_template.format(
        TYPE_SMELL=TYPE_SMELL,
        smell_definition=smell_definition,
        positive_examples=positive_examples,
        additional_folder_context=code_context_for_prompt
    ).replace("[Smell Name]", user_query)

    print("\n--- Prompt Finale (prima dell'invio all'LLM) ---")
    print(final_prompt_string) # Decommenta per debug
    print(f"(Lunghezza prompt: {len(final_prompt_string)} caratteri)")

    token_count = count_tokens_for_llm_input(final_prompt_string, llm)
    if token_count != -1:
        print(f"Numero di token stimati per l'input all'LLM: {token_count}")

    print("\nRichiesta all'LLM in corso...")
    try:
        response = llm.invoke(final_prompt_string)
        answer = response.content
        print("\n--- Risposta dell'LLM ---")
        print(answer)
    except Exception as e:
        print(f"Errore durante l'invocazione dell'LLM: {e}")
        return # Esce dalla funzione in caso di errore

    # Logica di valutazione
    ground_truth = {
        "customer-core": ["publicly accessible microservice"],
        "customer-management-backend": ["publicly accessible microservice", "insufficient access control", "unauthenticated traffic", "hardcoded secrets"],
        "customer-management-frontend": ["publicly accessible microservice", "unauthenticated traffic"],
        "customer-self-service-backend": ["publicly accessible microservice", "insufficient access control", "unauthenticated traffic", "hardcoded secrets"],
        "customer-self-service-frontend": ["publicly accessible microservice", "unauthenticated traffic"],
        "policy-management-backend": ["publicly accessible microservice", "insufficient access control", "unauthenticated traffic", "hardcoded secrets"],
        "policy-management-frontend": ["publicly accessible microservice", "unauthenticated traffic"],
        "spring-boot-admin": ["publicly accessible microservice", "insufficient access control", "unauthenticated traffic"],
        "risk-management-server": ["publicly accessible microservice", "hardcoded secrets"]
    }

    smell_name = user_query.lower()
    predicted_services = extract_services_from_llm_output(answer)
    print(">>> Servizi predetti:", predicted_services)

    predicted = {(os.path.basename(s), smell_name) for s in predicted_services}
    true_labels = {(s, smell_name) for s, smells in ground_truth.items() if smell_name in smells}
    
    print(">>> Set Predetto:", predicted)
    print(">>> Set Ground Truth:", true_labels)

    TP = len(predicted & true_labels)
    FP = len(predicted - true_labels)
    FN = len(true_labels - predicted)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n--- Valutazione ---")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

print(f"\n-----------------RAG with Gemini for {TYPE_SMELL} Smell detection----------------")
print("Scrivi il nome dello smell su cui vuoi fare l'analisi (o 'exit' per uscire).")

while True:
    user_query = input(f"\nNome dello {TYPE_SMELL} Smell: ")
    if user_query.lower() in ["exit", "quit", "esci", "stop", "x", "q"]:
        print("Uscita dal programma.")
        break

    # Caricamento dello smell
    smell_data = load_smell_data(user_query, knowledge_base_dir)
    if not smell_data:
        continue # Chiedi un nuovo input se lo smell non esiste

    # Input della cartella base che contiene i microservizi
    folder_path_input = input("Specifica il path della cartella base contenente i microservizi da analizzare: ").strip()
    if not (folder_path_input and os.path.isdir(folder_path_input)):
        print("Path della cartella non valido o vuoto. Riprova.")
        continue

    # Chiama la nuova funzione di analisi che gestisce tutto il processo
    analyze_services_individually(smell_data, folder_path_input, user_query)