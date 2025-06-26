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

def load_single_file(file_path: str) -> list[Document]:
    all_documents = []

    if not os.path.isfile(file_path):
        print(f"'{file_path}' is not a valid file.")
        return []

    filename = os.path.basename(file_path)
    lower = filename.lower()

    if lower in ("package-lock.json", "yarn.lock"):
        print(f"Ignored file: {filename}")
        return []

    ext = os.path.splitext(filename)[-1].lower() if os.path.splitext(filename)[-1] else filename.lower()
    if ext not in LOADER_MAPPING:
        print(f"Unsupported file extension: {filename}")
        return []

    loader_class, loader_kwargs = LOADER_MAPPING[ext]
    print(f"Uploading single file: {file_path} (type: {ext})")
    try:
        loader = loader_class(file_path, **loader_kwargs)
        docs = loader.load()
        if docs:
            for doc in docs:
                doc.metadata["source"] = file_path
            all_documents.extend(docs)
    except Exception as e:
        print(f"Error while loading file '{file_path}': {e}")

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

SMELL_INSTRUCTIONS = {
    "hardcoded secrets": """
Look specifically for:
- Passwords, API keys, secrets or tokens written in code or configs.
- Environment variables or JSON keys like "password", "secret", "token", etc.
- Credentials exposed in `application.properties`, `.env`, `.json`, or code.

Do NOT flag:
- Configurations that read from environment variables (e.g., `${TOKEN}`).
- Placeholder values like `<password>` or `"***"` unless clearly real.

Explain:
- What the hardcoded secret is.
- Why it creates a security risk (e.g., credentials leak, repo exposure).
""",

    "publicly accessible microservice": """
Look specifically for:
- Services exposing ports via Docker or Kubernetes (e.g., "8080:8080", NodePort, LoadBalancer).
- Bindings to 0.0.0.0 or localhost where not appropriate.
- Services exposing internal tools like ActiveMQ, Redis, or databases.

Do NOT flag:
- Legitimately exposed public frontends behind an API Gateway.

Explain:
- Why this service is exposed and how that creates risk (unauthenticated access, attack surface).
""",

    "unauthenticated traffic": """
Look specifically for:
- Services that use HTTP instead of HTTPS in internal or external communication.
- Endpoints or services with no authentication mechanism.
- Lack of authorization headers or session validation in frontend/backend flows.

Do NOT flag:
- Endpoints that are clearly behind an auth gateway.

Explain:
- Why this traffic is unauthenticated.
- What access an attacker would gain.
""",

    "insufficient access control": """
Look specifically for:
- REST endpoints that expose sensitive operations without checking roles or permissions.
- Lack of use of access control annotations (e.g., @PreAuthorize, @RolesAllowed).
- No middleware/auth layer on critical routes.

Do NOT flag:
- Public resources clearly meant to be exposed (e.g., `/login`, `/docs`).

Explain:
- Which access control is missing.
- What type of unauthorized access could occur.
"""
}

# Nuovo prompt, si può migliorare
prompt_template_str = """Instructions:
1. You are an expert security auditor. Your task is to analyze specific code snippets for a given [Smell name].
2. The 'Smell Definition' provides the official description and remediation strategies for the [Smell name] smell.
3. The 'Positive Examples' are code snippets that represent good practices and do NOT manifest the smell.
4. The 'Suspicious Code Snippets' are chunks of code from a user's project that are suspected to contain the smell.
5. Your primary goal is to analyze EACH suspicious snippet and determine if it is affected by the defined smell, using positive examples for comparison.
6. Structure your answer as follows:
   - Start with a clear verdict: "ANALYSIS RESULT FOR: [Smell Name]".
   - List ALL the services that contain at least one confirmed instance of this smell. Format:
    "Analyzed services with security smell:
    - service-name-1
    - service-name-2"
    If no services are affected, return:
    "Analyzed services with security smell: []"
   - For each analyzed file, create a section, divided by a line of #.
   - Under each file, list the snippets that ARE VULNERABLE.
   - For each vulnerable snippet, provide:
     a. The snippet of code or that contains the smell.
     b. A clear and exhaustive explanation of WHY it is a vulnerability in this context.
   - If a snippet is NOT vulnerable, you don't need to mention it.
   - If, after analyzing all provided snippets, you find NO vulnerabilities, state clearly: "No instances of the '[Smell Name]' smell were found in the provided code snippets."

--- Smell Definition ---
{smell_definition}

--- Smell-specific detection instructions ---
{smell_specific_instructions}

--- Positive Examples (without smell) ---
{positive_examples}

--- Suspicious Code Snippets from Provided Folder ---
{additional_folder_context}

Answer (in the same language as the Question):"""

# Creazione efettiva del prompt
prompt_template = PromptTemplate(
    input_variables=["TYPE_SMELL", "smell_definition", "positive_examples", "additional_folder_context", "smell_specific_instructions"],
    template=prompt_template_str
)

# Funzione per il conteggio dei prompt
def count_tokens_for_llm_input(text_input: str, llm_instance: ChatGoogleGenerativeAI) -> int:
    try:
        return llm_instance.get_num_tokens(text_input)
    except Exception as e:
        print(f"Error while counting tokens: {e}")
        return -1

def analyze_services_individually(smell_data, base_folder_path, user_query):
    #Esegue l'analisi RAG iterando su ogni microservizio trovato nel percorso base.
    try:
        service_folders = [f for f in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, f))]
        service_folders = [f for f in service_folders if not f.startswith('.') and f not in ['kubernetes', 'knowledge_base']]
        if not service_folders:
            print(f"No service folders found in '{base_folder_path}'. Make sure the path contains the microservice folders.")
            return
        print(f"Found {len(service_folders)} services to analyze: {service_folders}")
    except Exception as e:
        print(f"Error reading service folders from '{base_folder_path}': {e}")
        return

    all_retrieved_snippets_from_all_services = []
    processed_content = set()

    # K snippets to retrieve for EACH service.
    k_per_service = 5

    for service_name in tqdm(service_folders, desc="Analyzing services"):
        service_path = os.path.join(base_folder_path, service_name)
        print(f"\n--- Analyzing service: {service_name} ---")

        source_code_docs = load_folder_path_documents(service_path)
        if not source_code_docs:
            print(f"No source code documents found for service '{service_name}'.")
            continue
            
        code_chunks = get_code_chunks(source_code_docs)
        if not code_chunks:
            print(f"Unable to split source code into chunks for service '{service_name}'.")
            continue

        print(f"Creating temporary vector store for '{service_name}'...")

        service_vectorstore = FAISS.from_documents(code_chunks, embeddings_model)
        print(f"Vector store for '{service_name}' created successfully.")

        search_queries = [ex['negative_example'] for ex in smell_data.get('manifestations', [])]
        if not search_queries:
            print("Warning: No 'negative_example' found in the KB for this smell. The analysis may be inaccurate.")
            continue
        search_query_str = "\n".join(search_queries)

        print(f"Searching for {k_per_service} suspicious snippets for service '{service_name}'...")
        retrieved_for_service = service_vectorstore.similarity_search(
            query=search_query_str,
            k=k_per_service
        )
        
        print(f"Retrieved {len(retrieved_for_service)} snippets for service '{service_name}'.")

        for snippet in retrieved_for_service:
            if snippet.page_content not in processed_content:
                all_retrieved_snippets_from_all_services.append(snippet)
                processed_content.add(snippet.page_content)

    try:
        all_entries = os.listdir(base_folder_path)
        top_level_files = [f for f in all_entries if os.path.isfile(os.path.join(base_folder_path, f))]

        # Filter only supported files
        supported_files = []
        for f in top_level_files:
            ext = os.path.splitext(f)[-1].lower()
            if ext in LOADER_MAPPING:
                supported_files.append(f)

        if not supported_files:
            print("No supported files found in the root directory.")
            return

        print(f"Found {len(supported_files)} supported files in the root: {supported_files}")

        for filename in tqdm(supported_files, desc="Analyzing top-level files"):
            file_path = os.path.join(base_folder_path, filename)
            print(f"\nAnalyzing file: {file_path}")

            try:
                docs = load_single_file(file_path)
                if not docs:
                    print(f"No documents loaded from {filename}")
                    continue

                code_chunks = get_code_chunks(docs)
                if not code_chunks:
                    print(f"No chunks generated from {filename}")
                    continue

                print(f"Creating vector store for '{filename}'...")
                file_vectorstore = FAISS.from_documents(code_chunks, embeddings_model)

                search_queries = [ex['negative_example'] for ex in smell_data.get('manifestations', [])]
                if not search_queries:
                    print("No 'negative_example' found in the KB. Skipping search.")
                    continue

                search_query_str = "\n".join(search_queries)
                retrieved_snippets = file_vectorstore.similarity_search(
                    query=search_query_str,
                    k=k_per_service
                )

                print(f"Found {len(retrieved_snippets)} suspicious snippets in '{filename}'.")

                for snippet in retrieved_snippets:
                    if snippet.page_content not in processed_content:
                        all_retrieved_snippets_from_all_services.append(snippet)
                        processed_content.add(snippet.page_content)

            except Exception as e:
                print(f"Error while processing file '{filename}': {e}")

    except Exception as e:
        print(f"Error reading files in '{base_folder_path}': {e}")

    if not all_retrieved_snippets_from_all_services:
        print("\nNo code snippets similar to the examples were found in the services. The code is likely clean from this smell.")
        return

    print(f"\nFound a total of {len(all_retrieved_snippets_from_all_services)} potentially suspicious code snippets to analyze.")

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
        additional_folder_context=code_context_for_prompt,
        smell_specific_instructions=SMELL_INSTRUCTIONS.get(user_query.lower(), "No specific instructions available.")
    ).replace("[Smell Name]", user_query)

    print("\n--- Final Prompt (before submission to the LLM) ---")
    print(final_prompt_string) # Decommenta per debug
    print(f"(Prompt lenght: {len(final_prompt_string)})")

    token_count = count_tokens_for_llm_input(final_prompt_string, llm)
    if token_count != -1:
        print(f"Number of tokens estimated for input to the LLM: {token_count}")

    print("\nRequest to the LLM in progress...")
    try:
        response = llm.invoke(final_prompt_string)
        answer = response.content
        print("\n--- LLM Response ---")
        print(answer)
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
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

    print(f"\n--- Evaluation ---")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

print(f"\n-----------------RAG with Gemini for {TYPE_SMELL} Smell detection----------------")
print("Scrivi il nome dello smell su cui vuoi fare l'analisi (o 'exit' per uscire).")

while True:
    user_query = input(f"\nName of the {TYPE_SMELL} Smell: ")
    if user_query.lower() in ["exit", "quit", "esci", "stop", "x", "q"]:
        print("Exiting the programme.")
        break

    # Caricamento dello smell
    smell_data = load_smell_data(user_query, knowledge_base_dir)
    if not smell_data:
        continue 

    # Input della cartella base che contiene i microservizi
    folder_path_input = input("Specifies the path to the base folder containing the microservices to be analysed:").strip()
    if not (folder_path_input and os.path.isdir(folder_path_input)):
        print("Invalid or empty folder path. Please try again.")
        continue

    # Chiama la nuova funzione di analisi che gestisce tutto il processo
    analyze_services_individually(smell_data, folder_path_input, user_query)