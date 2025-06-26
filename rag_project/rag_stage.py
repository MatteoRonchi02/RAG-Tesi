import os
import warnings
import json
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
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

# Folders to completely ignore (will not even be explored)
IGNORE_DIRS = {'.git', '.idea', '.vscode', 'target', 'build', '.mvn', 'testutil'}
# Path fragments that, if present, cause the folder to be excluded
IGNORE_PATH_FRAGMENTS = {os.path.join('src', 'test')}
# Specific files to ignore by name
IGNORE_FILES = {'mvnw', 'Dockerfile',  'pom.xml'}
# Files to ignore based on suffix (for DTO, Entity, etc.)
IGNORE_FILENAME_SUFFIXES = {
    'DTO.java', 'Entity.java', 'Event.java',
    'Request.java', 'Response.java', 'Exception.java'
}

load_dotenv()
aiplatform.init(
    project=os.getenv("GPC_PROJECT_ID"),
    location= "europe-west1"
)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Ignore FutureWarning, non-blocking issue but to be solved
warnings.filterwarnings("ignore", category=FutureWarning)

# LLM model loading
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.1,
        max_tokens=8192 # Maximum tokens for the generated *response*
    )
    print(f"-----------Model LLM '{llm.model}' loaded successfully-----------")
    print("=============================================================================")
except Exception as e:
    raise ValueError(f"Error loading LLM model: {e}. Check the model name and API key.")

# Main knowledge base directory
knowledge_base_dir = 'knowledge_base'

# Mapping file extensions to their respective loader
LOADER_MAPPING = {
    ".java": (TextLoader, {"encoding": "utf-8"}),
    ".js": (TextLoader, {"encoding": "utf-8"}),
    ".vue": (TextLoader, {"encoding": "utf-8"}),
    ".html": (TextLoader, {"encoding": "utf-8"}),
    "dockerfile": (TextLoader, {"encoding": "utf-8"}),
    ".sh": (TextLoader, {"encoding": "utf-8"}),
    ".groovy": (TextLoader, {"encoding": "utf-8"}),
    ".json": (TextLoader, {"encoding": "utf-8"}),
    ".properties": (TextLoader, {"encoding": "utf-8"}),
    ".yml": (TextLoader, {"encoding": "utf-8"}),
    ".yaml": (TextLoader, {"encoding": "utf-8"}),
    ".gandle": (TextLoader, {"encoding": "utf-8"})
}

# Load KB data from json format
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

#Load all documents (entire) from a directory and subfolders.
def load_folder_path_documents(directory: str) -> list[Document]:
    all_documents = []
    print(f"Uploading documents from the chosen {directory} path")
    for root, dirs, files in os.walk(directory):

        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        if any(fragment in root for fragment in IGNORE_PATH_FRAGMENTS):
            continue

        for filename in files:
            if filename in IGNORE_FILES:
                print(f"Ignoring specific file: {os.path.join(root, filename)}")
                continue
            if any(filename.endswith(suffix) for suffix in IGNORE_FILENAME_SUFFIXES):
                print(f"Ignoring by suffix: {os.path.join(root, filename)}")
                continue

            ext = os.path.splitext(filename)[-1].lower() if os.path.splitext(filename)[-1] else filename.lower()
            if ext in LOADER_MAPPING:
                file_path = os.path.join(root, filename)
                loader_class, loader_kwargs = LOADER_MAPPING[ext]
                # print(f"Upload: {file_path} (type: {ext})") # Verbose, can be commented out
                try:
                    loader = loader_class(file_path, **loader_kwargs)
                    docs = loader.load()
                    if docs:
                        for doc in docs:
                            doc.metadata["source"] = file_path
                        all_documents.extend(docs)
                except Exception as e:
                    print(f"Error while loading {file_path}: {e}")

    print(f"Total documents uploaded by {directory}: {len(all_documents)}\n")
    return all_documents

# NUOVA FUNZIONE: Carica solo i file di orchestrazione dalla root del progetto
def load_orchestration_files(base_directory: str) -> list[Document]:
    """
    Loads only orchestration files (docker-compose*.yml, kubernetes.yml, etc.)
    from the project's root directory.
    """
    orchestration_documents = []
    # Assicurati che il file principale sia prima, per dare priorità logica
    orchestration_filenames = ['docker-compose.yml', 'docker-compose-common.yml', 'kubernetes.yml']
    print(f"Searching for orchestration files in: {base_directory}")

    # Carica prima il file principale se esiste
    for filename in orchestration_filenames:
        if os.path.exists(os.path.join(base_directory, filename)):
            file_path = os.path.join(base_directory, filename)
            ext = os.path.splitext(filename)[-1].lower()
            if ext in LOADER_MAPPING:
                loader_class, loader_kwargs = LOADER_MAPPING[ext]
                print(f"Loading orchestration file: {file_path}")
                try:
                    loader = loader_class(file_path, **loader_kwargs)
                    docs = loader.load()
                    if docs:
                        for doc in docs:
                            doc.metadata["source"] = file_path
                        orchestration_documents.extend(docs)
                except Exception as e:
                    print(f"Error loading orchestration file {file_path}: {e}")

    print(f"Total orchestration documents loaded: {len(orchestration_documents)}")
    return orchestration_documents


def load_single_file(file_path: str) -> list[Document]:
    all_documents = []

    if not os.path.isfile(file_path):
        print(f"'{file_path}' is not a valid file.")
        return []

    filename = os.path.basename(file_path)
    lower = filename.lower()

    if lower in ("package-lock.json", "yarn.lock"):
        print(f"File ignored: {filename}")
        return []

    ext = os.path.splitext(filename)[-1].lower() if os.path.splitext(filename)[-1] else filename.lower()
    if ext not in LOADER_MAPPING:
        print(f"Extension not supported for file: {filename}")
        return []

    loader_class, loader_kwargs = LOADER_MAPPING[ext]
    print(f"Upload single file: {file_path} (type: {ext})")
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

# MODIFICA: Rimosso Language.YAML
def get_code_chunks(code_documents: list[Document]) -> list[Document]:
    print("Splitting source code into manageable chunks...")
    all_chunks = []
    language_map = {
        ".java": Language.JAVA,
        ".js": Language.JS,
        ".html": Language.HTML,
        ".scala": Language.SCALA
        # .yml rimosso per evitare l'AttributeError
    }

    default_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for doc in tqdm(code_documents, desc="Splitting source code"):
        ext = os.path.splitext(doc.metadata["source"])[-1].lower()
        language = language_map.get(ext)

        if language:
            try:
                splitter = RecursiveCharacterTextSplitter.from_language(language=language, chunk_size=1000, chunk_overlap=150)
                chunks = splitter.split_documents([doc])
            except:
                chunks = default_splitter.split_documents([doc])
        else:
            # Il default splitter gestirà tutti gli altri file, inclusi i .yml
            chunks = default_splitter.split_documents([doc])
        all_chunks.extend(chunks)

    print(f"Source code split into {len(all_chunks)} chunks.")
    return all_chunks

def extract_services_from_llm_output(answer: str) -> list[str]:
    lines = answer.splitlines()
    services = []
    in_service_section = False

    for line in lines:
        if "Analyzed services with Architectural smell:" in line:
            in_service_section = True
            continue
        if in_service_section:
            content = line.strip()
            if content.startswith("-"):
                service_name = content.lstrip("-").strip()
                services.append(service_name)
            elif "[]" in content:
                return []
            elif content == "" and services:
                break
    return services


print("--------------------Initialization embeddings in progress--------------------")
try:
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("Embeddings model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error when creating embeddings model: {e}")

print("=============================================================================")

prompt_template_str = """Instructions:
1. You are an expert Architectural Smell auditor. Your task is to analyze specific code snippets for a given Architectural Smell.
2. The 'Smell Definition' provides the official description and remediation strategies for the Architectural vulnerability.
3. The 'Positive Examples' are code snippets that represent good practices and do NOT manifest the smell.
4. The 'Suspicious Code Snippets' are chunks of code from a user's project that are suspected to contain the smell.
5. Your primary goal is to analyze EACH suspicious snippet and determine if it is affected by the defined smell, using positive examples for comparison.
6. Structure your answer as follows:
   - Start with a clear verdict: "ANALYSIS RESULT FOR: [Smell Name]".
   - Create a list that contains only services that contain Architectural smell, like this: "Analyzed services with Architectural smell: \n - name of service".
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

prompt_template = PromptTemplate(
    input_variables=["smell_definition", "positive_examples", "additional_folder_context"],
    template=prompt_template_str
)

def count_tokens_for_llm_input(text_input: str, llm_instance: ChatGoogleGenerativeAI) -> int:
    try:
        return llm_instance.get_num_tokens(text_input)
    except Exception as e:
        print(f"Error while counting tokens: {e}")
        return -1

def analyze_services_individually(smell_data, base_folder_path, user_query):
    smell_definition = f"Description: {smell_data['brief_description']}"
    positive_examples = "\n\n".join(
        [f"--- Positive Example ({ex['language']}) ---\n{ex['positive_example']}\nExplanation: {ex['explanation']}" for ex in smell_data.get('positive', [])]
    ) if 'positive' in smell_data else "No positive examples available."

    code_context_for_prompt = "No context available."

    if user_query.lower() == 'no api gateway':
        print("\n=== Special analysis path for 'No API Gateway' ===")
        orchestration_docs = load_orchestration_files(base_folder_path)

        if not orchestration_docs:
            print("\nNo orchestration files (docker-compose.yml, etc.) found. Cannot analyze for 'No API Gateway' smell.")
            return

        code_context_for_prompt = "\n\n".join(
            [f"--- Content from file: {doc.metadata.get('source', 'Unknown')} ---\n```yaml\n{doc.page_content}\n```" for doc in orchestration_docs]
        )

    else:
        print("\n=== Standard analysis path for source code smells ===")
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
            retrieved_for_service = service_vectorstore.similarity_search(query=search_query_str, k=k_per_service)

            print(f"Retrieved {len(retrieved_for_service)} snippets for service '{service_name}'.")

            for snippet in retrieved_for_service:
                if snippet.page_content not in processed_content:
                    all_retrieved_snippets_from_all_services.append(snippet)
                    processed_content.add(snippet.page_content)

        # Analisi dei file nella root (se necessario per altri smell)
        # Questa logica è stata semplificata e resa più robusta
        try:
            all_entries = os.listdir(base_folder_path)
            top_level_files = [f for f in all_entries if os.path.isfile(os.path.join(base_folder_path, f))]

            # Filter only supported files
            supported_files = []
            for f in top_level_files:
                ext = os.path.splitext(f)[-1].lower() if os.path.splitext(f)[-1] else f.lower()
                if ext in LOADER_MAPPING:
                    supported_files.append(f)

            if supported_files:
                print(f"Found {len(supported_files)} supported files in the root: {supported_files}")
                for filename in tqdm(supported_files, desc="Analyzing top-level files"):
                    file_path = os.path.join(base_folder_path, filename)
                    docs = load_single_file(file_path)
                    if not docs: continue
                    code_chunks = get_code_chunks(docs)
                    if not code_chunks: continue

                    file_vectorstore = FAISS.from_documents(code_chunks, embeddings_model)
                    search_queries = [ex['negative_example'] for ex in smell_data.get('manifestations', [])]
                    if not search_queries: continue

                    retrieved_snippets = file_vectorstore.similarity_search(query="\n".join(search_queries), k=k_per_service)
                    for snippet in retrieved_snippets:
                        if snippet.page_content not in processed_content:
                            all_retrieved_snippets_from_all_services.append(snippet)
                            processed_content.add(snippet.page_content)
        except Exception as e:
            print(f"Error reading top-level files in '{base_folder_path}': {e}")


        if not all_retrieved_snippets_from_all_services:
            print("\nNo code snippets similar to the examples were found. The code is probably clean for this smell.")
            return

        print(f"\nFound a total of {len(all_retrieved_snippets_from_all_services)} potentially suspicious code snippets to analyze.")
        code_context_for_prompt = "\n\n".join(
            [f"--- Snippet from file: {doc.metadata.get('source', 'Unknown')} ---\n```\n{doc.page_content}\n```" for doc in all_retrieved_snippets_from_all_services]
        )

    final_prompt_string = prompt_template.format(
        smell_definition=smell_definition,
        positive_examples=positive_examples,
        additional_folder_context=code_context_for_prompt
    ).replace("[Smell Name]", user_query)

    print("\n--- Final Prompt (before sending to LLM) ---")
    print(final_prompt_string)
    # print(f"(Prompt length: {len(final_prompt_string)} characters)")

    # token_count = count_tokens_for_llm_input(final_prompt_string, llm)
    # if token_count != -1:
    #     print(f"Estimated number of tokens for LLM input: {token_count}")

    print("\nRequest to LLM in progress...")
    try:
        response = llm.invoke(final_prompt_string)
        answer = response.content
        print("\n--- LLM Response ---")
        print(answer)
    except Exception as e:
        print(f"Error while invoking the LLM: {e}")
        return

    # Evaluation logic
    ground_truth = {
        "customers-service": ["endpoint based service interaction"],
        "accounts-service": ["endpoint based service interaction"],
        "transactions-service": ["endpoint based service interaction"],
        "customers-view-service": ["shared persistence", "endpoint based service interaction"],
        "accounts-view-service": ["shared persistence", "endpoint based service interaction"],
        "api-gateway-service": []
    }

    smell_name = user_query.lower()
    predicted_services = extract_services_from_llm_output(answer)
    print("\n\n>>> Predicted services:", predicted_services)

    predicted = {(s.strip(), smell_name) for s in predicted_services}
    true_labels = {(s, smell_name) for s, smells in ground_truth.items() if smell_name in smells}

    print(">>> Predicted Set:", predicted)
    print(">>> Ground Truth Set:", true_labels)

    TP = len(predicted & true_labels)
    FP = len(predicted - true_labels)
    FN = len(true_labels - predicted)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n--- Evaluation ---")
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")

print("\n-----------------RAG with Gemini for Architectural Smell detection----------------")
print("Type the name of the smell you want to analyze (or 'exit' to quit).")

while True:
    user_query = input("\nName of the Architectural Smell: ")
    if user_query.lower() in ["exit", "quit", "esci", "stop", "x", "q"]:
        print("Exiting the program.")
        break

    smell_data = load_smell_data(user_query, knowledge_base_dir)
    if not smell_data:
        continue

    folder_path_input = input("Specify the path of the base folder containing the microservices to analyze: ").strip()
    if not (folder_path_input and os.path.isdir(folder_path_input)):
        print("Invalid or empty folder path. Try again.")
        continue

    analyze_services_individually(smell_data, folder_path_input, user_query)