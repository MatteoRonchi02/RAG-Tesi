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
            
            # Check to ignore unmapped extensions
            ext = os.path.splitext(filename)[-1].lower() if os.path.splitext(filename)[-1] else filename.lower()
            if ext in LOADER_MAPPING:
                file_path = os.path.join(root, filename)
                loader_class, loader_kwargs = LOADER_MAPPING[ext]
                print(f"Upload: {file_path} (type: {ext})")
                try:
                    loader = loader_class(file_path, **loader_kwargs)
                    docs = loader.load()
                    if docs:
                        # Add the full path as 'source' for clarity
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

# Server to chunk code based on its language
def get_code_chunks(code_documents: list[Document]) -> list[Document]:
    print("Splitting source code into manageable chunks...")
    all_chunks = []
    # Mapping extensions to language type for the splitter
    language_map = {
        ".java": Language.JAVA,
        ".js": Language.JS,
        ".html": Language.HTML,
        ".scala": Language.SCALA
    }
    
    # Default splitter for unmapped files (e.g. Dockerfile, .vue, .txt)
    default_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for doc in tqdm(code_documents, desc="Splitting source code"):
        ext = os.path.splitext(doc.metadata["source"])[-1].lower()
        language = language_map.get(ext)
        
        if language:
            splitter = RecursiveCharacterTextSplitter.from_language(language=language, chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents([doc])
        else:
             # Use default splitter for Dockerfile, Vue, etc.
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
    "no api gateway": """Your analysis for this specific smell MUST check for TWO distinct conditions:
        a. **Total Absence**: First, check the provided code for any evidence of a gateway component (e.g., using Spring Cloud Gateway, Zuul, Ocelot, etc.). If there is no such code across all services, you must report that the smell is present at the system level.
        b. **Bypassable Gateway**: If gateway code *does* exist, you MUST then inspect any `docker-compose.yml` file or similar container orchestration configuration. If services *other than* the designated API Gateway expose their `ports` to the public internet, this is a critical flaw. You must report this as a form of the 'No API Gateway' smell, as the gateway's routing is not being properly enforced, allowing direct access to microservices.""",
    
    "shared persistence": """Your analysis for the 'Shared Persistence' smell must be rigorous and technology-agnostic. To positively identify this smell in a service, you MUST find conclusive evidence for BOTH of the following conditions:
        1.  **Shared Data Source Configuration:**
            First, analyze configuration artifacts like `.yml`, `.properties`, `.json`, `.env` files, or container orchestration files (e.g., `docker-compose.yml`, Kubernetes manifests). You must find proof that multiple distinct services are configured to connect to the **exact same database instance** (e.g., same host, port, and database/schema name).

        2.  **Active Database Interaction Code:**
            Second, within the source code of that SAME service, you must find code that **actively interacts with a database**. Look for general patterns of database usage, such as:
            - The import and use of a database driver, client, or connector.
            - The instantiation or use of an Object-Relational Mapper (ORM) or Object-Document Mapper (ODM) client (e.g., Hibernate, Entity Framework, Sequelize, Mongoose, etc.).
            - The presence of classes or functions following patterns like Repository, Data Access Object (DAO), or Active Record.
            - Code that directly executes database queries (e.g., SQL statements).

        **Crucial Rule:** A service should only be reported if you find strong evidence for **BOTH** shared configuration (Condition 1) AND active database usage (Condition 2). If a service appears to have a shared configuration but you cannot find corresponding code that uses it, you must consider it a non-actionable configuration artifact and **NOT** report it as exhibiting the 'Shared Persistence' smell.
        """,

    "endpoint based service interaction": """Your analysis for this specific smell MUST focus on identifying synchronous, point-to-point communication between services that creates tight coupling. Look for:
        a. **HTTP Client Usage**: Identify the use of HTTP clients to call other internal services (e.g., `RestTemplate`, `WebClient`, `@FeignClient`, `HttpClient`, etc.).
        b. **Direct Service URLs**: Look for hard-coded URLs or configuration entries that point directly to another service's address.
        c. **Absence of Messaging**: The lack of message queue or event streaming clients (e.g., for RabbitMQ, Kafka) suggests synchronous communication.

        **Crucial Exception Rule:** An **API Gateway** component, whose primary function is to route external requests to internal services via HTTP, should **NOT** be reported as exhibiting this smell. The smell applies to direct, peer-to-peer communication between internal services, not to the centralized routing performed by a gateway.
    """,

    "wobbly service interaction": """Your analysis for this specific smell MUST focus on identifying inefficient and "chatty" communication patterns where a service makes multiple synchronous calls to another single service to complete one task. Look for:
        a. **Calls Inside Loops**: This is the strongest indicator. Search for code where an HTTP client (like `RestTemplate`, `WebClient`, `FeignClient`) is invoked REPEATEDLY inside a `for`, `while`, or `stream().forEach()` loop. For example, getting a list of IDs and then calling another service for each single ID in the list.
        b. **Sequential Calls to the Same Service**: Look for methods that make multiple, separate calls to the same remote service to gather different pieces of data about the same entity. For example, `product = productService.getProduct(id)`, then `stock = productService.getStock(id)`, then `reviews = productService.getReviews(id)`. This indicates the remote API is not coarse-grained enough.
        c. **Complex Data Aggregation**: Identify client-side logic that exists only to stitch together the results of multiple small calls from another service. This logic is a symptom of the wobbly interaction."""
}

# New prompt, can be improved
prompt_template_str = """Instructions:
1. You are an expert Architectural auditor. Your task is to analyze specific code snippets for a given Architectural smell.
2. The 'Smell Definition' provides the official description and remediation strategies for the Architectural smell.
3. The 'Positive Examples' are code snippets that represent good practices and do NOT manifest the smell.
4. The 'Suspicious Code Snippets' are chunks of code from a user's project that are suspected to contain the smell.
5. Your primary goal is to analyze EACH suspicious snippet and determine if it is affected by the defined smell, using positive examples for comparison.
6. Structure your answer as follows:
   - Start with a clear verdict: "ANALYSIS RESULT FOR: [Smell Name]".
   - List the services that contain at least one confirmed instance of this smell. Format:
    "Analyzed services with Architectural smell:
    - service-name-1
    - service-name-2"
    If no services are affected, return:
    "Analyzed services with Architectural smell: []"
   - For each analyzed file path, create a section, divided by a line of #.
   - Under each file path, list the snippets that ARE AFFECTED BY THE SMELL.
   - For each affected snippet, provide:
     a. The line of code or block that contains the smell.
     b. A clear explanation of WHY it is an Architectural smell in this context.
   - If a snippet is NOT affected by the smell, you don't need to mention it.
   - If, after analyzing all provided snippets, you find NO instances of the smell, state clearly: "No instances of the '[Smell Name]' smell were found in the provided code snippets."

--- Smell Definition ---
{smell_definition}

--- Smell-specific detection instructions ---
{smell_specific_instructions}

--- Positive Examples (without smell) ---
{positive_examples}

--- Suspicious Code Snippets from Provided Folder ---
{additional_folder_context}

Answer (in the same language as the Question):"""

# Actual prompt creation
prompt_template = PromptTemplate(
    input_variables=["smell_definition", "positive_examples", "additional_folder_context", "smell_specific_instructions"],
    template=prompt_template_str
)

# Function for counting prompt tokens
def count_tokens_for_llm_input(text_input: str, llm_instance: ChatGoogleGenerativeAI) -> int:
    try:
        return llm_instance.get_num_tokens(text_input)
    except Exception as e:
        print(f"Error while counting tokens: {e}")
        return -1

def analyze_services_individually(smell_data, base_folder_path, user_query):
    """
    Executes the RAG analysis by iterating over each microservice found in the base path.
    """
    # 1. Identify service folders
    try:
        service_folders = [f for f in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, f))]
        # Filter out irrelevant folders like .git, etc.
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

    # 2. Iterate over each service
    for service_name in tqdm(service_folders, desc="Analyzing services"):
        service_path = os.path.join(base_folder_path, service_name)
        print(f"\n--- Analyzing service: {service_name} ---")

        # 3. Load and split documents ONLY for the current service
        source_code_docs = load_folder_path_documents(service_path)
        if not source_code_docs:
            print(f"No source code documents found for service '{service_name}'.")
            continue
            
        code_chunks = get_code_chunks(source_code_docs)
        if not code_chunks:
            print(f"Unable to split source code into chunks for service '{service_name}'.")
            continue

        # 4. Create a TEMPORARY and ISOLATED vector store for the current service
        print(f"Creating temporary vector store for '{service_name}'...")
        # NOTE: from_documents is more direct if you don't need to manage batches manually
        service_vectorstore = FAISS.from_documents(code_chunks, embeddings_model)
        print(f"Vector store for '{service_name}' created successfully.")


        # 5. Perform similarity search on this isolated vector store
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

        # Add snippets to the global list, avoiding duplicate content
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
                    print(f"No document loaded from {filename}")
                    continue

                code_chunks = get_code_chunks(docs)
                if not code_chunks:
                    print(f"No chunk generated from {filename}")
                    continue

                print(f"Creating vector store for '{filename}'...")
                file_vectorstore = FAISS.from_documents(code_chunks, embeddings_model)

                search_queries = [ex['negative_example'] for ex in smell_data.get('manifestations', [])]
                if not search_queries:
                    print("No 'negative_example' in the KB. Search skipped.")
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
    # --- End of new loop ---

    if not all_retrieved_snippets_from_all_services:
        print("\nNo code snippets similar to the examples were found in the services. The code is probably clean for this smell.")
        return

    print(f"\n_Found a total of {len(all_retrieved_snippets_from_all_services)} potentially suspicious code snippets to analyze.")

    # The rest of the code to format the prompt and call the LLM remains almost unchanged
    smell_definition = f"Description: {smell_data['brief_description']}"
    positive_examples = "\n\n".join(
        [f"--- Positive Example ({ex['language']}) ---\n{ex['positive_example']}\nExplanation: {ex['explanation']}" for ex in smell_data.get('positive', [])]
    ) if 'positive' in smell_data else "No positive examples available."

    code_context_for_prompt = "\n\n".join(
        [f"--- Snippet from file: {doc.metadata.get('source', 'Unknown')} ---\n```\n{doc.page_content}\n```" for doc in all_retrieved_snippets_from_all_services]
    )
    
    final_prompt_string = prompt_template.format(
        smell_definition=smell_definition,
        positive_examples=positive_examples,
        additional_folder_context=code_context_for_prompt,
        smell_specific_instructions=SMELL_INSTRUCTIONS.get(user_query.lower(), "No specific instructions available for this smell.")
    ).replace("[Smell Name]", user_query)

    print("\n--- Final Prompt (before sending to LLM) ---")
    print(final_prompt_string) # Uncomment for debug
    print(f"(Prompt length: {len(final_prompt_string)} characters)")

    token_count = count_tokens_for_llm_input(final_prompt_string, llm)
    if token_count != -1:
        print(f"Estimated number of tokens for LLM input: {token_count}")

    print("\nRequest to LLM in progress...")
    try:
        response = llm.invoke(final_prompt_string)
        answer = response.content
        print("\n--- LLM Response ---")
        print(answer)
    except Exception as e:
        print(f"Error while invoking the LLM: {e}")
        return # Exit the function in case of error

    # Evaluation logic
    ground_truth = {
        "customers-service": ["no api gateway"],
        "accounts-service": ["no api gateway"],
        "transactions-service": ["no api gateway"],       #check if WSI is correct
        "customers-view-service": ["shared persistence", "no api gateway"],
        "accounts-view-service": ["shared persistence", "no api gateway"],
        "api-gateway-service": ["shared persistence", "endpoint based service interaction"]   
        #the smell is found in cummon-auth used by api gateway service
        #the gateway depends on fragile addresses (application.properties)
    }


    smell_name = user_query.lower()
    predicted_services = extract_services_from_llm_output(answer)
    print("\n\n>>> Predicted services:", predicted_services)

    predicted = {(os.path.basename(s), smell_name) for s in predicted_services}
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

    # Load the smell
    smell_data = load_smell_data(user_query, knowledge_base_dir)
    if not smell_data:
        continue # Ask for new input if the smell does not exist

    # Input of the base folder containing the microservices to analyze
    folder_path_input = input("Specify the path of the base folder containing the microservices to analyze: ").strip()
    if not (folder_path_input and os.path.isdir(folder_path_input)):
        print("Invalid or empty folder path. Try again.")
        continue

    # Call the new analysis function that handles the whole process
    analyze_services_individually(smell_data, folder_path_input, user_query)