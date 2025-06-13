import os
import re
import warnings
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from google.cloud import aiplatform
from dotenv import load_dotenv

load_dotenv()
aiplatform.init(
    project=os.getenv("GPC_PROJECT_ID"),
    location= "europe-west1"
)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Ignora i FutureWarning, problema non bloccante ma da risolvere
warnings.filterwarnings("ignore", category=FutureWarning)

#GEMINI_API_KEY_PRIV = os.environ.get("GEMINI_API_KEY_PRIV")
#if not GEMINI_API_KEY_PRIV:
#    raise ValueError("The Gemini API key must be set in the GEMINI_API_KEY_PRIV environment variable.")

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
    ".js": (TextLoader, {"encoding": "utf-8"})
}

# Divide file in sezioni iniziate con #
def split_document_by_sections(file_path: str, encoding="utf-8") -> list[Document]:
    try:
        with open(file_path, "r", encoding=encoding) as f:
            text = f.read()
    except Exception as e:
        print(f"Error in reading the file {file_path}: {e}")
        return []
    
    sections_content = []
    current_section_text = ""
    # Divide per righe; una linea che inizia con '#' viene considerata l'inizio di una nuova sezione
    for line in text.splitlines():
        if line.strip().startswith("#"):
            if current_section_text: # Salva la sezione precedente se esiste
                sections_content.append(current_section_text.strip())
            current_section_text = line + "\n" # Inizia una nuova sezione con l'header
        else:
            current_section_text += line + "\n"
    
    if current_section_text.strip(): # Aggiungi l'ultima sezione
        sections_content.append(current_section_text.strip())
    
    documents = []
    for i, section_text in enumerate(sections_content):
        # serve per avere un header nelle sezioni, non è obbligatorio
        # header_match = re.search(r'^#\s*(.+)', section_text.splitlines()[0] if section_text else "")
        # header = header_match.group(1).strip() if header_match else f"Sezione {i+1}"
        documents.append(Document(page_content=section_text, metadata={"source": file_path})) #, "header": header}))
    return documents

# Carica e processa tutti i documenti dalla knowledge base principale, solo di tipo .txt
def load_knowledge_base_documents(directory: str) -> list[Document]:
    all_documents = []
    print(f"Uploading documents from the knowledge base: {directory}")
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        ext = os.path.splitext(filename)[-1].lower()
        
        if ext == ".txt":
            default_loader_class, default_loader_kwargs = TextLoader, {"encoding": "utf-8"} # Default se .txt non fosse in LOADER_MAPPING (improbabile)
            loader_class, loader_kwargs = LOADER_MAPPING.get(ext, (default_loader_class, default_loader_kwargs) )
            print(f"Loading: {filename}")
            try:
                docs = split_document_by_sections(file_path, encoding=loader_kwargs.get("encoding", "utf-8"))
                if docs:
                    all_documents.extend(docs)
                    print(f"Uploaded {len(docs)} sections from {filename}")
                else:
                    print(f"No document loaded from {filename} (could be empty or splitting error).")

            except Exception as e:
                print(f"Error during loading or processing of {filename}: {e}")
        else:
            print(f"Unsupported file (extension {ext}): {filename}")
            
    print(f"Total sections loaded from the knowledge base: {len(all_documents)}\n")
    return all_documents

#Carica tutti i documenti (interi) da una directory e sottocartelle.
def load_folder_path_documents(directory: str) -> list[Document]:
    all_documents = []
    print(f"Uploading documents from the chosen {directory} path")
    for root, _, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            ext = os.path.splitext(filename)[-1].lower()
            
            if ext in LOADER_MAPPING:
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

# --- CREAZIONE VECTORSTORE E RETRIEVER ---
print("------------Initialization embeddings and vectorstore in progress------------")
kb_documents = load_knowledge_base_documents(knowledge_base_dir)

if not kb_documents:
    print("WARNING: No documents loaded from the knowledge base. RAG may have no context to rely on.")
    # Potresti voler gestire questo caso in modo più specifico, es. uscendo o avvisando chiaramente l'utente.

# Usare try-except per robustezza
try:
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # Solo se ci sono documenti da indicizzare
    if kb_documents:
        vectorstore = FAISS.from_documents(kb_documents, embeddings_model)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}) # Recupera i top 3 chunk dalla KB
        print("---Vectorstore and retriever successfully created from the knowledge base----")
        print("=============================================================================")
    else:
        vectorstore = None
        retriever = None
        print("----Vectorstore not created, there aren't documents in the knowledge base----")
        print("=============================================================================")
except Exception as e:
    raise RuntimeError(f"Error when creating embeddings or vectorstore: {e}")


# --- PROMPT TEMPLATE --- DA RIFORMULARE
prompt_template_str = """Instructions:
1. You are an expert of Security smells in microservices application.
2. The 'Context from Provided Folder' is the source code of a microservice application or a snippet of code.
3. The Question include a Security smell that you will search in the 'Context from Provided Folder'.
3. Combine the information from the Question and the 'Context from Knowledge Base' to analize the 'Context from Provided Folder' (if any) and answer the Question.
4. Your answer should include a list of the services you analyzed and a list of the services affected by the security smell entered in the Question and an explanation of the problem, it is possible that none of the services are affected by the smell.
5. If you identify a security smell relevant to the question, try to show the specific code snippet from the 'Context from Provided Folder' where the smell occurs.
6. If you don't identify ay security smell any analized services, wrote it under the list of the analied service.
7. Remember to deeply analyze ALL files uploaded by the  'Context from Provided Folder', if possible even analyze them more than once.

Question: {query}

--- Context from Knowledge Base ---
{knowledge_base_context}

--- Context from Provided Folder ---
{additional_folder_context}

Answer (in the same language as the Question):"""

prompt_template = PromptTemplate(
    input_variables=["query", "additional_folder_context", "knowledge_base_context"],
    template=prompt_template_str
)

# FUNZIONE PER CONTARE I TOKEN
def count_tokens_for_llm_input(text_input: str, llm_instance: ChatGoogleGenerativeAI) -> int:
    try:
        return llm_instance.get_num_tokens(text_input)
    except Exception as e:
        print(f"Errore durante il conteggio dei token: {e}")
        return -1

print("\n-----------------RAG with Gemini for Security Smell detection----------------")
print("Type the name of the smell you want to do the analysis on (or 'exit' to exit).")

while True:
    user_query = input("Name of Security Smell: ")
    if user_query.lower() in ["exit", "quit", "esci", "stop", "x", "q"]:
        print("Exit from the program.")
        break

    # Recupero contesto dal Path
    additional_folder_context_str = "No additional context from folder provided."
    folder_path_input = input("Specify a folder path for the context to be analyzed: ").strip()
    
    if folder_path_input and os.path.isdir(folder_path_input):
        print(f"\nLoading context from folder: {folder_path_input}")
        folder_documents = load_folder_path_documents(folder_path_input)
        if folder_documents:
            additional_folder_context_str = "\n\n".join([f"--- File content: {doc.metadata.get('source', 'Unknown')} ---\n{doc.page_content}" for doc in folder_documents])
            print(f"Context from {len(folder_documents)} file loaded from the folder.")
        else:
            print("No loadable document found in the specified folder.")
            additional_folder_context_str = "The specified folder contained no loadable documents."
    elif folder_path_input:
        print("Invalid folder path.")

    # Recupero contesto dalla Knowledge Base 
    knowledge_base_context_str = "No context retrieved from the Knowledge Base."
    retrieved_kb_docs_with_similarity = [] # Lista per (doc, percentuale_similarità)

    # Modifichiamo questa sezione per usare vectorstore.similarity_search_with_relevance_scores
    if vectorstore: # Controlla se vectorstore è stato inizializzato
        print(f"\nSearch the Knowledge Base for: '{user_query}'")
        try:
            k_retrieval = 3 
            # Ottieni documenti e punteggi di rilevanza (0-1, più alto è meglio)
            results_with_relevance = vectorstore.similarity_search_with_relevance_scores(
                user_query, 
                k=k_retrieval
            )
            
            if results_with_relevance:
                # Estrai i documenti e calcola la percentuale di similarità
                retrieved_docs_for_context = []
                for doc, relevance_score in results_with_relevance:
                    similarity_percentage = relevance_score * 100
                    retrieved_kb_docs_with_similarity.append((doc, similarity_percentage))
                    retrieved_docs_for_context.append(doc)

                knowledge_base_context_str = "\n\n".join([f"--- Chunk from the KB (Source: {doc.metadata.get('source', 'N/A')}) ---\n{doc.page_content}" for doc in retrieved_docs_for_context])
                print(f"Retrieved {len(retrieved_docs_for_context)} relevant chunks from the Knowledge Base.")
                
                # Mostra snippet con percentuale di similarità
                print("\nRelevant chunks from the Knowledge Base (with similarities):")
                for i, (doc, sim_pct) in enumerate(retrieved_kb_docs_with_similarity):
                    snippet = doc.page_content.strip().replace("\n", " ")[:150] + "..."
                    print(f"  Chunk KB {i+1} (Similarities: {sim_pct:.2f}%)")
                    print(f"    Source: {doc.metadata.get('source', 'N/A')}")
                    print(f"    Snippet: {snippet}\n")
            else:
                print("No relevant chunk found in the Knowledge Base.")
        except Exception as e:
            print(f"Error while retrieving from Knowledge Base: {e}")
            knowledge_base_context_str = "Error while retrieving from Knowledge Base."
    else:
        print("Knowledge Base Vectorstore not available (no initial document or creation error?).")

    # Creazione del prompt finale
    try:
        final_prompt_string = prompt_template.format(
            query=user_query,
            additional_folder_context=additional_folder_context_str,
            knowledge_base_context=knowledge_base_context_str
        )
    except Exception as e:
        print(f"Error while formatting the prompt: {e}")
        continue

    # ("\n--- Prompt Finale (prima dell'invio all'LLM) ---")
    # è commentato perchè è lungo, usarlo come debug
    # print(final_prompt_string) 
    print(f"(Prompt length: {len(final_prompt_string)} characters)")


    # Conteggio dei token (opzionale)
    token_count = count_tokens_for_llm_input(final_prompt_string, llm)
    if token_count != -1:
        print(f"Number of tokens estimated for input to the LLM: {token_count}")

    # 5. Invocazione dell'LLM con il prompt composto
    print("\nRequest to the LLM in progress...")
    try:
        response = llm.invoke(final_prompt_string)
        answer = response.content
        print("\n--- LLM's response ---")
        print(answer)
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        print("Make sure that the LLM model is configured correctly and that the API key is valid and enabled.")