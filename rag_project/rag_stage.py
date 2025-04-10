import os
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
    model="gemini-2.5-pro-exp-03-25",   
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

# Creazione degli embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_documents(documents, embeddings)


# Creazione del template del prompt
# È un prompt di prova, andrà adattato alle nostre esigienze 
prompt_template = PromptTemplate(
    input_variables=["query", "context"],
    template="Question: {query}"
             "Context: {context}"
            """Instructions:
                1. The Question may include either source code excerpts or a file path pointing to a microservices application.
                2. Analyze the Question carefully and use any provided file path to retrieve and review the relevant source code.
                3. Combine the information from the Question, the Context, and your internal expertise as a security smell expert in microservices applications to generate a detailed and comprehensive response.
                4. Your answer should include a thorough analysis of potential TYPE smells along with practical insights and recommendations.
                5. Answer must have the same language of question.
                6. If the provided Context and file content do not contain sufficient information to deliver an accurate answer, please respond with: "I don't have the necessary information to answer".

            Answer:"""
) #TYPE = scrivi il tipo di smell che ti interessa

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
print("\n RAG with Gemini for Security Smell detection , Language English")
print(" Write your question (or ‘exit’ to exit).")
while True:
    query = input("\nInsert question: ")
    if query.lower() in ["exit", "quit", "esci"]:
        print("Exit program.")
        break
    
    # Traduzione da italiano a inglese
    query_translated = GoogleTranslator(source='auto', target='en').translate(query)

    k = 5  # Numero di chunk da recuperare
    results = vectorstore.similarity_search_with_score(query_translated, k=k)
    
    if results:
        print("\nChunk found with similarity percentage:")
        for i, (doc, score) in enumerate(results, start=1):
            # Se score è una distanza, ipotizziamo che 0 corrisponda al 100% di similarità e 1 a 0%
            similarity_pct = score
            #similarity_pct = max(0, 100 - (score * 100))
            snippet = doc.page_content.strip().replace("\n", " ")
            print(f"--- Chunk {i} ---")
            print(f"Similarity: {similarity_pct:.1f}%")
            print(f"Context: {snippet}\n")
        # Usa il primo chunk come contesto per la risposta
        first_relevant_doc = results[0][0]
    else:
        first_relevant_doc = None
        print("No relevant documents found.\n")

    # Genera la risposta finale con il QA chain
    prompt = generate_prompt(query, first_relevant_doc)

    # Mostra il prompt in output, parte di codice di prova (poi da cancellare)
    #print("\n🔍 Prompt generato:")
    #print(prompt)

    response = qa_chain.invoke({"query": query, "context": prompt}) 
    print("\n Answer:\n", response['result'])
