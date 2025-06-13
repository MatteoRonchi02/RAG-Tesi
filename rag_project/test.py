import os
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)

LOADER_MAPPING = {
    ".txt": (TextLoader, {"encoding": "utf-8"}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".pdf": (UnstructuredPDFLoader, {}),
    ".java": (TextLoader, {"encoding": "utf-8"}),
    ".js": (TextLoader, {"encoding": "utf-8"})
}

while True:
    folder_path_input = input("Vuoi specificare un percorso a una cartella per contesto aggiuntivo (es. codice sorgente)? Lascia vuoto per saltare: ").strip()

    def load_folder_path_documents(directory: str) -> list[Document]:

        all_documents = []
        print(f"Caricamento documenti aggiuntivi da: {directory}")
        for root, _, files in os.walk(directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                ext = os.path.splitext(filename)[-1].lower()
                
                if ext in LOADER_MAPPING:
                    loader_class, loader_kwargs = LOADER_MAPPING[ext]
                    print(f"  Caricamento: {file_path} (tipo: {ext})")
                    try:
                        # Per i file da folder_path, non applichiamo lo split_by_sections qui,
                        # carichiamo l'intero contenuto del file come un singolo Document.
                        # Questo è intenzionale per fornire il contesto completo del file come richiesto.
                        loader = loader_class(file_path, **loader_kwargs)
                        docs = loader.load() # docs sarà una lista, solitamente con un solo Document per file
                        if docs:
                            # Aggiungiamo il percorso completo come 'source' per chiarezza
                            for doc in docs:
                                doc.metadata["source"] = file_path 
                            all_documents.extend(docs)
                    except Exception as e:
                        print(f"    Errore durante il caricamento di {file_path}: {e}")
        print(f"Totale documenti caricati da {directory}: {len(all_documents)}\n")
        return all_documents

    print(f"\nCaricamento contesto dalla cartella: {folder_path_input}")
    # ATTENZIONE: questo carica interi file. Per grandi codebase, considera strategie di chunking o selezione.
    folder_documents = load_folder_path_documents(folder_path_input)
    if folder_documents:
        additional_folder_context_str = "\n\n".join([f"--- Contenuto dal file: {doc.metadata.get('source', 'Sconosciuto')} ---\n{doc.page_content}" for doc in folder_documents])
        print(f"Contesto da {len(folder_documents)} file caricato dalla cartella.")