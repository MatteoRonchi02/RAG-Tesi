import os

file_path = os.path.abspath("knowledge_base/security_guidelines.md")

try:
    with open(file_path, "r") as file:
        content = file.read()
    print("✅ Il file è stato letto correttamente!")
    print(content[:200])  # Stampa i primi 200 caratteri
except Exception as e:
    print(f"❌ Errore nel leggere il file: {e}")
