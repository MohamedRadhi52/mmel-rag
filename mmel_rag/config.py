from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv   

load_dotenv()
# Racine du projet 
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dossier des PDFs réglementaires 
DOCS_DIR = Path(os.getenv("MMEL_RAG_DOCS_DIR", PROJECT_ROOT / "Docs"))

# Dossier pour les résultats d'évaluation 
EVAL_DIR = Path(os.getenv("MMEL_RAG_EVAL_DIR", PROJECT_ROOT / "artifacts" / "eval"))
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# Chemin du CSV de devset 
DEVSET_CSV = Path(os.getenv("MMEL_RAG_DEVSET_CSV", DOCS_DIR / "devset_ready.csv"))

# Modèles utilisés 
OPENAI_EMBEDDING_MODEL = os.getenv("MMEL_RAG_EMBED_MODEL","text-embedding-3-large",)

TRANSLATION_MODEL = os.getenv("MMEL_RAG_TRANSLATION_MODEL","gpt-4o",)

# CHUNK_SIZE et CHUNK_OVERLAP 
CHUNK_SIZE = 700 # meilleurs résultats observés avec cette valeur
CHUNK_OVERLAP = 120