# MMEL-RAG – Question/Réponse sur la réglementation aérienne (EASA / CS-25)

Ce dépôt contient un petit projet de **Retrieval-Augmented Generation (RAG)** appliqué à des documents réglementaires aéronautiques (CS-25, MMEL, AMC, etc.).  
L'idée est de pouvoir poser des questions métier en langage naturel et d'obtenir une réponse appuyée sur les textes officiels (PDF).

Le projet est structuré comme un vrai **package Python** (`mmel_rag`) avec :
- un pipeline d'indexation de PDFs,
- plusieurs stratégies de **recherche** (BM25, vecteur, hybride),
- un script d'**évaluation** sur un dev set annoté,
- quelques **tests unitaires**.

> Projet perso réalisé en M1 Informatique à l'ENSEEIHT (Toulouse) dans le cadre d'une recherche d'alternance / stage en IA .

---

## 1. Fonctionnalités principales

-  **Recherche hybride sur PDF réglementaires**
  - Recherche dense (embeddings OpenAI + FAISS)
  - BM25 avec traduction FR→EN et petites heuristiques de réécriture de requête
  - Fusion par **Reciprocal Rank Fusion (RRF)**

-  **Pipeline PDF adapté aux docs de réglementation**
  - Chargement page par page
  - Nettoyage texte (espaces, césures, ligatures…)
  - Filtrage des pages de **table des matières** (sommaires)

-  **Découpage en chunks optimisé**
  - `CHUNK_SIZE = 700`, `CHUNK_OVERLAP = 120`
  - Ce choix vient de plusieurs essais sur un dev set (~50 questions) : 700 caractères donnait le meilleur compromis rappel / bruit.

-  **Évaluation intégrée**
  - Dev set au format CSV (question + méta sur la "bonne" page)
  - Métriques de retrieval : `hit@1`, `hit@3`, `hit@5`, `MRR`, `nDCG@5`
  - Export en CSV pour analyser les erreurs.

-  **API Q/R simple + CLI**
  - Fonction Python `answer_question(...)`
  - Ligne de commande : `python -m mmel_rag -q "..." --mode hybrid`

-  **Tests unitaires**
  - Vérification de la config / des fichiers
  - Tests sur la détection des sommaires
  - Tests sur la déduplication de pages et la normalisation des références CS/AMC

---

## 2. Structure du projet

```text
.
├── mmel_rag/
│   ├── __init__.py        # API publique du package
│   ├── __main__.py        # Entrée CLI : python -m mmel_rag
│   ├── config.py          # Chemins, modèles, taille de chunks…
│   ├── indexing.py        # Chargement PDF, nettoyage, split, filtrage TOC
│   ├── retrieval.py       # BM25 / vecteur / hybride + RRF
│   ├── evaluation.py      # Chargement dev set + métriques
│   └── qa.py              # Pipeline Q/R au-dessus de la récupération
├── tests/
│   ├── test_config.py
│   ├── test_indexing.py
│   └── test_retrieval.py
├── Docs/
│   ├── <reglement_1>.pdf
│   ├── <reglement_2>.pdf
│   └── devset_ready.csv   # questions annotées pour l'évaluation
├── requirements.txt
└── README.md
```

---

## 3. Installation

### Prérequis

- Python 3.8+
- Clés API : OpenAI, LangChain 

### Étapes d'installation

**1. Cloner le dépôt et installer les dépendances :**

```bash
git clone <url_du_depot>
cd mmel-rag
pip install -r requirements.txt
```

**2. Créer un fichier `.env` à la racine du projet :**

```env
# Clés API obligatoires
OPENAI_API_KEY=sk-...

# Optionnel : pour le tracing LangSmith
LANGCHAIN_API_KEY=lsv2_pt_...

# Optionnel : personnaliser les chemins
MMEL_RAG_DOCS_DIR=./Docs
MMEL_RAG_EVAL_DIR=./artifacts/eval
MMEL_RAG_DEVSET_CSV=./Docs/devset_ready.csv

# Optionnel : choisir les modèles
MMEL_RAG_EMBED_MODEL=text-embedding-3-large
MMEL_RAG_TRANSLATION_MODEL=gpt-4o
MMEL_RAG_QA_MODEL=gpt-4o-mini
```

**3. Organiser vos documents dans le dossier `Docs/` :**

```
Docs/
├── CS-25 (Amendment 27).pdf      
├── AMC-25 (Amendment 27).pdf     
└── devset_ready.csv              
```

> **Note :** Le système charge automatiquement tous les fichiers `.pdf` du dossier `Docs/` au démarrage.

---

## 4. Format du dev set (`devset_ready.csv`)

Le fichier CSV doit contenir les colonnes suivantes :

| Colonne | Description | Exemple |
|---------|-------------|---------|
| `q` | Question en langage naturel | `"What are the requirements for emergency exits?"` |
| `gold` | Pages de référence (format `source:page`) séparées par `;` | `"Docs/CS-25 (Amendment 27).pdf:430;Docs/AMC-25.pdf:215"` |
| `must_include` | Mots-clés obligatoires dans les résultats (séparés par `\|`) | `"emergency\|exit\|evacuation"` |
| `lang` | Langue de la question | `en` ou `fr` |

---

## 5. Utilisation

### 5.1 Poser une question (CLI)

```bash
# Question simple avec mode hybride (recommandé)
python -m mmel_rag -q "Quelles sont les exigences sur les freins d'atterrissage?"

# Choisir un mode de recherche spécifique
python -m mmel_rag -q "Emergency exit requirements" --mode vector

# Ajuster le nombre de documents récupérés
python -m mmel_rag -q "Landing gear stability" --mode hybrid -k 8
```

**Modes de recherche disponibles :**
- `vector` : Recherche sémantique pure (FAISS + OpenAI embeddings)
- `bm25translate` : Recherche lexicale avec traduction FR→EN
- `hybrid` : Fusion RRF des deux approches (**recommandé**)

### 5.2 Lancer l'évaluation

```bash
# Évaluer tous les modes sur le dev set
python -m mmel_rag --eval
```

**Sortie attendue :**

```
— RÉSUMÉ —
vector         | hit@1=0.760 | hit@5=0.960 | MRR=0.839 | nDCG@5=0.776
bm25translate  | hit@1=0.860 | hit@5=0.980 | MRR=0.909 | nDCG@5=0.851
hybrid_rrf     | hit@1=0.760 | hit@5=1.000 | MRR=0.860 | nDCG@5=0.796
```


### 5.3 Utilisation programmatique (Python)

```python
from mmel_rag import answer_question, get_docs

# Obtenir une réponse complète
answer = answer_question(
    "Quelles sont les exigences de stabilité longitudinale?",
    mode="hybrid",
    k=5
)
print(answer)

# Récupérer seulement les documents pertinents (sans générer de réponse)
docs = get_docs(
    "Landing gear requirements CS-25.1309",
    mode="vector",
    k=3
)

for doc in docs:
    print(f"[{doc.metadata['source']} p.{doc.metadata['page']}]")
    print(doc.page_content[:200])
    print()
```

---

## 6. Configuration avancée

### 6.1 Paramètres de chunking (`config.py`)

```python
CHUNK_SIZE = 700        # Taille des chunks en caractères
CHUNK_OVERLAP = 120     # Recouvrement entre chunks consécutifs
```

> **Choix empirique :** Ces valeurs ont été optimisées sur le dev set. Un chunk trop petit (< 500) perdait du contexte, trop grand (> 1000) introduisait du bruit.

### 6.2 Paramètres de recherche vectorielle (`retrieval.py`)

```python
retriever_vec = vs.as_retriever(
    search_type="mmr",           # Maximal Marginal Relevance
    search_kwargs={
        "k": 8,                  # Nombre de résultats finaux
        "fetch_k": 80,           # Taille du pool avant MMR (↑ rappel)
        "lambda_mult": 0.7,      # Balance similarité/diversité (0=diversité, 1=similarité)
    }
)
```

### 6.3 Expansion de termes BM25 (`retrieval.py`)

Le dictionnaire `EXPAND` ajoute des synonymes pour améliorer le recall :

```python
EXPAND = {
    "frein": ["brake", "braking"],
    "train d'atterrissage": ["landing gear", "undercarriage"],
    "stabilité": ["stability", "longitudinal stability"],
    # ... ajoutez vos propres mappings
}
```

---

## 7. Résultats attendus

### 7.1 Métriques de référence (sur ~50 questions)

| Métrique | Vector | BM25 Translate | **Hybrid (RRF)** |
|----------|--------|----------------|------------------|
| hit@1    | 0.760  | 0.860          | **0.760**        |
| hit@5    | 0.960  | 0.980          | **1.000**        |
| MRR      | 0.839  | 0.909          | **0.860**        |
| nDCG@5   | 0.776  | 0.851          | **0.796**        |


> **Note :** En pratique, le mode `bm25translate` est un très bon choix par défaut pour ce corpus très textuel et normé (CS-25 / AMC). Le mode `hybrid` reste intéressant si l'on privilégie le rappel (retrouver au moins un bon document dans les k premiers).


### 7.2 Interprétation des métriques

- **hit@k** : Proportion de questions où au moins un document pertinent apparaît dans les k premiers résultats
- **MRR** (Mean Reciprocal Rank) : 1/(rang du premier document pertinent), moyenné sur toutes les questions
- **nDCG@5** (Normalized Discounted Cumulative Gain) : Mesure la qualité du classement en pénalisant les documents pertinents mal positionnés

### 7.3 Tolérance de page

L'évaluation utilise une **tolérance de ±3 pages** : un document est considéré comme pertinent s'il est à ±3 pages de la référence gold (pour gérer les décalages de numérotation PDF).

---

## 8. Tests

### 8.1 Vérifier la configuration

```bash
python -m mmel_rag.tests.test_config
```

Vérifie :
- Présence des clés API (`OPENAI_API_KEY`, `LANGCHAIN_API_KEY`)
- Existence du dossier `Docs/` avec 2 PDFs
- Existence du fichier `devset_ready.csv`

### 8.2 Lancer les tests unitaires

```bash
# Tous les tests
pytest mmel_rag/tests/ -v

# Tests spécifiques
pytest mmel_rag/tests/test_indexing.py -v
pytest mmel_rag/tests/test_retrieval.py -v
```

