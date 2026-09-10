# Knowledge-Graph Based Augmentation versus Retrieval Augmented Generation for Cultural-Related Question Answering

## Overview

Large language models (LLMs) suffer from a long-tail deficit: culturally specific facts, particularly those concerning underrepresented regions such as Latin America, appear too rarely in pretraining corpora to be reliably memorized. Retrieval-Augmented Generation (RAG) addresses this by grounding generation in external text, but structured alternatives such as Knowledge Graphs (KGs) offer tighter control over what enters the context, along with potential gains in explainability and updatability. We benchmark Graph-RAG against standard RAG on LatamQA, a culturally grounded multiple-choice dataset spanning eight thematic categories. The graphs are built end-to-end from Wikipedia articles with KGGen, a recent open-domain extractor, without manual curation in our main setting. G-Retriever is competitive with RAG and reduces the error of the base LLM by 72\% with a standard KG and 78\% with a benchmark-aware variant, the gap to RAG narrowing further as the graph is oriented toward task-relevant content. The trained projection transfers zero-shot to Portuguese without target-language fine-tuning, indicating multilingual reach.

## Paper

## Setup

```bash
conda env create -f environment.yml
conda activate kg_rag_env
```

Experiments were conducted on a system equipped with two NVIDIA RTX 3090 GPUs.

## Data

## Reproducing the experiments

### 1. Generation of Knowledge Graph via KG-GEN

### 2. LLM evaluation via G-Retriever & RAG

## Repository structure

```
.
├── `.vscode/` – VS Code settings
├── `CODES/` – notebooks and scripts
│   ├── `EXTRACT_SUBSETS.ipynb` – crawls Wikipedia for themed subsets
│   ├── `G-RETRIEVER/` – evaluation/training experiments
│   │   ├── `RUN_G-RETRIEVER_GNN_MODULE.ipynb` – KG + GNN training/eval
│   │   ├── `RUN_G-RETRIEVER_LINEAR_MODULE.ipynb` – linear module k-fold CV
│   │   ├── `RUN_MODES.py` – runs all modes across categories
│   │   └── `src/` – G-Retriever utilities
│   ├── `KG-GEN/` – knowledge graph generation assets
│   │   ├── `KG_GEN.ipynb` – pipeline for extraction, clustering, provenance
│   │   ├── `KGGEN_API_RELATIONS.ipynb` – KGGen with relation/entity hints
│   │   └── `src/` – KGGen helpers
│   └── `QUESTIONS_TO_RELATIONS.ipynb` – extracts relations/entities from MCQs
├── `DATA/` – datasets, subsets, graphs, indexes
│   ├── `ARTICLES/` – raw article CSVs (ES/PT)
│   ├── `QUESTIONS/` – MCQ banks for both languages
│   ├── `RAG_INDEX/` – built RAG indexes per category
│   ├── `SUBSETS_ES/` – Spanish subsets and graph artifacts
│   │   ├── `ARTICLES_SUBSETS_ES/`
│   │   └── `GRAPHS/`
│   ├── `SUBSETS_PT/` – Portuguese subset data
│   │   ├── `ARTICLES_SUBSETS_PT/`
│   │   └── `GRAPHS/`
│   └── `SUBSETS_with_relations/` – labeled subsets plus hints/graphs
│       ├── `ARTICLES_SUBSETS_ES/`
│       ├── `ENTITES_EXTRAITES/`
│       ├── `GRAPHS/`
│       └── `RELATIONS_EXTRAITES/`
├── `environment.yml` – conda environment spec
├── `README.md` – this overview
└── `requirements.txt` – pip dependencies
```
