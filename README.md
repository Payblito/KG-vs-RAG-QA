# Knowledge-Graph Based Augmentation versus Retrieval Augmented Generation for Cultural-Related Question Answering

## Overview

Large language models (LLMs) suffer from a long-tail deficit: culturally specific facts, particularly those concerning underrepresented regions such as Latin America, appear too rarely in pretraining corpora to be reliably memorized. Retrieval-Augmented Generation (RAG) addresses this by grounding generation in external text, but structured alternatives such as Knowledge Graphs (KGs) offer tighter control over what enters the context, along with potential gains in explainability and updatability. We benchmark Graph-RAG against standard RAG on LatamQA, a culturally grounded multiple-choice dataset spanning eight thematic categories. The graphs are built end-to-end from Wikipedia articles with KGGen, a recent open-domain extractor, without manual curation in our main setting. G-Retriever is competitive with RAG and reduces the error of the base LLM by 72\% with a standard KG and 78\% with a benchmark-aware variant, the gap to RAG narrowing further as the graph is oriented toward task-relevant content. The trained projection transfers zero-shot to Portuguese without target-language fine-tuning, indicating multilingual reach.

## Paper

## How to cite

## Setup

```bash
conda env create -f environment.yml
conda activate kg_rag_env
```

Experiments were conducted on a system equipped with two NVIDIA RTX 3090 GPUs.

## Data

## Reproducing the experiments

### 1. Generation of Knowledge Graph via KG-GEN

- `CODES/KG-GEN/KG_GEN.ipynb` – baseline KGGen pipeline that builds graphs without relation/entity hints.
- `CODES/KG-GEN/KGGEN_API_RELATIONS.ipynb` – benchmark-aware version that injects pre-extracted relation and entity hints.

Both notebooks reuse and lightly adapt the original KGGen codebase (Mo et al., 2025), running the generation step with `mistral/mistral-small-2506`; the two variants illustrate the standard vs. hint-guided graph creation paths.

Reference: [KGGen: Extracting Knowledge Graphs from Plain Text with Language Models (Mo et al.,2025)](https://arxiv.org/abs/2502.09956) and the KGGen GitHub repo at [https://github.com/stair-lab/kg-gen/](https://github.com/stair-lab/kg-gen/).

### 2. LLM evaluation via G-Retriever & RAG

Three entry points cover the full evaluation suite:

- `CODES/G-RETRIEVER/RUN_MODES.py` - runs all non-trainable modes across the eight thematic categories: zero-shot (LLM alone), rag (text retrieval over the per-category FAISS index), and top-k-triples (KG triples retrieved by embedding similarity and linearized into the prompt). No training is involved, so this script only performs inference and writes the per-category accuracies.
- `CODES/G-RETRIEVER/RUN_G-RETRIEVER_GNN_MODULE.ipynb` – trains and evaluates G-Retriever with the GNN module (PCST subgraph retrieval → graph encoder → projection into the LLM token space), including the ablation variants reported in the paper.
- `CODES/G-RETRIEVER/RUN_G-RETRIEVER_LINEAR_MODULE.ipynb` – same pipeline, but the graph encoder is replaced by the Linear module (mean-pooled node/edge embeddings projected directly), with the same k-fold cross-validation protocol.

Two configuration files control every run:

- `CODES/G-RETRIEVER/src/config.py` – general parameters: paths to the data folder, language, model names (LLM and text encoder), retrieval settings (top-k, PCST budget), batch size, and evaluation mode.
- `CODES/G-RETRIEVER/src/gnn_config.py` – parameters specific to the trainable modules: GNN architecture (type, number of layers, hidden dimension, heads, dropout), projection dimension, learning rate, number of epochs, and k-fold settings.

Switching datasets only requires changing the data directory in the config: pointing it to `DATA/SUBSETS_ES/` runs the standard Spanish setting, `DATA/SUBSETS_with_relations/` runs the benchmark-aware (hint-guided) graphs, and `DATA/SUBSETS_PT/` runs the zero-shot Portuguese transfer experiment.

These scripts reuse and adapt the original G-Retriever codebase (He et al., 2024), with modifications for multiple-choice evaluation, the Linear module ablation, and the multilingual setting.
Reference: [G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering (He et al., 2024)](https://arxiv.org/abs/2402.07630) and the G-Retriever GitHub repo at [https://github.com/XiaoxinHe/G-Retriever](https://github.com/XiaoxinHe/G-Retriever).

## Repository structure

```
.
├── CODES/                            # Notebooks and scripts
│   ├── EXTRACT_SUBSETS.ipynb         # Crawls Wikipedia for themed subsets
│   ├── G-RETRIEVER/                  # Evaluation/training experiments
│   │   ├── RUN_G-RETRIEVER_GNN_MODULE.ipynb      # KG + GNN training/eval
│   │   ├── RUN_G-RETRIEVER_LINEAR_MODULE.ipynb   # Linear module k-fold CV
│   │   ├── RUN_MODES.py                          # Runs all non-trainable modes across categories
│   │   └── src/                      # G-Retriever utilities
│   ├── KG-GEN/                       # Knowledge graph generation assets
│   │   ├── KG_GEN.ipynb              # KGGen
│   │   ├── KGGEN_API_RELATIONS.ipynb # KGGen with relation/entity hints
│   │   └── src/                      # KGGen helpers
│   └── QUESTIONS_TO_RELATIONS.ipynb  # Extracts relations/entities from MCQs
├── DATA/                             # Datasets, subsets, graphs, indexes
│   ├── ARTICLES/                     # Raw article CSVs (ES/PT)
│   ├── QUESTIONS/                    # MCQ banks for both languages
│   ├── RAG_INDEX/                    # Built RAG indexes per category
│   ├── SUBSETS_ES/                   # Spanish subsets and graph artifacts
│   │   ├── ARTICLES_SUBSETS_ES/
│   │   └── GRAPHS/
│   ├── SUBSETS_PT/                   # Portuguese subset data
│   │   ├── ARTICLES_SUBSETS_PT/
│   │   └── GRAPHS/
│   └── SUBSETS_with_relations/        # Labeled subsets plus hints/graphs
│       ├── ARTICLES_SUBSETS_ES/
│       ├── ENTITES_EXTRAITES/
│       ├── GRAPHS/
│       └── RELATIONS_EXTRAITES/
├── environment.yml                   # Conda environment spec
└── requirements.txt                  # Pip dependencies
```
