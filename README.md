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
