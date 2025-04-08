# 📊 XAI for RAG Systems in Financial Applications

## 🔍 Project Overview

This project aims to **analyze and enhance the explainability of Retrieval-Augmented Generation (RAG) systems**, specifically in **financial domains**. As RAG becomes increasingly used to query large-scale financial documents such as reports and statements, it is critical to ensure:
- **Robust retrieval quality**
- **Minimized hallucinations**
- **Transparent and interpretable answers**

We combine **retrieval benchmarking**, **hallucination detection**, **ablation testing**, and **self-rationale extraction** to deeply understand and improve the behavior of RAG systems.

---

## 🌟 Objectives

- 📌 **Evaluate retrieval effectiveness** of popular retrievers on financial question-answering.
- 📌 **Assess hallucination risk** of generated outputs using trusted benchmarks.
- 📌 **Perform ablation testing** to measure context sensitivity and robustness.
- 📌 **Extract rationales** from model outputs to improve transparency and trust.
- 📌 Ultimately, to help build reliable RAG pipelines for **financial companies** that meet real-world standards in **accuracy, explainability, and reliability**.

---

## 📂 Project Structure

```bash
.
├── main.py                         # Entry point to run the full XAI pipeline
├── XAI_System/                    # Core logic for explainable RAG
│   ├── models.py                  # Retrieval, generation, hallucination detection, ablation, rationale
│   ├── pipeline.py                # Full RAG + XAI workflow orchestration
│   └── preprocessing.py           # PDF text and table extraction for financial documents
│
├── Retrieval_study/              # Retrieval performance benchmarking
│   ├── pipeline.py                # Runs experiments on FinQA and TATQA
│   ├── retrievals.py              # Retrieval strategies (E5, FinBERT, Contriever, FinGPT)
│   └── metrics.py                 # Evaluation metrics (Recall, NDCG, etc.)
│
├── Hallucination_study/          # Hallucination detection evaluation
│   ├── pipeline.py                # Evaluation loop
│   ├── dataset.py                 # Prompt formatting
│   ├── prepare_dataset.py         # Preprocessing from RAGTruth dataset
│   └── dataset/                   # Contains the formatted train/dev/test sets
│
└── requirements.txt               # Dependencies
```

---

## 📙 Datasets

### ✅ **RAGTruth** *(for hallucination detection)*  
- Used in `Hallucination_study`
- Source: [RAGTruth: Evaluating Hallucination in Retrieval-Augmented Generation](https://arxiv.org/abs/2310.03682)
- Covers various task types: `QA`, `Summary`, `Data2txt`

### ✅ **RAGBench** *(for retrieval benchmarking)*  
- Used in `Retrieval_study`
- Source: [RAGBench: Evaluating Retrieval-Augmented Generation](https://huggingface.co/datasets/rungalileo/ragbench)
- We focus on **`finqa`** and **`tatqa`**, which are tailored to financial QA.

---

## 🛠️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full XAI Pipeline
```bash
python main.py
```

This will:
- Extract context from a financial PDF
- Generate an answer using LLMs
- Detect hallucinations
- Run ablation (context sensitivity) testing
- Extract rationale behind the response

---

## 🤖 Models Used

### ✅ Retrieval Models:
- `E5` (`intfloat/e5-large`)
- `Contriever` (`facebook/contriever`)
- `FinBERT` (`yiyanghkust/finbert-tone`)
- `FinGPT` (LoRA over `NousResearch/Llama-2-7b-hf`)

### ✅ Generation & Judging Models:
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- `unsloth/Llama-3.3-70B-Instruct-bnb-4bit` (for hallucination judgment)

---

## 📈 Key Features

### 🔍 Retrieval Study
- Compare retrievers on real-world financial QA data
- Evaluate with `Recall@k`, `NDCG@k`
- Helps select best retriever for financial RAG systems

### ⚠️ Hallucination Study
- Uses fine-grained labels from RAGTruth
- Compares hallucination detection performance across models
- Outputs JSON-formatted hallucination spans

### ⚙️ Ablation Testing
- Tests how removal of individual context pieces changes the generated answer
- Measures semantic similarity via SBERT

### 🧠 Self-Rationale Extraction
- Uses LLM prompting to extract key context spans that justify the model’s answer
- Increases transparency in model decision-making

---

## 📌 Why This Matters

Financial applications demand **high precision**, **explainability**, and **auditability**. A RAG system that can explain itself and avoid hallucinations is essential for:
- Investment advisory tools
- Automated financial reporting
- Corporate intelligence platforms

This project provides both a framework and a benchmark for building **trustworthy** and **interpretable** RAG systems in the financial domain.

---

## 📬 Contact

For questions or collaboration inquiries, feel free to reach out!

