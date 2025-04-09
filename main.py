from Production_Pipeline.pipeline import run_xai_system
from Retrieval_study.pipeline import run_retrieval_study
from Hallucination_study.pipeline import run_hallucination_study






if __name__ == "__main__":

    # -----------------------------
    # 1. Run Explainable RAG System
    # -----------------------------
    run_xai_system(
        pdf_path="XAI_System/financial_report.pdf",  
        # str: Path to a financial PDF document to process.
        
        query='What is the impact of rising interest rates on corporate investments and real estate?',  
        # str: Natural language question related to the financial document.
        
        k=5,  
        # int: Number of top relevant chunks to retrieve for generation.
        
        approach='cosine'  
        # str: Retrieval strategy to use, must be either 'cosine' or 'svm'.
    )

    # -----------------------------------------
    # 2. Run Retrieval Performance Comparison
    # -----------------------------------------
    run_retrieval_study(
        DATASET_TYPE='tatqa',  
        # str: Financial dataset for evaluation, must be either 'finqa' or 'tatqa'.
        
        retrievals=['E5', 'Contriever', 'FinBERT', 'FinGPT'],  
        # list[str]: One or more retriever model names to benchmark. 
        # Options: 'E5', 'Contriever', 'FinBERT', 'FinGPT'.
        
        k=5  
        # int: Number of top documents to retrieve for scoring (Recall@k, NDCG@k).
    )

    # --------------------------------------
    # 3. Run Hallucination Detection Study
    # --------------------------------------
    run_hallucination_study(
        raw_dataset="./Hallucination_study/dataset/test.jsonl",  
        # str: Path to test set in RAGTruth format (JSONL file).
        
        output_file="./Hallucination_study/dataset/prediction.jsonl",  
        # str: Path to save the hallucination detection predictions (JSONL format).
        
        model_id="unsloth/Llama-3.3-70B-Instruct-bnb-4bit"  
        # str: Hugging Face model ID used as the hallucination judge.
        # Alternatives: 
        #   "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
        #   "unsloth/llama-3-8b-Instruct-bnb-4bit"
        #   "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        #   "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )

    
    
    
    