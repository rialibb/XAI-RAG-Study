from XAI_System.pipeline import run_xai_system
from Hallucination_study.pipeline import run_hallucination_study









if __name__ == "__main__":
    
    run_xai_system(pdf_path = "XAI_System/financial_report.pdf",
                   query = 'What is the impact of rising interest rates on corporate investments and real estate?',
                   k = 5,
                   approach = 'cosine')
    
    
    
    run_hallucination_study(raw_dataset="./Hallucination_study/dataset/test.jsonl",
                            output_file="./Hallucination_study/dataset/prediction.jsonl",
                            model_id="unsloth/Llama-3.3-70B-Instruct-bnb-4bit")    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"   "unsloth/llama-3-8b-Instruct-bnb-4bit"
    
    
    
    
    
    
    