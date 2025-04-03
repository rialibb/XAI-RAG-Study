from models import (retrieve_with_contriever,
                    generate_llm_response, 
                    detect_hallucination,
                    analyze_context_influence,
                    extract_rationale)
from preprocessing import preprocess_pdf






#____________________ TO MODIFY _________________________#
pdf_path = "RAG_system/financial_report.pdf"  # Update with actual file path
k = 5  # context size
query = 'What is the impact of rising interest rates on corporate investments and real estate?'
#query = 'How did the S&P 500 and NASDAQ indices perform in 2023?'
#________________________________________________________#


if __name__ == '__main__':
    
    # Split PDF to chunks
    chunks = preprocess_pdf(pdf_path)
    
    # Extract context
    retrieved = retrieve_with_contriever(query, chunks, k=k, approach='svm')   # approch in ['cosine', 'svm']
    contexts = [c for c, _ in retrieved]
    print('THIS IS THE CONTEXTs:..............')
    for i in range(len(contexts)):
        print(retrieved[i], '\n')
    
    # Generate response
    response = generate_llm_response(query, contexts)
    print('GENERATING RESPONSE:.............')
    print(response)
    
    # Detect hallucination
    data={'question':query,
        'reference':contexts,
        'response':response}
    halu = detect_hallucination(data)
    print('Hallucination Detection:..........')
    print(halu)
    
    # Ablation testing (Context Sensitivity via Perturbation)
    print('Ablation Testing:.................')
    analyze_context_influence(query, contexts, response)
    
    # Self-Rationale Extraction (Chain-of-Thought Prompting)
    print("EXTRACTING RATIONALE FROM CONTEXT.......")
    rationale_output = extract_rationale(query, contexts, response)
    print(rationale_output)








