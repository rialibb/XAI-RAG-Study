from XAI_System.models import (retrieve_with_contriever,
                    generate_llm_response, 
                    detect_hallucination,
                    analyze_context_influence,
                    extract_rationale)
from XAI_System.preprocessing import preprocess_pdf







def run_xai_system(pdf_path,
                   query,
                   k=5,
                   approach = 'cosine'): # approch in ['cosine', 'svm']
    
    # Split PDF to chunks
    chunks = preprocess_pdf(pdf_path)
    
    # Extract context
    retrieved = retrieve_with_contriever(query, chunks, k=k, approach=approach)   
    contexts = [c for c, _ in retrieved]
    print('-'*90)
    print('-'*40 + 'CONTEXTS' + '-'*40)
    print('-'*90)
    for i in range(len(contexts)):
        print(retrieved[i], '\n')
    
    # Generate response
    response = generate_llm_response(query, contexts)
    print('-'*90)
    print('-'*35 +'GENERATING RESPONSE'+ '-'*35)
    print('-'*90)
    print(response)
    
    # Detect hallucination
    data={'question':query,
        'reference':contexts,
        'response':response}
    halu = detect_hallucination(data)
    print('-'*90)
    print('-'*33 +'Hallucination Detection'+ '-'*33)
    print('-'*90)
    print(halu)
    
    # Ablation testing (Context Sensitivity via Perturbation)
    print('-'*90)
    print('-'*35 +'Ablation Testing'+ '-'*35)
    print('-'*90)
    analyze_context_influence(query, contexts, response)
    
    # Self-Rationale Extraction (Chain-of-Thought Prompting)
    print('-'*90)
    print('-'*30 +"EXTRACTING RATIONALE FROM CONTEXT"+ '-'*30)
    print('-'*90)
    rationale_output = extract_rationale(query, contexts, response)
    print(rationale_output)








