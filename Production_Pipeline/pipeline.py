from Production_Pipeline.models import (
    extract_metrics,
    retrieve_with_contriever,
    generate_llm_response,
    detect_hallucination,
    analyze_context_influence,
    extract_rationale,
)
from Production_Pipeline.preprocessing import preprocess_pdf
from Production_Pipeline.prompts.faithfulness import faithfulness_prompt
from Production_Pipeline.prompts.utils import JSON_metric_parser
from transformers import pipeline
import torch


def run_xai_system(
    pdf_path, query, k=5, approach="cosine"
):  # approch in ['cosine', 'svm']
    """
    Runs a full explainability pipeline on a financial PDF using a RAG system.

    Args:
        pdf_path (str): Path to the input financial PDF document.
        query (str): The financial question to be answered.
        k (int, optional): Number of top context chunks to retrieve. Default is 5.
        approach (str, optional): Retrieval strategy ('cosine' or 'svm'). Default is 'cosine'.

    Returns:
        None: The function prints all intermediate and final results to the console.

    Process:
        - Extracts and preprocesses content (text and tables) from the PDF.
        - Retrieves the top-k most relevant context chunks using Contriever and the selected similarity approach.
        - Generates a response to the query using an LLM, based strictly on the retrieved context.
        - Detects hallucinations in the generated response using a dedicated LLM judge model.
        - Performs ablation testing by removing one context chunk at a time and evaluating semantic similarity to the original response.
        - Extracts rationale from the context that directly supports the final answer using a structured prompt to an LLM.
    """

    # Split PDF to chunks
    chunks = preprocess_pdf(pdf_path)

    # Extract context
    retrieved = retrieve_with_contriever(query, chunks, k=k, approach=approach)
    contexts = [c for c, _ in retrieved]
    print("-" * 90)
    print("-" * 40 + "CONTEXTS" + "-" * 40)
    print("-" * 90)
    for i in range(len(contexts)):
        print(retrieved[i], "\n")

    # Generate response
    response = generate_llm_response(query, contexts)
    print("-" * 90)
    print("-" * 35 + "GENERATING RESPONSE" + "-" * 35)
    print("-" * 90)
    print(response)

    # Detect hallucination
    data = {"question": query, "reference": contexts, "response": response}
    halu = detect_hallucination(data)
    print("-" * 90)
    print("-" * 33 + "Hallucination Detection" + "-" * 33)
    print("-" * 90)
    print(halu)

    # Ablation testing (Context Sensitivity via Perturbation)
    print("-" * 90)
    print("-" * 35 + "Ablation Testing" + "-" * 35)
    print("-" * 90)
    analyze_context_influence(query, contexts, response)

    # Self-Rationale Extraction (Chain-of-Thought Prompting)
    print("-" * 90)
    print("-" * 30 + "EXTRACTING RATIONALE FROM CONTEXT" + "-" * 30)
    print("-" * 90)
    generator = pipeline(
        "text-generation",
        model="unsloth/Llama-3.3-70B-Instruct-bnb-4bit",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    rationale_output = extract_rationale(generator, query, contexts, response)
    print(rationale_output)
    faithfulness = extract_metrics(
        generator,
        faithfulness_prompt,
        lambda x: JSON_metric_parser(x, ["faithfulness"]),
        query,
        contexts,
        response,
    )
    print(faithfulness)
