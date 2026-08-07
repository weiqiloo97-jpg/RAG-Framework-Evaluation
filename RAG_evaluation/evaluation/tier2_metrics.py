import json
import re
import random
import numpy as np
from generation.answer_generator import LLMWrapper
from evaluation.tier1_metrics import is_relevant

def parse_llm_score(response: str, default: float = 0.5) -> float:
    """
    Safely extracts a float score from the LLM output.
    First tries to parse JSON containing a 'score' key.
    Falls back to regex searching for a decimal or integer between 0 and 1.
    """
    try:
        # Search for JSON block
        json_match = re.search(r"\{.*?\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if "score" in data:
                return float(data["score"])
    except Exception:
        pass
        
    # Regex fallback for numbers like 0.8, 1.0, 0, 1
    numbers = re.findall(r"\b0\.\d+|\b1\.0|\b0\b|\b1\b", response)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            pass
            
    return default

def parse_relevance_array(response: str, expected_len: int) -> list:
    """
    Parses a relevance boolean array from the LLM response.
    """
    try:
        json_match = re.search(r"\{.*?\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if "relevance" in data:
                res = [bool(x) for x in data["relevance"]]
                if len(res) < expected_len:
                    res += [False] * (expected_len - len(res))
                return res[:expected_len]
    except Exception:
        pass
        
    res = []
    lines = response.split("\n")
    for line in lines:
        if "yes" in line.lower() or "true" in line.lower():
            res.append(True)
        elif "no" in line.lower() or "false" in line.lower():
            res.append(False)
            
    if len(res) < expected_len:
        res += [False] * (expected_len - len(res))
    return res[:expected_len]

def compute_average_precision(relevance: list) -> float:
    """
    Computes Average Precision (AP) for context precision.
    AP = sum(Precision@k * relevance_k) / sum(relevance_k)
    """
    if not relevance or not any(relevance):
        return 0.0
        
    precision_sums = 0.0
    relevant_so_far = 0
    
    for i, rel in enumerate(relevance):
        if rel:
            relevant_so_far += 1
            precision_at_k = relevant_so_far / (i + 1)
            precision_sums += precision_at_k
            
    return precision_sums / relevant_so_far

def introduce_typos(text: str) -> str:
    """
    Programmatically introduces spelling errors/typos into query text.
    Swaps letters, deletes letters, or doubles letters randomly.
    """
    words = text.split()
    if len(words) < 2:
        return text
        
    eligible = [i for i, w in enumerate(words) if len(w) > 4]
    if not eligible:
        return text
        
    to_perturb = random.sample(eligible, min(len(eligible), 2))
    for idx in to_perturb:
        word = list(words[idx])
        choice = random.choice(["swap", "delete", "double"])
        pos = random.randint(1, len(word) - 2)
        
        if choice == "swap" and pos < len(word) - 1:
            word[pos], word[pos+1] = word[pos+1], word[pos]
        elif choice == "delete":
            word.pop(pos)
        elif choice == "double":
            word.insert(pos, word[pos])
            
        words[idx] = "".join(word)
        
    return " ".join(words)


# --- Heuristic Fallback Metric Implementations ---

def heuristic_faithfulness(answer: str, context_docs: list) -> float:
    """
    Computes faithfulness by checking the ratio of answer content terms present in the context.
    If the model states it has no information, faithfulness is 1.0.
    """
    ans_lower = answer.lower()
    if any(phrase in ans_lower for phrase in ["don't have", "not have", "not enough", "don't know", "could not find", "no information"]):
        return 1.0
        
    ans_words = set(re.findall(r'\b[a-z0-9]{4,}\b', ans_lower))
    stop_words = {"based", "context", "synthesized", "answer", "movie", "film", "features", "members", "document", "provided", "provides"}
    ans_words = ans_words - stop_words
    
    if not ans_words:
        return 1.0
        
    context_text = " ".join([d["text"].lower() for d in context_docs])
    matched = sum(1 for w in ans_words if w in context_text)
    return float(matched / len(ans_words))

def heuristic_answer_relevancy(query: str, answer: str, embed_model) -> float:
    """
    Computes answer relevancy by measuring cosine embedding similarity between the query and generated answer.
    """
    if not answer or not embed_model:
        return 0.0
    try:
        q_emb = embed_model.encode([query])[0]
        a_emb = embed_model.encode([answer])[0]
        dot_product = np.dot(q_emb, a_emb)
        norm_q = np.linalg.norm(q_emb)
        norm_a = np.linalg.norm(a_emb)
        if norm_q == 0 or norm_a == 0:
            return 0.0
        return float(dot_product / (norm_q * norm_a))
    except Exception:
        return 0.5

def heuristic_context_recall(ground_truth: str, context_docs: list) -> float:
    """
    Computes context recall by measuring the ratio of ground truth content terms present in the retrieved context.
    """
    gt_lower = ground_truth.lower()
    gt_words = set(re.findall(r'\b[a-z0-9]{4,}\b', gt_lower))
    stop_words = {"directed", "director", "features", "released", "movie", "film", "short", "genre", "cast", "written"}
    gt_words = gt_words - stop_words
    
    if not gt_words:
        return 1.0
        
    context_text = " ".join([d["text"].lower() for d in context_docs])
    matched = sum(1 for w in gt_words if w in context_text)
    return float(matched / len(gt_words))

def heuristic_noise_sensitivity(retrieved_docs: list, ground_truth_docs: list, answer: str, query: str) -> float:
    """
    Computes noise sensitivity by checking if the generated answer includes words unique to irrelevant (noise) documents.
    """
    noise_docs = [doc for doc in retrieved_docs if not is_relevant(doc, ground_truth_docs)]
    if not noise_docs or len(noise_docs) == len(retrieved_docs):
        return 0.0
        
    relevant_docs = [doc for doc in retrieved_docs if is_relevant(doc, ground_truth_docs)]
    
    noise_words = set()
    for d in noise_docs:
        noise_words.update(re.findall(r'\b[a-z0-9]{4,}\b', d["text"].lower()))
        
    clean_words = set()
    for d in relevant_docs:
        clean_words.update(re.findall(r'\b[a-z0-9]{4,}\b', d["text"].lower()))
    clean_words.update(re.findall(r'\b[a-z0-9]{4,}\b', query.lower()))
    
    misleading_words = noise_words - clean_words
    if not misleading_words:
        return 0.0
        
    ans_words = set(re.findall(r'\b[a-z0-9]{4,}\b', answer.lower()))
    if not ans_words:
        return 0.0
        
    overlap = ans_words.intersection(misleading_words)
    return float(len(overlap) / len(ans_words))


# --- Main Tier 2 Metrics Orchestrator ---

def compute_tier2_metrics(
    query: str, 
    ground_truth: str, 
    generated_answer: str, 
    retrieved_docs: list, 
    llm: LLMWrapper,
    original_retrieved_ids: list,
    perturbed_retrieved_ids: list,
    embed_model = None,
    fast_mode: bool = False
) -> dict:
    """
    Computes Tier 2A metrics:
    - Faithfulness
    - Answer Relevancy
    - Context Precision (Objective Average Precision of retrieved docs compared to ground truth)
    - Context Recall
    - Noise Sensitivity
    - Robustness (Retrieval ID overlap under perturbed query)
    """
    # 1. Context Precision (Objective metric calculated mathematically against ground truth)
    ground_truth_docs = [ground_truth] # treat the ground truth answer string as a match point
    if not fast_mode:
        relevance_array = [is_relevant(doc, ground_truth_docs) for doc in retrieved_docs]
        context_precision = compute_average_precision(relevance_array)
    else:
        context_precision = 0.0
    
    # 2. Robustness (Objective metric based on query perturbations)
    if not fast_mode:
        orig_set = set(original_retrieved_ids)
        pert_set = set(perturbed_retrieved_ids)
        if orig_set or pert_set:
            robustness = len(orig_set.intersection(pert_set)) / len(orig_set.union(pert_set))
        else:
            robustness = 1.0
    else:
        robustness = 0.0
        
    # Check if we are running under the local fallback generator
    is_fallback = (llm.__class__.__name__ == "FallbackLLM")
    
    # Compute heuristic baselines
    h_faith = heuristic_faithfulness(generated_answer, retrieved_docs)
    h_relev = heuristic_answer_relevancy(query, generated_answer, embed_model)
    
    if not fast_mode:
        h_rec = heuristic_context_recall(ground_truth, retrieved_docs)
        h_noise = heuristic_noise_sensitivity(retrieved_docs, ground_truth_docs, generated_answer, query)
    else:
        h_rec = 0.0
        h_noise = 0.0
    
    if is_fallback:
        # Fallback mode uses pure mathematical heuristics
        return {
            "Faithfulness": round(h_faith, 4),
            "Answer Relevancy": round(h_relev, 4),
            "Context Precision": round(context_precision, 4),
            "Context Recall": round(h_rec, 4),
            "Noise Sensitivity": round(h_noise, 4),
            "Robustness": round(robustness, 4)
        }
        
    # Live LLM evaluation
    context_text = "\n".join([f"Doc {i+1}: {d['text']}" for i, d in enumerate(retrieved_docs)])
    
    # 1. LLM Faithfulness
    faithfulness_prompt = (
        "Analyze the following retrieved documents (Context) and the generated answer.\n"
        "Determine if the statements in the generated answer are fully supported by the Context.\n"
        "Output your response strictly in the following JSON format:\n"
        "{\n"
        '  "reasoning": "your step-by-step reasoning",\n'
        '  "score": 0.9\n'
        "}\n"
        "Where score is a float between 0.0 and 1.0 (1.0 = completely grounded/faithful, 0.0 = completely ungrounded/hallucinated).\n\n"
        f"Context:\n{context_text}\n\n"
        f"Generated Answer:\n{generated_answer}\n"
    )
    faithfulness_resp = llm.generate(faithfulness_prompt)
    llm_faithfulness = parse_llm_score(faithfulness_resp, default=h_faith)
    # Blend LLM rating and Jaccard heuristic to stabilize and prevent flat 1.0/0.0 scores
    faithfulness = 0.5 * llm_faithfulness + 0.5 * h_faith
    
    # 2. LLM Answer Relevancy
    relevancy_prompt = (
        "Evaluate the relevance of the generated answer compared to the user question.\n"
        "Check if the answer directly and concisely addresses the question, without containing irrelevant information.\n"
        "Output your response strictly in the following JSON format:\n"
        "{\n"
        '  "reasoning": "your step-by-step reasoning",\n'
        '  "score": 0.85\n'
        "}\n"
        "Where score is a float between 0.0 and 1.0 (1.0 = highly relevant and direct, 0.0 = completely off-topic or empty).\n\n"
        f"Question:\n{query}\n\n"
        f"Generated Answer:\n{generated_answer}\n"
    )
    relevancy_resp = llm.generate(relevancy_prompt)
    llm_relevancy = parse_llm_score(relevancy_resp, default=h_relev)
    relevancy = 0.5 * llm_relevancy + 0.5 * h_relev
    
    if not fast_mode:
        # 3. LLM Context Recall
        recall_prompt = (
            "Compare the Ground Truth answer and the retrieved documents.\n"
            "Check if the key information and details present in the Ground Truth answer are fully covered by the retrieved documents.\n"
            "Output your response strictly in the following JSON format:\n"
            "{\n"
            '  "reasoning": "your step-by-step reasoning",\n'
            '  "score": 0.95\n'
            "}\n"
            "Where score is a float between 0.0 and 1.0 (1.0 = all details of ground truth are present in the retrieved docs, 0.0 = none are).\n\n"
            f"Ground Truth:\n{ground_truth}\n\n"
            f"Retrieved Documents:\n{context_text}\n"
        )
        recall_resp = llm.generate(recall_prompt)
        llm_context_recall = parse_llm_score(recall_resp, default=h_rec)
        context_recall = 0.5 * llm_context_recall + 0.5 * h_rec
        
        # 4. LLM Noise Sensitivity
        noise_prompt = (
            "Analyze the retrieved documents, the user query, and the generated answer.\n"
            "Identify if there are any documents in the context that are irrelevant to the user query (noise).\n"
            "Determine if the generated answer was misled, corrupted, or confused by the information in those irrelevant documents.\n"
            "Output your response strictly in the following JSON format:\n"
            "{\n"
            '  "reasoning": "your step-by-step reasoning",\n'
            '  "score": 0.1\n'
            "}\n"
            "Where score is a float between 0.0 and 1.0 (0.0 = completely unaffected by noise/correct, 1.0 = heavily misled or corrupted by noise).\n\n"
            f"Query: {query}\n\n"
            f"Context Documents:\n{context_text}\n\n"
            f"Generated Answer:\n{generated_answer}\n"
        )
        noise_resp = llm.generate(noise_prompt)
        llm_noise_sensitivity = parse_llm_score(noise_resp, default=h_noise)
        noise_sensitivity = 0.5 * llm_noise_sensitivity + 0.5 * h_noise
    else:
        context_recall = 0.0
        noise_sensitivity = 0.0
    
    return {
        "Faithfulness": round(faithfulness, 4),
        "Answer Relevancy": round(relevancy, 4),
        "Context Precision": round(context_precision, 4),
        "Context Recall": round(context_recall, 4),
        "Noise Sensitivity": round(noise_sensitivity, 4),
        "Robustness": round(robustness, 4)
    }
